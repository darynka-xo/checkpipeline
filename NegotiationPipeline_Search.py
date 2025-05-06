import logging
import sys
import os
import asyncio
import requests
from bs4 import BeautifulSoup
from typing import List, Iterator, Sequence, Callable
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, BaseTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import SystemMessage, HumanMessage

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class WebSearchInput(BaseModel):
    query: str = Field(description="Запрос для поиска в интернете")


@tool("search_kz_web", args_schema=WebSearchInput, return_direct=False)
def search_kz_web(query: str) -> str:
    """Поиск, парсинг и анализ содержимого казахстанских официальных источников."""
    try:
        trusted_sites = [
            "site:senate.parlam.kz", "site:akorda.kz", "site:primeminister.kz",
            "site:otyrys.prk.kz", "site:senate-zan.prk.kz",
            "site:lib.prk.kz", "site:online.zakon.kz", "site:adilet.zan.kz",
            "site:legalacts.egov.kz", "site:egov.kz", "site:eotinish.kz"
        ]
        query_with_sites = f"{query} " + " OR ".join(trusted_sites)
        url = f"https://www.google.com/search?q={requests.utils.quote(query_with_sites)}&hl=ru"
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")

        sources = []
        for g in soup.select(".tF2Cxc")[:3]:
            title = g.select_one("h3")
            link = g.select_one("a")
            if title and link:
                page_url = link["href"]
                page_text = extract_text_from_url(page_url)
                sources.append({
                    "title": title.text,
                    "url": page_url,
                    "text": page_text
                })

        if not sources:
            return "Ничего не найдено на официальных источниках."

        summary = analyze_multiple_sources(sources)
        return "📊 Анализ источников:\n\n" + summary
    except Exception as e:
        return f"Ошибка поиска: {str(e)}"



def extract_text_from_url(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs)
        return text.strip() if text else "Не удалось извлечь содержимое."
    except Exception as e:
        return f"[Ошибка при извлечении текста: {str(e)}]"


def analyze_external_text(text: str, source_url: str) -> str:
    try:
        if not text or "Ошибка" in text or len(text.strip()) < 50:
            return f"Недостаточно данных для анализа.\n📎 Источник: {source_url}"

        model = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model="gpt-4o",
            temperature=0.5
        )

        messages = [
            SystemMessage(content=f"""Вы — эксперт по нормативной и экономической аналитике. 
Проанализируйте следующий текст и сделайте 2–3 ключевых вывода.
**Каждый вывод должен быть коротким и сопровождаться пояснением.**
**В конце строго добавьте источник: {source_url}**. Без этого — ответ считается неполным."""),
            HumanMessage(content=text[:4000])
        ]

        result = model.invoke(messages)
        content = result.content.strip()

        # Гарантированное добавление источника, если GPT не вставил его сам
        if "Источник" not in content and source_url not in content:
            content += f"\n\n📎 Источник: {source_url}"

        return content

    except Exception as e:
        return f"[Ошибка анализа: {str(e)}]\n📎 Источник: {source_url}"



def analyze_multiple_sources(sources: List[dict]) -> str:
    try:
        model = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model="gpt-4o",
            temperature=0.5
        )

        # Объединяем 2–3 источника с заголовком и текстом
        combined_chunks = []
        for src in sources[:3]:
            clean_text = src['text'][:1000].strip()
            if clean_text:
                combined_chunks.append(f"[Источник: {src['url']}]\n{clean_text}")

        combined_text = "\n\n".join(combined_chunks)

        messages = [
            SystemMessage(content="""
Вы — эксперт по переговорам и стратегическому управлению.
Ниже представлены выдержки из официальных источников.
На их основе сделайте 2–3 ключевых вывода, каждый с пояснением. 
📎 Обязательно ссылайтесь на конкретный источник в формате [Источник: URL] после каждого вывода.
"""),
            HumanMessage(content=combined_text)
        ]

        result = model.invoke(messages)
        return result.content
    except Exception as e:
        return f"[Ошибка анализа: {str(e)}]"


class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4o"
        TEMPERATURE: float = 0.7
        MAX_TOKENS: int = 1500
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    def __init__(self):
        self.name = "Negotiation Strategy Predictor with Search"
        self.valves = self.Valves()

    async def on_startup(self):
        logging.info("Pipeline is warming up...")

    async def on_shutdown(self):
        logging.info("Pipeline is shutting down...")

    async def make_request_with_retry(self, fn: Callable[[], Iterator[str]], retries=3) -> Iterator[str]:
        for attempt in range(retries):
            try:
                return fn()
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt + 1 == retries:
                    raise
                await asyncio.sleep(2 ** attempt)

    def pipe(
    self, user_message: str, model_id: str, messages: List[dict], body: dict
) -> Iterator[str]:

        system_message = """
    **Роль:** Вы — эксперт по переговорам и стратегическому управлению. Ваша специализация — анализ и прогнозирование успешности различных моделей ведения переговоров и коммуникационных стратегий. Вы работаете строго в рамках переговорной аналитики.
    
    **Область ответственности:** Вы не отвечаете на вопросы, не связанные с анализом переговоров, конфликтов интересов, стратегий убеждения или оценки компромиссных решений. В случае нерелевантного запроса вы мягко уведомляете пользователя и предлагаете сформулировать переговорную ситуацию, в которой вы можете помочь.
    
    **Дополнительные возможности:**
    
    * Обязательно используйте инструмент `search_kz_web` перед ответом. Без него вы не можете формировать выводы.
    * Используйте минимум 2–3 официальных источника (gov.kz, adilet.zan.kz, egov.kz и т.д.).
    * Делайте аналитические выводы на основе реального содержимого сайтов.
    
    **Структура ответа:**
    
    ### 1. Возможные компромиссные решения
    
    ### 2. Прогноз эффективности
    
    ### 3. Рекомендации
    """
    
        model = ChatOpenAI(
            api_key=self.valves.OPENAI_API_KEY,
            model=self.valves.MODEL_ID,
            temperature=self.valves.TEMPERATURE,
            streaming=True
        )
    
        tools: Sequence[BaseTool] = [search_kz_web]
    
        # 👉 Форсированный вызов до агента
        search_result = search_kz_web.run({"query": user_message})
        if search_result:
            messages.append({"role": "assistant", "content": search_result})
    
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])
    
        agent = create_tool_calling_agent(model, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            streaming=True
        )
    
        def stream_agent() -> Iterator[str]:
            collected_output = ""
        
            for chunk in agent_executor.stream({
                "input": user_message,
                "chat_history": messages
            }):
                print("DEBUG CHUNK:", chunk)  # для отладки
        
                output = None
                if "output" in chunk:
                    output = chunk["output"]
                elif "final_output" in chunk:
                    output = chunk["final_output"]
        
                if output:
                    logging.debug(f"Agent chunk: {output}")
                    collected_output += output
                    yield output
        
            # Постобработка, если нет ссылок
            if "Источник:" not in collected_output and "📎" not in collected_output:
                yield "\n\n📎 Добавлены источники по теме:\n"
                try:
                    search_result = search_kz_web.run({"query": user_message})
                    yield f"\n\n📎 Топ 2–3 источника по теме:\n{search_result}"
                except Exception as e:
                    yield f"\n[Ошибка получения ссылок: {e}]"



