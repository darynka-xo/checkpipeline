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
        if not text or "Ошибка" in text:
            return text

        model = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model_name="gpt-4o",
            temperature=0.3,
            max_tokens=512
        )

        messages = [
            SystemMessage(
                content=(
                    "Вы — эксперт по нормативной и экономической аналитике.\n"
                    "Сделайте 2–3 очень коротких вывода из приведённого текста, "
                    "каждый снабдите пояснением. В конце ОБЯЗАТЕЛЬНО добавьте "
                    f"строку 'Источник: {source_url}'."
                )
            ),
            HumanMessage(content=text[:3500])  # ограничим, чтобы не выбиться из token-лимита
        ]

        result = model.invoke(messages)
        return result.content.strip()
    except Exception as e:
        return f"[Ошибка анализа: {str(e)}]"


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

        system_message = system_message = """
**Роль:** эксперт по переговорам и стратегическому управлению.  
**Отвечаешь только на темы переговоров, конфликтов интересов, стратегий убеждения.**  
Если информации из внутренних данных недостаточно, СНАЧАЛА вызови инструмент `search_kz_web`
и проанализируй официальные казахстанские источники, затем оформи ответ.

**Структура ответа (сохраняй заголовки):**
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
            for chunk in agent_executor.stream({
                "input": user_message,
                "chat_history": messages
            }):
                output = chunk.get("output")
                if output:
                    logging.debug(f"Agent chunk: {output}")
                    yield output

        return asyncio.run(self.make_request_with_retry(stream_agent))
