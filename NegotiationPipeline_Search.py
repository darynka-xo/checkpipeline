import logging
import sys
import os
import asyncio
import requests
from bs4 import BeautifulSoup
from typing import List, Union, Generator, Iterator, Sequence
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
            "site:adilet.zan.kz",
            "site:gov.kz",
            "site:egov.kz",
            "site:nao.kz",
            "site:nbrk.kz",
            "site:kase.kz",
            "site:primeminister.kz",
            "site:stat.gov.kz"
        ]
        query_with_sites = f"{query} " + " OR ".join(trusted_sites)
        url = f"https://www.google.com/search?q={requests.utils.quote(query_with_sites)}&hl=ru"
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")

        results = []
        for g in soup.select(".tF2Cxc")[:3]:
            title = g.select_one("h3")
            snippet = g.select_one(".VwiC3b")
            link = g.select_one("a")
            if title and snippet and link:
                full_text = extract_text_from_url(link['href'])
                summary = analyze_external_text(full_text)
                results.append(f"🔗 {title.text}\n{snippet.text}\n{link['href']}\n---\n{summary}\n")

        return "Результаты анализа внешних источников:\n\n" + "\n".join(results) if results else "Ничего не найдено на официальных источниках."
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


def analyze_external_text(text: str) -> str:
    try:
        if not text or "Ошибка" in text:
            return text

        model = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model="gpt-4o",
            temperature=0.5
        )

        messages = [
            SystemMessage(content="Вы — эксперт по нормативной и экономической аналитике. Проанализируйте следующий текст и выделите ключевые выводы, которые могут повлиять на стратегию переговоров или инвестиционную политику."),
            HumanMessage(content=text[:4000])
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
        self.name = "Negotiation Strategy Predictor with search"
        self.valves = self.Valves()

    async def on_startup(self):
        logging.info("Pipeline is warming up...")

    async def on_shutdown(self):
        logging.info("Pipeline is shutting down...")

    async def make_request_with_retry(self, fn, retries=3, *args, **kwargs):
        for attempt in range(retries):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt + 1 == retries:
                    raise
                await asyncio.sleep(2 ** attempt)

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:

        system_message = """
**Роль:** Вы — эксперт по переговорам и стратегическому управлению. Ваша специализация — анализ и прогнозирование успешности различных моделей ведения переговоров и коммуникационных стратегий. Вы работаете строго в рамках переговорной аналитики.

**Область ответственности:**
Вы не отвечаете на вопросы, не связанные с анализом переговоров, конфликтов интересов, стратегий убеждения или оценки компромиссных решений. В случае нерелевантного запроса вы мягко уведомляете пользователя и предлагаете сформулировать переговорную ситуацию, в которой вы можете помочь.

**Пример реакции на нерелевантный вопрос:**
> «Прошу прощения, моя компетенция ограничена анализом переговоров и стратегических коммуникаций. Могу ли я помочь вам с анализом конкретной переговорной ситуации или конфликта интересов?»

**Цель:**
1. Анализировать представленные конфликтные или сложные переговорные ситуации.
2. Предлагать эффективные, взвешенные и компромиссные решения.
3. Прогнозировать успешность каждой переговорной модели с точки зрения устойчивости, выгод для сторон и долгосрочного эффекта.

**Дополнительные возможности:**
- Используйте только официальные источники (gov.kz, adilet.zan.kz, egov.kz и т.д.), если нужно искать внешнюю информацию.
- Автоматически анализируйте содержимое внешних источников и формулируйте выводы.
- Если информации недостаточно, предложите уточняющие вопросы пользователю.

**Структура ответа:**

### 1. Возможные компромиссные решения
- Конкретные, практически реализуемые предложения, учитывающие интересы обеих сторон.

### 2. Прогноз эффективности
- Оценка каждой предложенной модели или решения по таким критериям, как: устойчивость, масштабируемость, риски, потенциал реализации.

### 3. Рекомендации
- Какая модель наилучшим образом подойдет в данной ситуации и почему.
- Уточняющие вопросы при необходимости.
"""

        model = ChatOpenAI(
            api_key=self.valves.OPENAI_API_KEY,
            model=self.valves.MODEL_ID,
            temperature=self.valves.TEMPERATURE
        )

        tools: Sequence[BaseTool] = [search_kz_web]

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder("agent_scratchpad")
        ])

        agent = create_tool_calling_agent(model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        response = agent_executor.invoke({
            "input": user_message,
            "chat_history": messages
        })

        return response["output"]