import logging
import sys
import os
import asyncio
import requests
from typing import List, Union, Generator, Iterator, Sequence
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool, BaseTool
from langchain.agents import create_tool_calling_agent, AgentExecutor

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class WebSearchInput(BaseModel):
    query: str = Field(description="Запрос для поиска в интернете")

@tool("search_kz_web", args_schema=WebSearchInput, return_direct=False)
def search_kz_web(query: str) -> str:
    """Поиск по казахстанским правовым и государственным источникам через Google."""
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
        url = f"https://www.google.com/search?q={requests.utils.quote(query_with_sites)}"
        return f"Поиск по запросу: {query}\nПроверь результаты по ссылке: {url}"
    except Exception as e:
        return f"Ошибка поиска: {str(e)}"


class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4o"
        TEMPERATURE: float = 0.7
        MAX_TOKENS: int = 1500
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    def __init__(self):
        self.name = "Negotiation Strategy Predictor"
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
**Роль:** Вы — эксперт по переговорам и стратегическому управлению, специализирующийся на прогнозировании успешности различных моделей ведения переговоров и каналов коммуникации.

**Цель:**
1. Анализировать представленные конфликтные или сложные переговорные ситуации.
2. Предлагать эффективные, взвешенные и компромиссные решения.
3. Прогнозировать успешность каждой переговорной модели с точки зрения устойчивости, выгод для сторон и долгосрочного эффекта.

**Дополнительные возможности:**
- Используйте только официальные источники (gov.kz, adilet.zan.kz, egov.kz и т.д.), если нужно искать внешнюю информацию.

**Структура ответа:**

### 1. Возможные компромиссные решения
- Конкретные, практически реализуемые предложения, учитывающие интересы обеих сторон.

### 2. Прогноз эффективности
- Оценка каждой предложенной модели или решения по таким критериям, как: устойчивость, масштабируемость, риски, потенциал реализации.

### 3. Рекомендации
- Какая модель наилучшим образом подойдет в данной ситуации и почему.
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
