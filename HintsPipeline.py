import logging
import sys
import os
import asyncio
from typing import List, Iterator, Callable, Any, Dict
from pydantic import BaseModel
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4o"
        TEMPERATURE: float = 0.7
        MAX_TOKENS: int = 1800
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        SEARCH_API_URL: str = os.getenv("SEARCH_API_URL", "http://localhost:8008/check_and_search")

    def __init__(self):
        self.name = "Подсказки и рекомендации по литературе для законодательных проектов на основе ИИ"
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

    async def call_search_api(self, prompt: str) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    self.valves.SEARCH_API_URL,
                    json={"prompt": prompt, "pipeline": "Hints"}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logging.error(f"Search API error: {e}")
            return {"search_required": False, "context": "", "citations": []}

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Iterator[str]:

        system_message = """
**Роль:** Вы — интеллектуальный помощник для законотворцев и аналитиков, специализирующийся на предоставлении релевантных подсказок, обзоров литературы и сравнительных анализов, основанных на международном опыте, официальных источниках и лучших практиках.

**Цель:**
1. Помочь пользователю найти качественную информацию и рекомендации по теме законодательной инициативы.
2. Составить структурированное и аналитическое обоснование или обзор с учётом казахстанского и международного контекста.
3. Привести ссылки на официальные или научные источники, если они доступны.

**Формат ответа:**
- Введение в тему
- Краткий международный и казахстанский контекст
- Сравнительный анализ (если применимо)
- Практики, рекомендации или подсказки
"""

        search_result = asyncio.run(self.call_search_api(user_message))

        enriched_prompt = user_message
        if search_result["search_required"] and search_result["context"]:
            enriched_prompt += "\n\nКонтекст из найденных источников:\n" + search_result["context"]

        model = ChatOpenAI(
            api_key=self.valves.OPENAI_API_KEY,
            model=self.valves.MODEL_ID,
            temperature=self.valves.TEMPERATURE,
            streaming=True,
            max_tokens=self.valves.MAX_TOKENS
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template("{user_input}")
        ])
        formatted_messages = prompt.format_messages(user_input=enriched_prompt)

        def generate_stream() -> Iterator[str]:
            for chunk in model.stream(formatted_messages):
                content = getattr(chunk, "content", None)
                if content:
                    yield content

            if search_result["search_required"] and search_result["citations"]:
                yield "\n\n### Использованные источники:"
                for i, link in enumerate(search_result["citations"], 1):
                    yield f"\n[{i}] {link}"

        return asyncio.run(self.make_request_with_retry(generate_stream))
