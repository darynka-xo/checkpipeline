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
        TEMPERATURE: float = 0.3
        MAX_TOKENS: int = 1800
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        SEARCH_API_URL: str = os.getenv("SEARCH_API_URL", "http://localhost:8008/check_and_search")

    def __init__(self):
        self.name = "Эксперт по предложениям"
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
                    json={"prompt": prompt, "pipeline": "LawExp"}
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
**Роль:** Вы — ИИ-эксперт по правовой экспертизе законодательства.

**Задача:**
1. Провести анализ предложенного законопроекта или инициативы.
2. Проверить, не дублирует ли он положения действующего законодательства.
3. Выявить существующие нормы, которые уже реализуют те же меры.
4. Дать рекомендации по устранению дублирования или корректировке инициативы.

**Формат ответа:**

### 📄 Краткое содержание инициативы

- [1–2 предложения]

### 🔍 Анализ по блокам
1. **Налоговые меры** — что уже есть в Налоговом кодексе, пересечения.
2. **Гранты и субсидии** — есть ли программы поддержки.
3. **Регистрация бизнеса** — дублирование с Кодексом о предпринимательстве и Egov.

### ⚖️ Заключение

- Есть ли дубли, противоречия или риски.
- Рекомендации для уточнения/согласования.
"""

        search_result = asyncio.run(self.call_search_api(user_message))

        enriched_prompt = user_message
        if search_result["search_required"] and search_result["context"]:
            enriched_prompt += "\n\nКонтекст из найденных официальных источников:\n" + search_result["context"]

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
                yield "\n\n### 📚 Использованные источники:"
                for i, link in enumerate(search_result["citations"], 1):
                    yield f"\n[{i}] {link}"

        return asyncio.run(self.make_request_with_retry(generate_stream))
