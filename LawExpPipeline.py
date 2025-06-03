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
    async def inlet(self, body: dict, user: dict) -> dict:
        import json

        logging.info(f"📥 Received inlet body:\n{json.dumps(body, indent=2, ensure_ascii=False)}")
        files = body.get("files", [])
        extracted_texts = []

        for file in files:
            content_url = file["url"] + "/content"
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(content_url)
                response.raise_for_status()
                content = response.content

            mime_type = file.get("mime_type") or mimetypes.guess_type(file.get("name", ""))[0]

            if mime_type == "application/pdf":
                doc = fitz.open(stream=content, filetype="pdf")
                text = "\n".join([page.get_text() for page in doc])
                extracted_texts.append(text)

            elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                with open("temp.docx", "wb") as f:
                    f.write(content)
                text = docx2txt.process("temp.docx")
                os.remove("temp.docx")
                extracted_texts.append(text)

            elif mime_type and mime_type.startswith("image/"):
                image = Image.open(io.BytesIO(content))
                from openai import OpenAI
                client = OpenAI(api_key=self.valves.OPENAI_API_KEY)
                base64_image = base64.b64encode(content).decode("utf-8")
                ocr_response = client.chat.completions.create(
                    model=self.valves.MODEL_ID,
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": "Распознай текст на изображении."},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                        ]}
                    ]
                )
                image_text = ocr_response.choices[0].message.content.strip()
                extracted_texts.append(image_text)

        # Добавляем извлечённый текст
        body["file_text"] = "\n".join(extracted_texts)

        # ✅ Обязательная проверка для OpenWebUI
        if "messages" not in body or not isinstance(body["messages"], list):
            raise ValueError("Field 'messages' is required and must be a list.")
        if "model" not in body or not isinstance(body["model"], str):
            raise ValueError("Field 'model' is required and must be a string.")

        logging.info("✅ inlet completed successfully.")
        return body
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
        file_text = body.get("file_text", "")
        if file_text:
            user_message += f"\n\nТекст из прикреплённых документов:\n{file_text}"
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
        deep_ctx = {}
        if search_result["search_required"] and search_result["citations"]:
            deep_ctx = asyncio.run(
                self.call_search_api(  # 👈 переиспользуем метод, но меняем path
                    user_message.replace(self.valves.SEARCH_API_URL,
                    self.valves.SEARCH_API_URL.replace("/check_and_search",
                    "/deep_extract_and_analyze")),
                )
            )
        enriched_prompt = user_message
        if search_result["search_required"] and search_result["context"]:
            enriched_prompt += "\n\nКонтекст из найденных официальных источников:\n" + search_result["context"]
        if deep_ctx.get("legal_context"):
            enriched_prompt += "\n\n📘 Конкретные нормы и история правок:\n" + deep_ctx["legal_context"]

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
