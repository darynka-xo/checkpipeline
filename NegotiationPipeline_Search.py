import logging
import sys
import os
import asyncio
from typing import List, Iterator, Callable, Any, Dict
from pydantic import BaseModel
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4o"
        TEMPERATURE: float = 0.7
        MAX_TOKENS: int = 1500
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        SEARCH_API_URL: str = os.getenv("SEARCH_API_URL", "http://localhost:8008/check_and_search")

    def __init__(self):
        self.name = "Negotiation Strategy Predictor with Evidence"
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
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(
                    self.valves.SEARCH_API_URL,
                    json={"prompt": prompt, "pipeline": "NegotiationPipeline"}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logging.error(f"Search API error: {e}")
            return {"search_required": False, "context": "", "citations": []}

    async def call_deep_extract_api(self, prompt: str, citations: List[str]) -> str:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    self.valves.SEARCH_API_URL.replace("/check_and_search", "/deep_extract_and_analyze"),
                    json={"prompt": prompt, "citations": citations, "pipeline": "NegotiationPipeline"}
                )
                response.raise_for_status()
                return response.json().get("legal_context", "")
        except Exception as e:
            logging.error(f"Deep extract API error: {e}")
            return ""

    async def inlet(self, body: dict, user: dict) -> dict:
        import json
        import base64
        import mimetypes
        import io
        import fitz
        from PIL import Image
        import docx2txt
        from openai import OpenAI

        logging.info(f"📥 Received inlet body:\n{json.dumps(body, indent=2, ensure_ascii=False)}")
        files = body.get("files", [])
        extracted_texts = []
        if files:
            for file in files:
                url = file.get("url", "")
                content = None
        
                if url.startswith("http://") or url.startswith("https://"):
                    content_url = url + "/content"
                    async with httpx.AsyncClient(timeout=30) as c:
                        resp = await c.get(content_url)
                        resp.raise_for_status()
                        content = resp.content
                elif url.startswith("data:"):
                    # Пример: data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,...
                    header, b64data = url.split(",", 1)
                    content = base64.b64decode(b64data)
                else:
                    logging.warning(f"⚠️ Unsupported or missing URL scheme: {url}")
                    continue
        
                mime = file.get("mime_type") or mimetypes.guess_type(file.get("name", ""))[0]
                if mime == "application/pdf":
                    doc = fitz.open(stream=content, filetype="pdf")
                    extracted.append("\n".join(p.get_text() for p in doc))
                elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    with open("_tmp.docx", "wb") as tmp:
                        tmp.write(content)
                    extracted.append(docx2txt.process("_tmp.docx"))
                    os.remove("_tmp.docx")
                elif mime and mime.startswith("image/"):
                    b64 = base64.b64encode(content).decode()
                    res = self.client.chat.completions.create(
                        model=self.valves.MODEL_ID,
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": "Распознай текст на изображении."},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                        ]}]
                    )
                    extracted.append(res.choices[0].message.content.strip())
        
            body["file_text"] = "\n".join(extracted)
        else:
            body["file_text"] = ""
        return body

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Iterator[str]:
        file_text = body.get("file_text", "")
        if file_text:
            user_message += f"\n\nТекст из прикреплённых документов:\n{file_text}"

        search_result = asyncio.run(self.call_search_api(user_message))
        if search_result["search_required"] and search_result["context"]:
            user_message += "\n\nКонтекст из официальных источников:\n" + search_result["context"]

        deep_legal_context = ""
        if search_result["search_required"] and search_result["citations"]:
            deep_legal_context = asyncio.run(self.call_deep_extract_api(user_message, search_result["citations"]))
            if deep_legal_context:
                user_message += ("\n\n📘 Подтверждённые нормы закона (из официальных источников):\n"
                                 f"{deep_legal_context}\n"
                                 "\n❗️Используй только эти нормы для анализа.")

        system_message = f"""
**Роль:** Вы — эксперт по переговорам и стратегическому управлению. Ваша специализация — анализ и прогнозирование успешности различных моделей ведения переговоров, а также построение компромиссных стратегий для разрешения конфликтов между несколькими сторонами.

**Контекст из нормативных документов:**
{deep_legal_context or "—"}

**Область ответственности:**
Вы анализируете конфликтные интересы между сторонами, выделяете их цели, выявляете сильные и слабые стороны их позиций и предлагаете обоснованные стратегии компромисса.

**Цель:**
1. Проанализировать конфликтную ситуацию с участием двух или более сторон.
2. Рассмотреть аргументы и позиции каждой стороны.
3. Оценить слабые и сильные стороны этих позиций.
4. Найти компромисс, удовлетворяющий интересам всех сторон (насколько это возможно).

**Структура ответа:**
### 1. Стороны переговоров
- Краткое описание участников и их интересов.

### 2. Позиции сторон
#### Сторона A:
- Основная позиция: ...
- Сильные стороны: ...
- Слабые стороны: ...

#### Сторона B:
- Основная позиция: ...
- Сильные стороны: ...
- Слабые стороны: ...

### 3. Потенциальные компромиссы
- Перечень предложений, которые могут частично или полностью удовлетворить все стороны.
- Указание на то, какие интересы каждой стороны учтены и какие остаются спорными.

### 4. Прогноз устойчивости
- Анализ рисков, долговечности и масштабируемости компромисса.

### 5. Рекомендации
- Предложение оптимального подхода с объяснением.
- Уточняющие вопросы для сторон (если информация неполна).
"""

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

        formatted_messages = prompt.format_messages(user_input=user_message)

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
