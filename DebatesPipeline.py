import logging
import sys
import os
import asyncio
from typing import List, Iterator, Callable
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from typing import List, Iterator, Callable, Any, Dict
from pydantic import BaseModel
import httpx
from PIL import Image
import io
import fitz  # PyMuPDF for PDF
import docx2txt
import base64
import mimetypes

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4o"
        TEMPERATURE: float = 0.5
        MAX_TOKENS: int = 1500
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        SEARCH_API_URL: str = os.getenv("SEARCH_API_URL", "http://localhost:8008/check_and_search")

    def __init__(self):
        self.name = "Debate Pipeline"
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
                    json={"prompt": prompt, "pipeline": "DebatePipeline"}
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
                    json={"prompt": prompt, "citations": citations, "pipeline": "DebatePipeline"}
                )
                response.raise_for_status()
                return response.json().get("legal_context", "")
        except Exception as e:
            logging.error(f"Deep extract API error: {e}")
            return ""


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

    async def call_legal_reference_api(self, prompt: str) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(
                    self.valves.SEARCH_API_URL.replace("/check_and_search", "/extract_legal_references"),
                    json={"prompt": prompt, "pipeline": "DebatePipeline"}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logging.error(f"Legal Reference API error: {e}")
            return {"legal_snippets": [], "citations": []}


    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Iterator[str]:
        file_text = body.get("file_text", "")
        if file_text:
            user_message += f"\n\nТекст из прикреплённых документов:\n{file_text}"
        # Временно отключён блок векторного поиска
        legal_context = ""

        # System message with injected legal context
        system_message = f"""
**Роль**: Вы - профессиональный аналитик дебатов, специализирующийся на глубоком и беспристрастном разборе аргументов с точки зрения нормативных актов Казахстана.

**Контекст из нормативных документов**:
{legal_context}

**Цель анализа**:
1. Провести всестороннюю экспертизу предоставленного аргумента
2. Оценить логическую структуру и соответствие нормативным требованиям
3. Подготовить две линии выступлений: выступающий и оппонент
4. Сформировать ключевые вопросы и ответы, актуальные для дебатов

**СТРУКТУРА ОТВЕТА (ОБЯЗАТЕЛЬНАЯ):**

### 1. Аналитическая записка
- Полный структурированный разбор законопроекта
- Выявление сильных и слабых сторон с аргументацией
- Указание неявных допущений и уязвимостей
- Классификация тезисов по уровню убедительности

### 2. Сценарий дебатов
- Выделение двух сторон: выступающий (инициатор) и слушающий (оппонент)
- Подготовка ключевых вопросов для каждой стороны:
  - Вопросы, усиливающие аргументацию выступающего
  - Вопросы, раскрывающие уязвимости (от оппонента)
- Подготовка примерных ответов на эти вопросы
- Определение тактических линий защиты и атаки

### 3. Рекомендации по улучшению
- Методические советы для усиления риторики выступающего
- Стратегии реагирования на возможные критические замечания
- Предложения по корректировке формулировок и позиции

### 4. ИТОГОВАЯ ТАБЛИЦА АРГУМЕНТОВ (ОБЯЗАТЕЛЬНО!)

В конце анализа ВСЕГДА приводи сводную таблицу основных аргументов:

| Аргумент | Источник | Поддержка | Возражения |
|----------|----------|-----------|-------------|
| [Ключевой аргумент] | [Источник/документ] | [Основные доводы в поддержку] | [Главные контраргументы] |
| ... | ... | ... | ... |

**Правила для таблицы:**
- Выделить в аргументах конкретные упомянутые нормы: статьи, пункты, законы, кодексы
- Указать название и номер статьи (например, «статья 9 Закона „О персональных данных“»)
- Отражать источник не просто как "Законопроект", а как конкретный закон/кодекс
- Обосновывать позицию сторон с опорой на эти нормы

БЕЗ ПОЛНОГО АНАЛИЗА И ИТОГОВОЙ ТАБЛИЦЫ ОТВЕТ НЕДОПУСТИМ!
"""

        search_result = asyncio.run(self.call_search_api(user_message))
        legal_refs = asyncio.run(self.call_legal_reference_api(user_message))
        if search_result["search_required"] and search_result["context"]:
            user_message += "\n\nКонтекст из официальных источников:\n" + search_result["context"]
        if legal_refs["legal_snippets"]:
            user_message += "\n\n📘 Найденные нормы закона:\n" + "\n".join(f"- {line}" for line in legal_refs["legal_snippets"])
        deep_legal_context = ""
        if search_result["search_required"] and search_result["citations"]:
            deep_legal_context = asyncio.run(self.call_deep_extract_api(user_message, search_result["citations"]))
            if deep_legal_context:
                user_message += "\n\n📘 Подтверждённые нормы закона:\n" + deep_legal_context
        model = ChatOpenAI(
            api_key=self.valves.OPENAI_API_KEY,
            model=self.valves.MODEL_ID,
            temperature=self.valves.TEMPERATURE,
            streaming=True
        )
    
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template("{user_input}")
        ])
    
        formatted_messages = prompt.format_messages(user_input=user_message)
    
        def stream_model() -> Iterator[str]:
            for chunk in model.stream(formatted_messages):
                content = getattr(chunk, "content", None)
                if content:
                    logging.debug(f"Model chunk: {content}")
                    yield content
            if search_result["search_required"] and search_result["citations"]:
                yield "\n\n### Использованные источники:"
                for i, link in enumerate(search_result["citations"], 1):
                    yield f"\n[{i}] {link}"
    
        if body.get("stream", True):
            return asyncio.run(self.make_request_with_retry(stream_model))
        else:
            full_response = "".join(asyncio.run(self.make_request_with_retry(stream_model)))
            return full_response
