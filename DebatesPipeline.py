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
                    json={"prompt": prompt, "pipeline": "NegotiationPipeline"}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logging.error(f"Search API error: {e}")
            return {"search_required": False, "context": "", "citations": []}

    async def inlet(self, body: dict, user: dict) -> dict:
        print(f"Received body: {body}")
        messages = body.get("messages", [])
        extracted_texts = []

        for message in messages:
            for item in message.get("content", []):
                if item["type"] == "file":
                    file_url = item["file"]["url"]
                    file_name = item["file"].get("name", "file")
                    mime_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"

                    async with httpx.AsyncClient(timeout=30) as client:
                        response = await client.get(file_url)
                        response.raise_for_status()
                        content = response.content

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

                elif item["type"] == "image_url":
                    image_url = item["image_url"]["url"]
                    detail = item["image_url"].get("detail", "auto")
                    if image_url.startswith("data:image/"):  # base64
                        header, base64_data = image_url.split(",", 1)
                        mime_type = header.split(";")[0].split(":")[1]
                        image_bytes = base64.b64decode(base64_data)

                        from openai import OpenAI
                        client = OpenAI(api_key=self.valves.OPENAI_API_KEY)
                        ocr_response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Распознай текст на изображении."},
                                    {"type": "image_url", "image_url": {"url": image_url, "detail": detail}}
                                ]
                            }]
                        )
                        image_text = ocr_response.choices[0].message.content.strip()
                        extracted_texts.append(image_text)

        body["file_text"] = "\n".join(extracted_texts)
        return body


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

**Требования к анализу**:

#### 1. Раздел "Аналитическая записка"
- Полный структурированный разбор законопроекта
- Выявление сильных и слабых сторон с аргументацией
- Указание неявных допущений и уязвимостей
- Классификация тезисов по уровню убедительности

#### 2. Раздел "Сценарий дебатов"
- Выделение двух сторон: выступающий (инициатор) и слушающий (оппонент)
- Подготовка ключевых вопросов для каждой стороны:
  - Вопросы, усиливающие аргументацию выступающего
  - Вопросы, раскрывающие уязвимости (от оппонента)
- Подготовка примерных ответов на эти вопросы
- Определение тактических линий защиты и атаки

#### 3. Раздел "Рекомендации по улучшению"
- Методические советы для усиления риторики выступающего
- Стратегии реагирования на возможные критические замечания
- Предложения по корректировке формулировок и позиции
"""

        search_result = asyncio.run(self.call_search_api(user_message))
        if search_result["search_required"] and search_result["context"]:
            user_message += "\n\nКонтекст из официальных источников:\n" + search_result["context"]
    
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
    
        return asyncio.run(self.make_request_with_retry(stream_model))
