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

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Iterator[str]:
        # Подключение к векторной БД
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

        vectorstore = PGVector(
            connection_string=os.getenv("PGVECTOR_URL", "postgresql://admin:tester123@localhost:5432/postgres"),
            collection_name="laws_chunks_ru",
            embedding_function=embeddings
        )

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents(user_message)
        legal_context = "\n\n".join([f"- {doc.page_content}" for doc in docs])

        # System message with injected legal context
        system_message = f"""
**Роль**: Вы - профессиональный аналитик дебатов, специализирующийся на глубоком и беспристрастном разборе аргументов с точки зрения нормативных актов Казахстана.

**Контекст из нормативных документов**:
{legal_context}

**Цель анализа**:
1. Провести всестороннюю экспертизу предоставленного аргумента
2. Оценить логическую структуру и соответствие нормативным требованиям
3. Выявить потенциальные слабости и предложить конкретные улучшения

**Требования к анализу**:

#### 1. Раздел "Аналитическая записка"
- Полный структурированный разбор аргумента
- Оценка логической связности и аргументированности
- Выявление скрытых посылок и неявных допущений
- Определение уровня доказательности каждого тезиса

#### 2. Раздел "Отчет о соответствии нормативным актам"
- Детальная проверка каждого утверждения на соответствие официальным нормам
- Точные ссылки на конкретные статьи и пункты нормативных документов
- Квалификация обнаруженных отклонений (minor/significant)
- Правовая оценка потенциальных нарушений

#### 3. Раздел "Рекомендации по улучшению"
- Конкретные предложения по доработке аргументации
- Методические рекомендации по усилению слабых позиций
- Альтернативные формулировки и подходы
- Стратегические советы для повышения убедительности
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
