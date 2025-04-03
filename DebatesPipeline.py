import logging
import sys
import os
import asyncio
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import LLMResult
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4o"
        TEMPERATURE: float = 0.5
        MAX_TOKENS: int = 1500
        OPENAI_API_KEY: str = ""

    def __init__(self):
        self.name = "Debate Pipeline"
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

        # Оригинальный system prompt + контекст из БД
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

        for chunk in model.stream(formatted_messages):
            content = getattr(chunk, "content", None)
            if content:
                yield content