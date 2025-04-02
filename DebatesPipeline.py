import logging
import sys
import os
import asyncio
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

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
    ) -> Union[str, Generator, Iterator]:

        system_message = """
    **Роль**: Вы — профессиональный аналитик дебатов, специализирующийся на глубоком, беспристрастном и нормативно обоснованном разборе аргументов. Ваша экспертиза строго ограничивается анализом содержания высказываний, заявлений, тезисов и спорных утверждений с точки зрения законодательства и нормативных актов Республики Казахстан.

    **Область ответственности**:
    Вы не отвечаете на вопросы, не относящиеся к анализу аргументации или нормативному соответствию. В случае нерелевантного запроса вы мягко уведомляете пользователя и предлагаете вернуться к основному фокусу — экспертизе аргумента. Вы можете вежливо спросить, может ли пользователь сформулировать тезис или утверждение для анализа.

    **Цель анализа**:
    1. Провести всестороннюю экспертизу представленного аргумента
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

    **Поведение при нерелевантных запросах**:
    Если пользователь задаёт вопрос, не связанный с анализом аргументов или правовой экспертизой, вы отвечаете:
    > «Извините, моя специализация ограничена анализом аргументов и их нормативной оценки. Могу ли я помочь вам с экспертизой какого-либо утверждения или дискуссионного тезиса?»
    """

        model = ChatOpenAI(
            api_key=self.valves.OPENAI_API_KEY,
            model=self.valves.MODEL_ID,
            temperature=self.valves.TEMPERATURE
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", user_message)
        ])

        response = model.invoke(prompt.format_messages())

        # Добавим приветствие, если это первое сообщение
        if len(messages) <= 1:
            response.content = "Здравствуйте! Буду рад Вам помочь. Ниже представлен разбор вашего аргумента:\n\n" + response.content

        return response.content
