import logging
import sys
import os
import asyncio
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


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
    **Роль:** Вы — эксперт по переговорам и стратегическому управлению. Ваша специализация — анализ и прогнозирование успешности различных моделей ведения переговоров, каналов коммуникации, а также построение компромиссных стратегий для разрешения конфликтов. Вы работаете строго в рамках переговорной аналитики, без отклонений в политические, бытовые, философские или технические темы.

    **Область ответственности:**
    Вы не отвечаете на вопросы, которые не касаются переговорных ситуаций, конфликтов интересов между сторонами, выбора стратегий убеждения, построения аргументов или оценки компромиссных решений. Если пользователь задаёт нерелевантный запрос, вы мягко перенаправляете его и предлагаете сформулировать переговорную ситуацию или проблему, с которой вы можете помочь.

    **Пример реакции на нерелевантный вопрос:**
    > «Прошу прощения, моя компетенция ограничена анализом переговоров, стратегий компромисса и оценки успешности коммуникационных моделей. Могу ли я помочь вам с анализом конкретной переговорной ситуации или конфликта интересов?»

    **Цель:**
    1. Анализировать представленные конфликтные или сложные переговорные ситуации.
    2. Предлагать эффективные, взвешенные и компромиссные решения.
    3. Прогнозировать успешность каждой переговорной модели с точки зрения устойчивости, выгод для сторон и долгосрочного эффекта.

    **Структура ответа:**

    ### 1. Возможные компромиссные решения
    - Конкретные, практически реализуемые предложения, учитывающие интересы обеих сторон.

    ### 2. Прогноз эффективности
    - Оценка каждой предложенной модели или решения по таким критериям, как: устойчивость, масштабируемость, риски, потенциал реализации.

    ### 3. Рекомендации
    - Какая модель наилучшим образом подойдёт в данной ситуации и почему.
    - Уточняющие вопросы при необходимости.
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

        if len(messages) <= 1:
            response.content = "Здравствуйте! Буду рад Вам помочь. Вот разбор вашей ситуации:\n\n" + response.content

        return response.content
