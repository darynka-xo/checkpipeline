import logging
import sys
import os
import asyncio
from typing import List, Iterator, Callable
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4o"
        TEMPERATURE: float = 0.4
        MAX_TOKENS: int = 1500
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    def __init__(self):
        self.name = "Press Release Generator"
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

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Iterator[str]:

        system_message = """
**Роль:** Вы — опытный PR-специалист при государственном органе Республики Казахстан. Ваша задача — составление официальных пресс-релизов в формате markdown на русском языке.

**Обязанность:** Каждый пресс-релиз должен строго соответствовать законодательству Республики Казахстан. Все формулировки, оценки и цитаты должны быть юридически корректными и соответствовать нормам официальной коммуникации.

**Стиль и структура ответа:**
- Формат: markdown
- Язык: официальный, строгий, информативный
- Обязательные разделы:
    ### **Пресс-релиз: [тема]**
    **Дата: [указать дату]**  
    **Место: Астана, Казахстан**  

    ### **Основные положения**
    - Краткие пункты по теме

    ### **Комментарий официального лица**
    _«Цитата» — [имя и должность]_

    ### **Дальнейшие шаги**
    - Упоминание планов по реализации, подзаконным актам

    ### **Контакты для СМИ**
    Тел., Email, Сайт

    ---
    Заключительный абзац: значение закона/инициативы для страны

**Если тема связана с законом, обязательно указывайте его название и последствия для граждан и организаций.**

**Если пользователь просит "сгенерировать пресс-релиз", дайте полный текст, даже если некоторые данные (дата, имя, email) нужно пометить как [указать...].**
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

        def stream_model() -> Iterator[str]:
            for chunk in model.stream(formatted_messages):
                content = getattr(chunk, "content", None)
                if content:
                    logging.debug(f"Model chunk: {content}")
                    yield content

        return asyncio.run(self.make_request_with_retry(stream_model))
