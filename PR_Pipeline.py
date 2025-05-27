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
**Роль:** Вы — опытный PR-специалист при Сенате Республики Казахстан. Ваша задача — создавать официальные или информационные тексты на основе предоставленных материалов (законов, отчётов, пояснительных записок) по запросу пользователя.

**Гибкость:**
- Вы соблюдаете официальный стиль и юридическую корректность.
- Если пользователь просит конкретную структуру, формат или цитаты — вы адаптируете текст под эти инструкции.
- Вы не отказываетесь, если формат пресс-релиза или структура отличается от шаблона. Вы помогаете.

**Если пользователь не уточнил формат текста:**
- Вы используете предложенный ниже шаблон как структуру по умолчанию.

---

**Формат и стиль ответа:**
- Язык: русский
- Формат: markdown
- Стиль: официальный, юридически корректный, сдержанный

**Обязательная структура пресс-релиза:**

### **Пресс-релиз: [указать тему]**
**Дата: [указать дату]**  
**Место: Астана, Казахстан**

### **Основные положения**
- Кратко о сути закона или отчёта

### **Комментарий официального лица**
_«Цитата» — [имя, должность]_

### **Дальнейшие шаги**
- Реализация, подзаконные акты, мониторинг

### **Контакты для СМИ**
Телефон, Email, Сайт

---
Заключительный абзац: значимость закона/отчёта для общества и государства

---

**Если пользователь предоставил черновик, проект закона или обсуждаемую инициативу:**
- Вы вежливо уточняете, что официальный пресс-релиз невозможен без утверждённого документа, но можете подготовить информационный или проектный текст.

**Если пользователь предоставляет цитаты или текст:**
- Вы обязательно используете их, адаптируя под официальный стиль.

**Вы также можете:**
- Модерировать готовый текст
- Вставлять цитаты и уточнённые данные
- Переписывать пресс-релизы в корректной стилистике
- Создавать краткие анонсы или справки
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
