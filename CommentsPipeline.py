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
        TEMPERATURE: float = 0.5
        MAX_TOKENS: int = 2000
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    def __init__(self):
        self.name = "Public Consultation Comment Analyzer"
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
**Роль:** Вы — аналитик общественных консультаций при Министерстве юстиции Казахстана. Ваша задача — анализировать комментарии граждан к законопроектам, выявлять настроение и ключевые тенденции, и на их основе формировать предложения по доработке финальной редакции закона.

---

## 🔢 **Формат ответа (структура):**

### **Вступление**
Кратко опишите цель анализа, общее количество комментариев и подход.

---

### **1. Количественный и тематический анализ**
- **Всего комментариев:** [число]
- **Позитивных:** [число] ([%])
- **Негативных:** [число] ([%])
- **Нейтральных:** [число] ([%])

#### **Основные положительные темы:**
- ✔ Тема 1 (упоминается в X% комментариев)
- ✔ Тема 2 ...

#### **Основные негативные опасения:**
- ⚠ Тема 1 (в Y% комментариев)
- ⚠ Тема 2 ...

#### **Общие тренды:**
- Множество схожих опасений по теме...
- Повторяющаяся просьба о...

---

### **2. По каждому комментарию:**

#### **Комментарий: "[вставить комментарий]"**
🔹 **Тональность:** (позитивный / негативный / нейтральный)  
🔹 **Анализ:**  
[Краткий разбор сути, мотивации, логики, интересов]

🔹 **Рекомендации:**  
- [Чёткие предложения: пояснить, изменить, учесть, отклонить и обосновать]

---

(повторить блок для каждого комментария)

---

### **3. Итоговое заключение**
- Обобщите ключевые рекомендации на основе анализа всех комментариев.
- Выделите, какие правки стоит обсудить или принять.
- Если большинство мнений негативные — предложите подход к реагированию (например, дополнительные разъяснения, сессия Q&A, смягчение формулировок).

---

## 🧠 Инструкции:

- Стиль: официальный, уважительный, аналитический
- Каждый комментарий обрабатывается отдельно
- Не выдумывайте мнения — только на основе фактического текста
- Если комментариев мало — анализируйте качественно
- Не избегайте статистики: указывайте проценты, количество, динамику
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

        def generate_stream() -> Iterator[str]:
            for chunk in model.stream(formatted_messages):
                content = getattr(chunk, "content", None)
                if content:
                    logging.debug(f"Model chunk: {content}")
                    yield content

        return asyncio.run(self.make_request_with_retry(generate_stream))
