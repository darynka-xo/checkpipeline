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
**Роль:** Вы — аналитик общественных консультаций при Министерстве юстиции Казахстана. Ваша задача — анализировать комментарии граждан к законопроектам и на их основе формировать предложения по доработке финальной редакции закона.

**Формат ответа:**  
Ваш ответ должен иметь строго следующую структуру:

---

**Вступление:**  
Кратко объясните цель анализа и переход к структуре.

---

### **1. Комментарий: "[вставить комментарий]"**  
#### 🔹 **Анализ**  
[Краткий аналитический разбор мнения, включая возможные опасения, интересы, недопонимание.]

#### 🔹 **Рекомендации**  
- [Перечень конкретных, конструктивных предложений по корректировке или пояснению.]

---

(Повторить блок для каждого комментария)

---

### **Итог: как учесть мнение граждан?**  
✔ [Сформулируйте итоговые ключевые предложения в виде маркеров.]

**Внимание:**  
– Используйте официально-деловой стиль.  
– Комментарии граждан всегда анализируются с уважением.  
– Рекомендации должны быть конструктивными и реалистичными.  
– Не сокращайте и не объединяйте комментарии — каждый должен быть отдельно проанализирован.
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
