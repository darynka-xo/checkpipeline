import logging
import sys
import os
import asyncio
from typing import List, Iterator, Callable
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from openai import OpenAI

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
        self.client = OpenAI(api_key=self.valves.OPENAI_API_KEY)

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

        async def web_search_summary(self, query: str) -> str:
            try:
                search_query = (
                    f"Комментарии граждан и реакция на законопроект о {query} "
                    "site:eotinish.kz OR site:gov.kz OR site:adilet.zan.kz "
                    "OR site:legalacts.egov.kz OR site:online.zakon.kz "
                    "OR site:inform.kz OR site:zakon.kz OR site:liter.kz OR site:kazpravda.kz"
                )
    
                response = self.client.responses.create(
                    model="gpt-4.1",
                    tools=[{"type": "web_search_preview"}],
                    input=search_query
                )
                text = response.output_text.strip()
                sources = []
    
                if hasattr(response, 'citations') and response.citations:
                    for src in response.citations:
                        title = src.get("title", "Источник")
                        url = src.get("url", "")
                        sources.append(f"- [{title}]({url})")
                else:
                    sources += [
                        "- [eotinish.kz](https://eotinish.kz)",
                        "- [adilet.zan.kz](https://adilet.zan.kz)",
                        "- [legalacts.egov.kz](https://legalacts.egov.kz)",
                        "- [online.zakon.kz](https://online.zakon.kz)",
                        "- [inform.kz](https://inform.kz)",
                        "- [zakon.kz](https://zakon.kz)"
                    ]
    
                return (
                    text +
                    "\n\n🔗 Использованные источники:\n" +
                    "\n".join(sources)
                )
    
            except Exception as e:
                logging.warning(f"Web search error: {e}")
                return "📡 Не удалось получить комментарии из интернета."


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

        async def generate_augmented_input():
            web_summary = await self.web_search_summary(f"Комментарии граждан по теме: {user_message}")
            enriched = (
                user_message +
                "\n\n📡 Комментарии из интернета:\n" +
                web_summary +
                "\n\n🔗 Использованные источники (предварительный обзор):\n"
                "- Google Поиск\n"
                "- Обсуждения в социальных сетях и форумах\n"
                "- Новостные сайты и комментарии к статьям\n"
                "- Платформы для общественных обсуждений (например, eotinish.kz, gov.kz)\n"
                "(Детальная конкретизация источников может быть получена по ссылке в тексте или при последующем анализе)"
            )

            formatted_messages = prompt.format_messages(user_input=enriched)

            def generate_stream() -> Iterator[str]:
                for chunk in model.stream(formatted_messages):
                    content = getattr(chunk, "content", None)
                    if content:
                        logging.debug(f"Model chunk: {content}")
                        yield content

            return await self.make_request_with_retry(generate_stream)

        return asyncio.run(generate_augmented_input())


        return asyncio.run(generate_augmented_input())
