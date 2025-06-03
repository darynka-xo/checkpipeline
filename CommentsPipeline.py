import logging
import sys
import os
import asyncio
import re
import httpx
import mimetypes
import base64
import io
from typing import List, Iterator, Callable, Dict

from pydantic import BaseModel
from PIL import Image
import fitz  # PyMuPDF
import docx2txt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

TRUSTED_DOMAINS = [
    "open.zakon.kz", "legalacts.egov.kz", "adilet.zan.kz",
    "online.zakon.kz", "egov.kz", "eotinish.kz", "tengrinews.kz",
    "kursiv.media", "inbusiness.kz", "kapital.kz"
]

def _is_trusted(url: str) -> bool:
    return any(d in url for d in TRUSTED_DOMAINS)

def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

async def web_search(query: str) -> List[Dict[str, str]]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.responses.create(
            model="gpt-4.1",
            tools=[{"type": "web_search_preview"}],
            input=query
        )
        return [{
            "title": "Общественные комментарии",
            "link": "https://www.google.com/search?q=" + query.replace(" ", "+"),
            "snippet": response.output_text
        }]
    except Exception as e:
        logging.warning(f"OpenAI web_search_preview error: {e}")
        return []

async def open_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PublicConsultBot/1.0)"}
    try:
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            r = await client.get(url, follow_redirects=True)
            r.raise_for_status()
            return clean_html(r.text)[:15000]
    except Exception as e:
        logging.warning(f"open_url error for {url}: {e}")
        return f"__FETCH_ERROR__: {e}"

class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4.1"
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

    async def inlet(self, body: dict, user: dict) -> dict:
        logging.info("📥 Inlet body:")

        extracted = []
        for f in body.get("files", []):
            content_url = f["url"] + "/content"
            async with httpx.AsyncClient(timeout=30) as c:
                resp = await c.get(content_url)
                resp.raise_for_status()
                content = resp.content
            mime = f.get("mime_type") or mimetypes.guess_type(f.get("name", ""))[0]
            if mime == "application/pdf":
                doc = fitz.open(stream=content, filetype="pdf")
                extracted.append("\n".join(p.get_text() for p in doc))
            elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                with open("_tmp.docx", "wb") as tmp:
                    tmp.write(content)
                extracted.append(docx2txt.process("_tmp.docx"))
                os.remove("_tmp.docx")
            elif mime and mime.startswith("image/"):
                b64 = base64.b64encode(content).decode()
                res = self.client.chat.completions.create(
                    model=self.valves.MODEL_ID,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": "Распознай текст на изображении."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                    ]}]
                )
                extracted.append(res.choices[0].message.content.strip())
        body["file_text"] = "\n".join(extracted)

        if body.get("query"):
            search_results = await web_search(body["query"])
            search_texts = []
            for res in search_results:
                if _is_trusted(res["link"]):
                    html = await open_url(res["link"])
                    search_texts.append(f"Источник: {res['link']}\n{html}")
            body["search_comments"] = "\n---\n".join(search_texts)

        return body

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Iterator[str]:
        if body.get("file_text"):
            user_message += "\n\nТекст из прикреплённых документов:\n" + body["file_text"]
        if body.get("search_comments"):
            user_message += "\n\nПроанализируй и классифицируй следующие общественные комментарии:\n" + body["search_comments"]

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

        async def _generate() -> str:
            try:
                response = self.client.responses.create(
                    model="gpt-4.1",
                    tools=[{"type": "web_search_preview"}],
                    input=f"{system_msg}\n\n<проект>\n{user_message}"
                )
                return response.output_text
            except Exception as e:
                logging.error(f"❌ Ошибка генерации: {e}")
                return "❌ Ошибка генерации ответа. Попробуйте ещё раз."

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_generate())
