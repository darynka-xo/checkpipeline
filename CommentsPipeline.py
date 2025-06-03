import logging
import sys
import os
import time
import asyncio
import re
import httpx
import mimetypes
import base64
from textwrap import wrap
from typing import List, Dict, Iterator

from pydantic import BaseModel
from openai import OpenAI
import fitz  # PyMuPDF – PDF → text
import docx2txt  # .docx → text

# ======================================================================
# Logging
# ======================================================================
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

# ======================================================================
# Utilities
# ======================================================================

def clean_html(text: str) -> str:
    """Strip HTML tags & collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


async def web_search(query: str) -> List[Dict[str, str]]:
    """Use OpenAI *web_search_preview* tool to fetch comment snippets."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    rsp = await asyncio.to_thread(
        client.responses.create,
        model="gpt-4.1",
        tools=[{"type": "web_search_preview"}],
        input=query,
    )
    return [{
        "title": "OpenAI web search preview",
        "link": "https://www.google.com/search?q=" + query.replace(" ", "+"),
        "snippet": rsp.output_text,
    }]


# ======================================================================
# Pipeline
# ======================================================================


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

    # -------------------- Optional lifecycle hooks --------------------
    async def on_startup(self):
        logger.info("%s pipeline warming up…", self.name)

    async def on_shutdown(self):
        logger.info("%s pipeline shutting down…", self.name)

    # --------------------------- Inlet --------------------------------
    async def inlet(self, body: dict, user: dict | None = None):
        """Extract text from uploaded PDFs, DOCX, and images (OCR)."""
        extracted: List[str] = []
        for f in body.get("files", []):
            try:
                async with httpx.AsyncClient(timeout=30) as c:
                    resp = await c.get(f["url"] + "/content")
                    resp.raise_for_status()
                    content = resp.content

                mime = f.get("mime_type") or mimetypes.guess_type(f.get("name", ""))[0]
                if mime == "application/pdf":
                    doc = fitz.open(stream=content, filetype="pdf")
                    extracted.append("\n".join(p.get_text() for p in doc))
                    doc.close()
                elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    tmp = "_tmp.docx"
                    with open(tmp, "wb") as fh:
                        fh.write(content)
                    extracted.append(docx2txt.process(tmp))
                    os.remove(tmp)
                elif mime and mime.startswith("image/"):
                    b64 = base64.b64encode(content).decode()
                    ocr_resp = self.client.chat.completions.create(
                        model=self.valves.MODEL_ID,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Сделай OCR без интерпретации."},
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                            ],
                        }],
                        temperature=0.0,
                        max_tokens=1024,
                    )
                    extracted.append(ocr_resp.choices[0].message.content.strip())
                else:
                    logger.warning("Unsupported MIME %s", mime)
            except Exception as exc:
                logger.warning("Failed to process %s: %s", f.get("name"), exc)

        if extracted:
            body["file_text"] = "\n\n".join(extracted)
        return body

    # --------------------------- Main ---------------------------------
    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict | None = None) -> Iterator[str]:
        body = body or {}
        if body.get("file_text"):
            user_message += "\n\nТекст из прикреплённых документов:\n" + body["file_text"]

        # ---- Web search context
        try:
            search_results = asyncio.run(web_search(user_message))
        except Exception as exc:
            logger.warning("search error: %s", exc)
            search_results = []

        search_ctx = "".join(
            f"{i+1}. {r['snippet']} (источник: {r['link']})\n" for i, r in enumerate(search_results)
        )
        if search_ctx:
            search_ctx = "Найденные комментарии:\n" + search_ctx

        # ---- Prompt
        system_msg = """
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
        chat_messages = [
            {"role": "system", "content": system_msg},
            {"role": "assistant", "content": search_ctx},
            {"role": "user", "content": user_message},
        ]

        # ---- Single‑shot completion (no OpenAI Stream → avoids await bug)
        def generate_once() -> str:
            resp = self.client.chat.completions.create(
                model=self.valves.MODEL_ID,
                messages=chat_messages,
                temperature=self.valves.TEMPERATURE,
                max_tokens=self.valves.MAX_TOKENS,
            )
            return resp.choices[0].message.content

        try:
            full_answer = generate_once()
        except Exception as exc:
            logger.error("Completion failed: %s", exc)
            yield "❌ Ошибка генерации ответа. Попробуйте позже."
            return

        # ---- Chunk into ~800‑char pieces so OpenWebUI can stream gradually
        for piece in wrap(full_answer, 800):
            yield piece
            time.sleep(0.02)  # tiny pause for UI flushing
