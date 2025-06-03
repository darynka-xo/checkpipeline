import logging
import sys
import os
import asyncio
import re
import httpx
import mimetypes
import base64
from typing import List, Dict, Iterator, Callable

from pydantic import BaseModel
from openai import OpenAI
from PIL import Image  # noqa: F401 – kept for future image analysis extensions
import fitz  # PyMuPDF – PDF → text
import docx2txt  # .docx → text

###############################################################################
# Logging
###############################################################################
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

###############################################################################
# Helper utilities
###############################################################################

def clean_html(text: str) -> str:
    """Very-light HTML → plain-text cleaner (tags + extra whitespace)."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


async def web_search(query: str) -> List[Dict[str, str]]:
    """Call **OpenAI web_search_preview** tool and return a minimal result list.

    The endpoint returns its own textual summary; we wrap it in a pseudo-result so
    it can be fed back into the prompt alongside a Google link for manual review.
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        rsp = client.responses.create(
            model="gpt-4.1",
            tools=[{"type": "web_search_preview"}],
            input=query,
        )
        return [
            {
                "title": "OpenAI web search preview",  # stable placeholder
                "link": "https://www.google.com/search?q=" + query.replace(" ", "+"),
                "snippet": rsp.output_text,
            }
        ]
    except Exception as exc:
        logger.warning("OpenAI web_search_preview error: %s", exc)
        return []


async def open_url(url: str) -> str:
    """Fetch raw page text (best-effort) for richer context if needed."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; CommentBot/1.0)"}
    try:
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            r = await client.get(url, follow_redirects=True)
            r.raise_for_status()
            return clean_html(r.text)[:15000]  # truncate to 15k chars
    except Exception as exc:
        logger.warning("open_url error for %s: %s", url, exc)
        return f"__FETCH_ERROR__: {exc}"

###############################################################################
# Pipeline
###############################################################################


class Pipeline:
    """OpenWebUI-compatible pipeline that:

    • Accepts arbitrary files/images (PDF, DOCX, images) and extracts their text
      into *body["file_text"]* via *inlet*.
    • Performs a lightweight OpenAI-powered web search for public comments that
      mention the user’s query and feeds those snippets to the model.
    • Streams the analysis back to OpenWebUI.
    """

    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4o"
        TEMPERATURE: float = 0.5
        MAX_TOKENS: int = 2000
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    def __init__(self):
        self.name = "Public Consultation Comment Analyzer"
        self.valves = self.Valves()
        self.client = OpenAI(api_key=self.valves.OPENAI_API_KEY)

    # ---------------------------------------------------------------------
    # Lifecycle hooks (optional in OpenWebUI but kept for completeness)
    # ---------------------------------------------------------------------
    async def on_startup(self):
        logger.info("%s pipeline warming up…", self.name)

    async def on_shutdown(self):
        logger.info("%s pipeline shutting down…", self.name)

    # ------------------------------------------------------------------
    # Resilient execution wrapper (same semantics as in original template)
    # ------------------------------------------------------------------
    async def _run_with_retry(self, coro_factory: Callable[[], asyncio.Future], retries: int = 3):
        for attempt in range(1, retries + 1):
            try:
                return await coro_factory()
            except Exception as exc:
                logger.error("Attempt %d failed: %s", attempt, exc)
                if attempt == retries:
                    raise
                await asyncio.sleep(2 ** (attempt - 1))

    # ------------------------------------------------------------------
    # Inlet – extract text from any supported file types and stash to body
    # ------------------------------------------------------------------
    async def inlet(self, body: dict, user: dict | None = None):  # noqa: D401 – OWUI signature
        """Extract plain text from PDFs, DOCX files, and images.

        The extracted text is placed in `body["file_text"]` so it is available
        inside *pipe()*.
        """
        extracted_chunks: List[str] = []
        for f in body.get("files", []):
            try:
                # ---- Download binary content from WebUI-provided presigned URL
                content_url = f["url"] + "/content"
                async with httpx.AsyncClient(timeout=30) as c:
                    resp = await c.get(content_url)
                    resp.raise_for_status()
                    content = resp.content

                # ---- Detect MIME type and route to an extractor
                mime = f.get("mime_type") or mimetypes.guess_type(f.get("name", ""))[0]
                if mime == "application/pdf":
                    doc = fitz.open(stream=content, filetype="pdf")
                    extracted_chunks.append("\n".join(p.get_text() for p in doc))
                    doc.close()
                elif mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    tmp_name = "_tmp_uploaded.docx"
                    with open(tmp_name, "wb") as tmp:
                        tmp.write(content)
                    extracted_chunks.append(docx2txt.process(tmp_name))
                    os.remove(tmp_name)
                elif mime and mime.startswith("image/"):
                    b64 = base64.b64encode(content).decode()
                    rsp = self.client.chat.completions.create(
                        model=self.valves.MODEL_ID,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Извлеки (OCR) текст с изображения без интерпретации."},
                                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                                ],
                            }
                        ],
                        temperature=0.0,
                        max_tokens=1024,
                    )
                    extracted_chunks.append(rsp.choices[0].message.content.strip())
                else:
                    logger.warning("Unsupported MIME type %s – skipped", mime)
            except Exception as exc:
                logger.warning("Failed to extract from %s: %s", f.get("name"), exc)

        if extracted_chunks:
            body["file_text"] = "\n\n".join(extracted_chunks)
        return body  # WebUI expects the (possibly modified) body back

    # ------------------------------------------------------------------
    # Main pipeline logic – called for each user turn
    # ------------------------------------------------------------------
    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict | None = None,
    ) -> Iterator[str]:
        body = body or {}
        # --------------------------------------------------------------
        # 1️⃣ Augment user message with extracted file text (if any)
        # --------------------------------------------------------------
        if body.get("file_text"):
            user_message += "\n\nТекст из прикреплённых документов:\n" + body["file_text"]

        # --------------------------------------------------------------
        # 2️⃣ Fetch public commentary snippets via OpenAI search preview
        # --------------------------------------------------------------
        async def _collect_search_context():
            results = await web_search(user_message)
            if not results:
                return ""
            formatted = "\n".join(
                f"{i+1}. {r['snippet']} (источник: {r['link']})" for i, r in enumerate(results)
            )
            return "Найденные в открытом доступе комментарии:\n" + formatted

        search_context = asyncio.run(self._run_with_retry(_collect_search_context))

        # --------------------------------------------------------------
        # 3️⃣ Compose prompt (the original system instructions + context)
        # --------------------------------------------------------------
        system_message = (
            "**Роль:** Вы — аналитик общественных консультаций при Министерстве юстиции Казахстана. "
            "Ваша задача — анализировать комментарии граждан к законопроектам, выявлять настроение и ключевые тенденции, "
            "и на их основе формировать предложения по доработке финальной редакции закона.\n\n"
            "---\n\n"
            "## 🔢 **Формат ответа (структура):**\n\n"
            "### **Вступление**\n"
            "Кратко опишите цель анализа, общее количество комментариев и подход.\n\n"
            "---\n\n"
            "### **1. Количественный и тематический анализ**\n"
            "- **Всего комментариев:** [число]\n"
            "- **Позитивных:** [число] ([%])\n"
            "- **Негативных:** [число] ([%])\n"
            "- **Нейтральных:** [число] ([%])\n\n"
            "#### **Основные положительные темы:**\n"
            "- ✔ Тема 1 (упоминается в X% комментариев)\n"
            "- ✔ Тема 2 …\n\n"
            "#### **Основные негативные опасения:**\n"
            "- ⚠ Тема 1 (в Y% комментариев)\n"
            "- ⚠ Тема 2 …\n\n"
            "#### **Общие тренды:**\n"
            "- Множество схожих опасений по теме...\n"
            "- Повторяющаяся просьба о…\n\n"
            "---\n\n"
            "### **2. По каждому комментарию:**\n\n"
            "#### **Комментарий: \"[вставить комментарий]\"**\n"
            "🔹 **Тональность:** (позитивный / негативный / нейтральный)  \n"
            "🔹 **Анализ:**  \n"
            "[Краткий разбор сути, мотивации, логики, интересов]\n\n"
            "🔹 **Рекомендации:**  \n"
            "- [Чёткие предложения: пояснить, изменить, учесть, отклонить и обосновать]\n\n"
            "---\n\n"
            "(повторить блок для каждого комментария)\n\n"
            "---\n\n"
            "### **3. Итоговое заключение**\n"
            "- Обобщите ключевые рекомендации на основе анализа всех комментариев.\n"
            "- Выделите, какие правки стоит обсудить или принять.\n"
            "- Если большинство мнений негативные — предложите подход к реагированию (например, дополнительные разъяснения, сессия Q&A, смягчение формулировок).\n\n"
            "---\n\n"
            "## 🧠 Инструкции:\n\n"
            "- Стиль: официальный, уважительный, аналитический\n"
            "- Каждый комментарий обрабатывается отдельно\n"
            "- Не выдумывайте мнения — только на основе фактического текста\n"
            "- Если комментариев мало — анализируйте качественно\n"
            "- Не избегайте статистики: указывайте проценты, количество, динамику"
        )

        # Assemble chat messages. We pass the **search context** as an assistant
        # turn so the model treats it as background knowledge, not a direct user
        # instruction.
        chat_messages = [
            {"role": "system", "content": system_message},
            {"role": "assistant", "content": search_context},
            {"role": "user", "content": user_message},
        ]

        # --------------------------------------------------------------
        # 4️⃣ Helper: stream OpenAI response and yield chunks to WebUI
        # --------------------------------------------------------------
        async def _stream_response():
            response = await self._run_with_retry(
                lambda: self.client.chat.completions.create(
                    model=self.valves.MODEL_ID,
                    messages=chat_messages,
                    temperature=self.valves.TEMPERATURE,
                    max_tokens=self.valves.MAX_TOKENS,
                    stream=True,
                )
            )
            async for chunk in response:  # type: ignore – `response` is an async iterator
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

        # --------------------------------------------------------------
        # 5️⃣ Bridge async-generator → sync iterator (OpenWebUI requirement)
        # --------------------------------------------------------------
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        agen = _stream_response()
        try:
            while True:
                content = loop.run_until_complete(agen.__anext__())
                yield content
        except StopAsyncIteration:
            pass
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
