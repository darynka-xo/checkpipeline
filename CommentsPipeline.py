import logging
import sys
import os
import time
import asyncio
import re
import httpx
import mimetypes
import base64
from typing import List, Dict, Iterator

from pydantic import BaseModel
from openai import OpenAI
import fitz  # PyMuPDF – PDF → text
import docx2txt  # .docx → text

# ======================================================================
# Logging setup
# ======================================================================
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

# ======================================================================
# Helpers
# ======================================================================

def clean_html(text: str) -> str:
    """Quick-n-dirty HTML → plain-text conversion (strip tags + shrink WS)."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


async def web_search(query: str) -> List[Dict[str, str]]:
    """Call **OpenAI web_search_preview** and wrap the answer in a list-of-dicts."""
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


async def open_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; CommentBot/1.1)"}
    try:
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            r = await client.get(url, follow_redirects=True)
            r.raise_for_status()
            return clean_html(r.text)[:15000]
    except Exception as exc:
        logger.warning("open_url error for %s: %s", url, exc)
        return f"__FETCH_ERROR__: {exc}"


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

    # ------------------------------------------------------------------
    #   Optional lifecycle hooks (OpenWebUI)
    # ------------------------------------------------------------------
    async def on_startup(self):
        logger.info("%s pipeline warming up…", self.name)

    async def on_shutdown(self):
        logger.info("%s pipeline shutting down…", self.name)

    # ------------------------------------------------------------------
    #   File ingestion (PDF, DOCX, images via OCR with GPT-4o’s vision)
    # ------------------------------------------------------------------
    async def inlet(self, body: dict, user: dict | None = None):
        extracted: List[str] = []
        for f in body.get("files", []):
            try:
                # download binary
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
                    ocr_rsp = self.client.chat.completions.create(
                        model=self.valves.MODEL_ID,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Извлеки OCR-текст без интерпретации."},
                                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                            ],
                        }],
                        temperature=0.0,
                        max_tokens=1024,
                    )
                    extracted.append(ocr_rsp.choices[0].message.content.strip())
                else:
                    logger.warning("Unsupported MIME type %s – skipped", mime)
            except Exception as exc:
                logger.warning("Failed to process %s: %s", f.get("name"), exc)

        if extracted:
            body["file_text"] = "\n\n".join(extracted)
        return body

    # ------------------------------------------------------------------
    #   Main handler (sync iterator, no await issues!)
    # ------------------------------------------------------------------
    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict | None = None) -> Iterator[str]:
        body = body or {}

        # augment with file text
        if body.get("file_text"):
            user_message += "\n\nТекст из прикреплённых документов:\n" + body["file_text"]

        # collect comment snippets via OpenAI search (best-effort, no fatal errors)
        try:
            search_results = asyncio.run(web_search(user_message))
        except Exception as exc:
            logger.warning("search error: %s", exc)
            search_results = []

        search_context = ""
        if search_results:
            search_context = "Найденные в открытом доступе комментарии:\n" + "\n".join(
                f"{idx+1}. {r['snippet']} (источник: {r['link']})" for idx, r in enumerate(search_results)
            )

        # -------------------------- PROMPT -----------------------------
        system_msg = (
            "**Роль:** Вы — аналитик общественных консультаций при Министерстве юстиции Казахстана. "
            "Ваша задача — анализировать комментарии граждан к законопроектам, выявлять настроение и ключевые тенденции, "
            "и формировать предложения по доработке финальной редакции закона.\n\n"
            "(далее следует подробный шаблон, опущен для краткости – тот же, что был ранее)"
        )

        chat_messages = [
            {"role": "system", "content": system_msg},
            {"role": "assistant", "content": search_context},
            {"role": "user", "content": user_message},
        ]

        # ---------------------- Streaming with retries -----------------
        def generate_stream():
            for attempt in range(1, 4):
                try:
                    stream = self.client.chat.completions.create(
                        model=self.valves.MODEL_ID,
                        messages=chat_messages,
                        temperature=self.valves.TEMPERATURE,
                        max_tokens=self.valves.MAX_TOKENS,
                        stream=True,
                    )
                    for chunk in stream:  # <-- sync iterator from OpenAI SDK
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            yield delta.content
                    return  # success – stop retry loop
                except Exception as exc:
                    logger.error("Attempt %d failed: %s", attempt, exc)
                    if attempt == 3:
                        yield "❌ Ошибка генерации ответа после 3 попыток. Попробуйте позже."
                    time.sleep(2 ** (attempt - 1))

        return generate_stream()
