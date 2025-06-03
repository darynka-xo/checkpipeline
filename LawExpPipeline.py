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
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# Optional imports for file‑text extraction
from PIL import Image
import fitz                     # PyMuPDF
import docx2txt                 # .docx -> text

###############################################################################
# Logging
###############################################################################
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

###############################################################################
# Helper tools for the LLM agent
###############################################################################
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
TRUSTED_DOMAINS = [
    "akorda.kz", "senate.parlam.kz", "primeminister.kz",
    "otyrys.prk.kz", "senate-zan.prk.kz", "lib.prk.kz",
    "online.zakon.kz", "adilet.zan.kz", "legalacts.egov.kz",
    "egov.kz", "eotinish.kz"
]

def _is_trusted(url: str) -> bool:
    return any(d in url for d in TRUSTED_DOMAINS)

def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)  # remove tags
    text = re.sub(r"\s+", " ", text)
    return text.strip()

async def web_search(query: str) -> List[Dict[str, str]]:
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": 10},
                headers=headers,
            )
            r.raise_for_status()
            items = r.json().get("organic", [])
    except Exception as e:
        logging.warning(f"Serper error for '{query}': {e}")
        return []
    return [
        {"title": it["title"], "link": it["link"], "snippet": it.get("snippet", "")}
        for it in items
        if _is_trusted(it["link"])
    ]


async def open_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LawExpBot/1.0)"}
    try:
        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            r = await client.get(url, follow_redirects=True)
            r.raise_for_status()
            return clean_html(r.text)[:15000]
    except Exception as e:
        logging.warning(f"open_url error for {url}: {e}")
        return f"__FETCH_ERROR__: {e}"


# Wrap as LangChain tools
SEARCH_TOOL = Tool.from_function(
    name="web_search",
    description="Найди официальные документы и статьи казахстанских гос‑сайтов по заданному запросу (на русском). Возвращает список объектов {title, link, snippet}.",
    func=web_search,
)
FETCH_TOOL = Tool.from_function(
    name="open_url",
    description="Скачай HTML страницы по URL и верни чистый текст без тегов. Использовать только для ссылок с гос‑сайтов.",
    func=open_url,
)

###############################################################################
# The OpenWebUI Pipeline
###############################################################################
class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4o"
        TEMPERATURE: float = 0.3
        MAX_TOKENS: int = 1800
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    def __init__(self):
        self.name = "Эксперт по предложениям"
        self.valves = self.Valves()

        # init LLM + tools‑powered agent
        llm = ChatOpenAI(
            api_key=self.valves.OPENAI_API_KEY,
            model=self.valves.MODEL_ID,
            temperature=self.valves.TEMPERATURE,
            max_tokens=self.valves.MAX_TOKENS,
        )
        self.agent = initialize_agent(
            [SEARCH_TOOL, FETCH_TOOL],
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=False,
            max_iterations=12,
            max_execution_time=120,
            early_stopping_method="generate"  # остановка без ошибок
        )

    # ---------------------------------------------------------------------
    # OpenWebUI life‑cycle hooks
    # ---------------------------------------------------------------------
    async def on_startup(self):
        logging.info("LawExp pipeline warming up…")

    async def on_shutdown(self):
        logging.info("LawExp pipeline shutting down…")

    # ------------------------------------------------------------------
    # Helper: retry wrapper for generators (for streaming UI)
    # ------------------------------------------------------------------
    async def make_request_with_retry(self, fn: Callable[[], Iterator[str]], retries=3) -> Iterator[str]:
        for attempt in range(retries):
            try:
                return fn()
            except Exception as e:
                logging.error(f"Attempt {attempt+1} failed: {e}")
                if attempt + 1 == retries:
                    raise
                await asyncio.sleep(2 ** attempt)

    # ------------------------------------------------------------------
    # inlet: handle uploaded files → OCR/extract text
    # ------------------------------------------------------------------
    async def inlet(self, body: dict, user: dict) -> dict:
        import json
        logging.info("📥 Inlet body:\n" + json.dumps(body, indent=2, ensure_ascii=False))

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
                # cheap OCR via OpenAI Vision
                from openai import OpenAI
                client = OpenAI(api_key=self.valves.OPENAI_API_KEY)
                b64 = base64.b64encode(content).decode()
                res = client.chat.completions.create(
                    model=self.valves.MODEL_ID,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": "Распознай текст на изображении."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
                    ]}]
                )
                extracted.append(res.choices[0].message.content.strip())
        body["file_text"] = "\n".join(extracted)
        return body

    # ------------------------------------------------------------------
    # main pipe: build system prompt → delegate to LLM agent
    # ------------------------------------------------------------------
    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Iterator[str]:
        # attach text from uploaded files
        if body.get("file_text"):
            user_message += "\n\nТекст из прикреплённых документов:\n" + body["file_text"]

        # system instructions for the agent (NO clarifying questions!)
        system_msg = (
            "Ты — ИИ‑эксперт по сравнительному правовому анализу. Строго соблюдай правила:\n"
            "• Никогда не задавай пользователю уточняющих вопросов.\n"
            "• Если данных мало — делай лучшее возможное предположение и продолжай.\n"
            "• Используй web_search / open_url, но выводи ответ одним сообщением.\n"
            "Твоя задача:\n"
            "1. Найти действующие нормы в казахстанском праве, пересекающиеся или дублирующие положения проекта.\n"
            "2. Собрать краткую историю правок этих норм (если есть на adilet).\n"
            "3. Вывести 📊 таблицу: | № | Статья (новый) | Дублирующая норма | Источник | История правок | Комментарий |\n"
            "4. В конце дать ⚖️ Итог с рекомендациями.\n"
            "Не придумывай статьи — только реальные." )

        async def _generate() -> str:
            prompt = f"{system_msg}\n\n<проект>\n{user_message}"
            return await self.agent.arun(prompt)

        loop = asyncio.new_event_loop()
        asyncio.set
