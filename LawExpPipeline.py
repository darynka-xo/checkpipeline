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
from openai import OpenAI
from PIL import Image
import fitz
import docx2txt

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

TRUSTED_DOMAINS = [
    "akorda.kz", "senate.parlam.kz", "primeminister.kz",
    "otyrys.prk.kz", "senate-zan.prk.kz", "lib.prk.kz",
    "online.zakon.kz", "adilet.zan.kz", "legalacts.egov.kz",
    "egov.kz", "eotinish.kz"
]

def _is_trusted(url: str) -> bool:
    return any(d in url for d in TRUSTED_DOMAINS)

def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

async def web_search(query: str) -> List[Dict[str, str]]:
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.responses.create(
            model="gpt-4.1",
            tools=[{"type": "web_search_preview"}],
            input=query
        )
        return [{
            "title": "Поиск OpenAI",
            "link": "https://www.google.com/search?q=" + query.replace(" ", "+"),
            "snippet": response.output_text
        }]
    except Exception as e:
        logging.warning(f"OpenAI web_search_preview error: {e}")
        return []

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

class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4.1"
        TEMPERATURE: float = 0.3
        MAX_TOKENS: int = 1800
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    def __init__(self):
        self.name = "Эксперт по предложениям"
        self.valves = self.Valves()
        self.client = OpenAI(api_key=self.valves.OPENAI_API_KEY)

    async def on_startup(self):
        logging.info("LawExp pipeline warming up…")

    async def on_shutdown(self):
        logging.info("LawExp pipeline shutting down…")

    async def make_request_with_retry(self, fn: Callable[[], Iterator[str]], retries=3) -> Iterator[str]:
        for attempt in range(retries):
            try:
                return fn()
            except Exception as e:
                logging.error(f"Attempt {attempt+1} failed: {e}")
                if attempt + 1 == retries:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def inlet(self, body: dict, user: dict) -> dict:
        import json
        logging.info("📥 Inlet body:\n" + json.dumps(body, indent=2, ensure_ascii=False))

        extracted = []
        for f in body.get("files", []):
            url = f.get("url", "")
            content = None

            if url.startswith("http://") or url.startswith("https://"):
                content_url = url + "/content"
                async with httpx.AsyncClient(timeout=30) as c:
                    resp = await c.get(content_url)
                    resp.raise_for_status()
                    content = resp.content
            elif url.startswith("data:"):
                # Пример: data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,...
                header, b64data = url.split(",", 1)
                content = base64.b64decode(b64data)
            else:
                logging.warning(f"⚠️ Unsupported or missing URL scheme: {url}")
                continue

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
        return body


    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Iterator[str]:
        if body.get("file_text"):
            if not user_message.strip():
                user_message = body["file_text"]
            else:
                user_message += "\n\nТекст из прикреплённых документов:\n" + body["file_text"]

        system_msg = (
            "Ты — ИИ-аналитик по правовым вопросам, владеющий казахским и русским языками. Твоя задача — провести сравнительный анализ между предлагаемым законопроектом и действующими казахстанскими законами.\n"
            "Обязательные шаги:\n"
            "1. Найди существующие законы, статьи или пункты, которые ПЕРЕСЕКАЮТСЯ или ДУБЛИРУЮТ предлагаемый проект.\n"
            "2. Для каждого совпадения укажи: название закона, номер статьи, и сформируй гиперссылку на официальный источник (если есть).\n"
            "3. Если на adilet.zan.kz есть история правок этой нормы — кратко опиши её.\n"
            "4. После таблицы представь аналитическую справку (можно в формате markdown или HTML), где сделай выводы о потенциальных конфликтах, юридических рисках и возможностях улучшения проекта.\n"
            "5. Учитывай, если пользователь хочет видеть только определённый аспект (например, только дублирующие нормы или только правки).\n"
            "Таблица формата: | № | Статья проекта | Дублирующая норма | Источник | Правки | Ссылка | Комментарий |\n"
            "Не придумывай статьи. Отвечай строго по законам."
        )

        # response = self.client.responses.create(
        #         model="gpt-4.1",
        #         tools=[{"type": "web_search_preview"}],
        #         input=f"{system_msg}\n\n<проект>\n{user_message}",
        #         stream=True,
        #     )
        headers = {
            "Content-Type": "application/json"
        }
        headers["Authorization"] = f"Bearer {self.valves.OPENAI_API_KEY}"
        payload = {
            "model": 'gpt-4.1',
            "input": f"{system_msg}\n\n<проект>\n{user_message}",
            "tools": [
                {"type": "web_search_preview"}
            ],
            "stream": True
        }
        response = requests.post(
            url=f"https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            stream=True
        )

        for line in response.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("type") == "response.output_text.delta":
                yield event["delta"]
            elif event.get("type") == "response.output_text.done":
                pass