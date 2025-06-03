import logging
import sys
import os
import asyncio
import re
import httpx
import mimetypes
import base64
import io
from typing import List, Iterator, Callable, Any, Dict

from pydantic import BaseModel
from openai import OpenAI
from PIL import Image
import fitz
import docx2txt
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4.1"
        TEMPERATURE: float = 0.5
        MAX_TOKENS: int = 1800
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    def __init__(self):
        self.name = "ЛитГид"
        self.valves = self.Valves()
        self.client = OpenAI(api_key=self.valves.OPENAI_API_KEY)

    async def on_startup(self):
        logging.info("Hints pipeline warming up…")

    async def on_shutdown(self):
        logging.info("Hints pipeline shutting down…")

    async def make_request_with_retry(self, fn: Callable[[], Iterator[str]], retries=3) -> Iterator[str]:
        for attempt in range(retries):
            try:
                return fn()
            except Exception as e:
                logging.error(f"Attempt {attempt+1} failed: {e}")
                if attempt + 1 == retries:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def web_search(self, query: str) -> List[Dict[str, str]]:
        try:
            response = self.client.responses.create(
                model="gpt-4.1",
                tools=[{"type": "web_search_preview"}],
                input=query
            )
            return [{
                "title": "OpenAI Web Search",
                "link": "https://www.google.com/search?q=" + query.replace(" ", "+"),
                "snippet": response.output_text
            }]
        except Exception as e:
            logging.warning(f"OpenAI web_search_preview error: {e}")
            return []

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

        system_message = (
            "Вы — интеллектуальный помощник для законотворцев и аналитиков, специализирующийся на предоставлении релевантных подсказок, обзоров литературы и сравнительных анализов, основанных на международном опыте, официальных источниках и лучших практиках.\n"
            "Ваша задача — предложить релевантные академические и аналитические источники по теме проекта закона.\n"
            "**Цель:**\n"
            "1. Помочь пользователю найти качественную информацию и рекомендации по теме законодательной инициативы.\n"
            "2. Составить структурированное и аналитическое обоснование или обзор с учётом казахстанского и международного контекста.\n"
            "3. Привести ссылки на официальные или научные источники, если они доступны. \n"
            "**Формат ответа:\n**"
            "- Введение в тему\n"
            "- Краткий международный и казахстанский контекст\n"
            "- Сравнительный анализ (если применимо)\n"
            "- Практики, рекомендации или подсказки\n"
            "Обязательные шаги:\n"
            "1. Найти литературу: статьи, книги, доклады, аналитические отчёты.\n"
            "2. Для каждого источника укажите: название, краткое описание (аннотацию), ссылку.\n"
            "3. Разделите источники по типу: научные, аналитика, книги и т.д.\n"
            "4. Поддерживайте форматирование markdown или HTML.\n"
            "5. Добавьте краткую справку с рекомендациями по использованию найденной литературы."
        )

        async def _generate() -> str:
            try:
                results = await self.web_search(user_message)
                context = "\n".join([f"{r['title']}\n{r['link']}\n{r['snippet']}" for r in results])
                prompt = f"{system_message}\n\nЗапрос:\n{user_message}\n\nКонтекст из интернета:\n{context}"
                response = self.client.responses.create(
                    model="gpt-4.1",
                    tools=[{"type": "web_search_preview"}],
                    input=prompt
                )
                return response.output_text
            except Exception as e:
                logging.error(f"❌ Ошибка генерации: {e}")
                return "❌ Ошибка генерации ответа. Попробуйте ещё раз."

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_generate())
