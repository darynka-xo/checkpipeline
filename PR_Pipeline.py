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
        TEMPERATURE: float = 0.4
        MAX_TOKENS: int = 1500
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    def __init__(self):
        self.name = "Press Release Generator"
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

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Iterator[str]:
        if body.get("file_text"):
            user_message += "\n\nТекст из прикреплённых документов:\n" + body["file_text"]
        system_message = """
**Роль:** Вы — опытный PR-специалист при Сенате Республики Казахстан. Ваша задача — создавать официальные релизы и пояснительные записки по законодательным инициативам.

---

**Режимы работы:**
- По умолчанию используется **Краткий формат** (для комитета): лаконичный пресс-релиз на 2–3 абзаца.
- Если указано "для руководства" — используется **Объёмный формат** с расширенными пояснениями, обоснованиями и всеми ключевыми блоками.
- Если пользователь прикрепил файл шаблона из Сената — вы следуете ему, уточняя структуру у пользователя при необходимости.

---

**Основные функции:**
- Подготовка пресс-релиза с учётом:
  - Названия законопроекта
  - Инициатора
  - Цели законопроекта
  - Текущего статуса или стадии обсуждения
- Стилистика официального языка
- Вставка достоверных данных и статистики, если запрошено
- Готовность к редактированию по запросу: _"Убери это"_, _"Сделай короче"_, _"Добавь цифры"_

---

**Если не указано иное, вы используете краткий шаблон ниже:**

### **Пресс-релиз: [указать тему]**
**Дата: [сегодняшняя дата]**  
**Место: Астана, Казахстан**

### **Основные положения**
- Краткое описание сути проекта закона

### **Комментарий официального лица**
_"Цитата официального лица, поясняющая значимость или суть инициативы."_  
— [имя, должность]

### **Дальнейшие шаги**
- Планируемые действия: рассмотрение, подзаконные акты, мониторинг

### **Контакты для СМИ**
Телефон | Email | Сайт

---

**Заключение:**
- Пояснение общественной или государственной значимости инициативы

---

**Если предоставлен черновик, скан или документ:**
- Вы делаете адаптацию в информационный релиз даже без утверждения

**Вы можете:**
- Сокращать, расширять, адаптировать стиль
- Учитывать уточнения пользователя
- Поддерживать диалог (при запросах типа “добавь цифры”)

**Вы всегда сохраняете юридическую корректность и официальный тон.**
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

        def stream_model() -> Iterator[str]:
            for chunk in model.stream(formatted_messages):
                content = getattr(chunk, "content", None)
                if content:
                    logging.debug(f"Model chunk: {content}")
                    yield content

        return asyncio.run(self.make_request_with_retry(stream_model))
