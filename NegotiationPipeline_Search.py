import logging
import sys
import os
import asyncio
from typing import List, Iterator, Callable, Any
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from bs4 import BeautifulSoup
import requests

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4o"
        TEMPERATURE: float = 0.7
        MAX_TOKENS: int = 1500
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    def __init__(self):
        self.name = "Negotiation Strategy Predictor"
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

    trusted_sites = [
        "site:senate.parlam.kz", "site:akorda.kz", "site:primeminister.kz",
        "site:otyrys.prk.kz", "site:senate-zan.prk.kz", "site:lib.prk.kz",
        "site:online.zakon.kz", "site:adilet.zan.kz", "site:legalacts.egov.kz",
        "site:egov.kz", "site:eotinish.kz"
    ]

    def search_trusted_web(self, query: str, max_sources=3) -> List[tuple[str, str]]:
        # 1. Формируем поисковый запрос
        search_query = f'{query} ' + ' OR '.join(self.trusted_sites)
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(f"https://www.google.com/search?q={search_query}", headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        urls = []
        for a in soup.select("a"):
            href = a.get("href")
            if href and "/url?q=" in href:
                url = href.split("/url?q=")[1].split("&")[0]
                if any(domain in url for domain in [s.split(":")[1] for s in self.trusted_sites]):
                    urls.append(url)
            if len(urls) >= max_sources:
                break

        results = []
        for url in urls:
            try:
                page = requests.get(url, headers=headers, timeout=5)
                page_soup = BeautifulSoup(page.text, "html.parser")
                for tag in page_soup(["script", "style"]):
                    tag.extract()
                text = page_soup.get_text(separator="\n")
                cleaned = "\n".join([line.strip() for line in text.splitlines() if len(line.strip()) > 40])
                results.append((url, cleaned[:1500]))  # ограничим до 1500 символов
            except Exception as e:
                logging.warning(f"Failed to parse {url}: {e}")
                continue
        return results

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Iterator[str]:
        search_results = self.search_trusted_web(user_message)

        if not search_results:
            yield "Не удалось найти релевантную информацию на доверенных сайтах. Пожалуйста, уточните запрос."
            return

        web_context = "\n\n".join([f"Источник: {url}\n{text}" for url, text in search_results])

        system_message = f"""
**Роль:** Вы — эксперт по переговорам и стратегическому управлению. Ваша специализация — анализ и прогнозирование успешности различных моделей ведения переговоров, каналов коммуникации, а также построение компромиссных стратегий для разрешения конфликтов. Вы работаете строго в рамках переговорной аналитики, без отклонений в политические, бытовые, философские или технические темы.

**Область ответственности:**
Вы не отвечаете на вопросы, которые не касаются переговорных ситуаций, конфликтов интересов между сторонами, выбора стратегий убеждения, построения аргументов или оценки компромиссных решений. Если пользователь задаёт нерелевантный запрос, вы мягко перенаправляете его и предлагаете сформулировать переговорную ситуацию или проблему, с которой вы можете помочь.

**Пример реакции на нерелевантный вопрос:**
> «Прошу прощения, моя компетенция ограничена анализом переговоров, стратегий компромисса и оценки успешности коммуникационных моделей. Могу ли я помочь вам с анализом конкретной переговорной ситуации или конфликта интересов?»

**Цель:**
1. Анализировать представленные конфликтные или сложные переговорные ситуации.
2. Предлагать эффективные, взвешенные и компромиссные решения.
3. Прогнозировать успешность каждой переговорной модели с точки зрения устойчивости, выгод для сторон и долгосрочного эффекта.

**Структура ответа:**

### 1. Возможные компромиссные решения
- Конкретные, практически реализуемые предложения, учитывающие интересы обеих сторон.

### 2. Прогноз эффективности
- Оценка каждой предложенной модели или решения по таким критериям, как: устойчивость, масштабируемость, риски, потенциал реализации.

### 3. Рекомендации
- Какая модель наилучшим образом подойдёт в данной ситуации и почему.
- Уточняющие вопросы при необходимости.

**Дополнительные материалы:**
Вот выдержки из доверенных казахстанских источников, которые могут помочь в анализе:

{web_context}

На их основе сделайте анализ ситуации, обязательно указывая источники.
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

        # Wrap with retry logic
        return asyncio.run(self.make_request_with_retry(generate_stream))
