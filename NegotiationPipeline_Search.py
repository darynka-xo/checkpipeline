import logging
import sys
import os
import asyncio
from typing import List, Iterator, Callable, Any, Dict, Optional, Union, AsyncIterator
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
import json
import httpx
import re

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4o"
        TEMPERATURE: float = 0.7
        MAX_TOKENS: int = 1500
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")  # API ключ для Serper.dev

    def __init__(self):
        self.name = "Negotiation Strategy Predictor with Evidence"
        self.valves = self.Valves()
        self.trusted_sites = [
            "site:senate.parlam.kz", "site:akorda.kz", "site:primeminister.kz",
            "site:otyrys.prk.kz", "site:senate-zan.prk.kz",
            "site:lib.prk.kz", "site:online.zakon.kz", "site:adilet.zan.kz",
            "site:legalacts.egov.kz", "site:egov.kz", "site:eotinish.kz"
        ]

    async def on_startup(self):
        logging.info("Pipeline is warming up...")

    async def on_shutdown(self):
        logging.info("Pipeline is shutting down...")

    async def make_request_with_retry(
            self,
            fn: Callable[[], AsyncIterator[str]],
            retries: int = 3,
    ) -> AsyncIterator[str]:
        for attempt in range(retries):
            try:
                return fn()  # теперь это AsyncIterator
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt + 1 == retries:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Выполняет поиск в интернете с использованием Serper API с фокусом на доверенные источники.
        """
        # Подготовка запросов для поиска по каждому доверенному сайту
        search_queries = []

        # Сначала добавляем общий запрос без ограничений по сайтам
        search_queries.append(query)

        # Затем добавляем запросы для каждого доверенного сайта
        for site in self.trusted_sites:
            site_query = f"{query} {site}"
            search_queries.append(site_query)

        all_results = []

        # Выполняем запросы последовательно
        for search_query in search_queries:
            logging.info(f"Выполняем поиск: {search_query}")

            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "https://google.serper.dev/search",
                        headers={
                            "X-API-KEY": self.valves.SERPER_API_KEY,
                            "Content-Type": "application/json"
                        },
                        json={
                            "q": search_query,
                            "gl": "kz",  # Геолокация - Казахстан
                            "hl": "ru",  # Язык - русский
                            "num": 3  # Уменьшаем количество результатов для каждого запроса
                        }
                    )

                    if response.status_code != 200:
                        logging.error(f"Ошибка API поиска: {response.status_code}, {response.text}")
                        continue

                    search_results = response.json()

                    # Извлекаем и форматируем результаты
                    if "organic" in search_results:
                        for result in search_results["organic"]:
                            # Создаем уникальный идентификатор для результата
                            result_id = result.get("link", "")

                            # Проверяем, нет ли уже этого результата в списке
                            if not any(r["link"] == result_id for r in all_results):
                                # Проверяем, является ли источник доверенным
                                domain = self.extract_domain(result_id)
                                is_trusted = any(
                                    trusted_site.split(":")[1] in domain for trusted_site in self.trusted_sites)

                                # Добавляем результат в общий список
                                all_results.append({
                                    "index": len(all_results),
                                    "title": result.get("title", ""),
                                    "link": result_id,
                                    "snippet": result.get("snippet", ""),
                                    "source": domain,
                                    "is_trusted": is_trusted
                                })

                    # Не делаем слишком много запросов подряд
                    await asyncio.sleep(0.5)

            except Exception as e:
                logging.error(f"Ошибка при выполнении поиска: {str(e)}")

        # Сортируем результаты: сначала доверенные источники
        all_results.sort(key=lambda x: (not x.get("is_trusted"), x.get("index")))

        # Обновляем индексы после сортировки
        for i, result in enumerate(all_results):
            result["index"] = i + 1  # Начинаем индексацию с 1 для удобства пользователя

        # Возвращаем ограниченное количество результатов
        return all_results[:num_results]

    def extract_domain(self, url: str) -> str:
        """Извлекает домен из URL."""
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if match:
            return match.group(1)
        return url

    async def fetch_page_content(self, url: str) -> str:
        """
        Получает содержимое веб-страницы по URL.
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                if response.status_code == 200:
                    # Можно добавить парсинг HTML, чтобы извлечь чистый текст
                    return response.text
                else:
                    logging.error(f"Failed to fetch content from {url}: {response.status_code}")
                    return ""
        except Exception as e:
            logging.error(f"Error fetching page content: {e}")
            return ""

    def format_search_results_for_prompt(self, results: List[Dict]) -> str:
        """
        Форматирует результаты поиска для включения в промпт.
        Каждый результат представлен в виде структурированного текста с уникальным индексом.
        """
        if not results:
            return "Информация из доверенных источников не найдена."

        formatted_text = "### Информация из доверенных источников:\n\n"

        for result in results:
            index = result['index']
            title = result.get('title', 'Заголовок не указан')
            source = result.get('source', 'Источник не указан')
            link = result.get('link', '#')
            snippet = result.get('snippet', 'Описание отсутствует')

            # Разбиваем снипет на предложения для удобства цитирования
            sentences = re.split(r'(?<=[.!?])\s+', snippet)
            sentences_text = ""

            for i, sentence in enumerate(sentences):
                if sentence.strip():  # Пропускаем пустые предложения
                    sentences_text += f"{index}-{i + 1}: {sentence.strip()}\n"

            formatted_text += f"[{index}] {title}\n"
            formatted_text += f"Источник: {source}\n"
            formatted_text += f"Ссылка: {link}\n"
            formatted_text += f"Текст:\n{sentences_text}\n"

        return formatted_text

    def create_citation_instructions(self) -> str:
        """
        Создает инструкции по цитированию для включения в системный промпт.
        """
        return """
**Цитирование источников:**

Всегда подкрепляйте свои ключевые утверждения и рекомендации цитатами из доверенных источников.
Используйте следующий формат цитирования:

- Для прямых цитат: "Прямая цитата" [Номер источника]
- Для перефразирования: Перефразированная информация [Номер источника]

В конце ответа укажите список использованных источников в формате:

**Источники:**
[1] Название источника, URL
[2] Название источника, URL
...

Если для ответа на вопрос пользователя нет достаточной информации в предоставленных источниках, 
укажите это и используйте свои знания, но отметьте, что эта часть ответа не подтверждена 
официальными источниками Казахстана.
"""

    async def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> AsyncIterator[str]:

        # Выполняем поиск по запросу пользователя
        search_results = await self.search_web(user_message)
        search_content = self.format_search_results_for_prompt(search_results)

        system_message = """
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

* Конкретные, практически реализуемые предложения, учитывающие интересы обеих сторон.

### 2. Прогноз эффективности

* Оценка каждой предложенной модели или решения по таким критериям, как: устойчивость, масштабируемость, риски, потенциал реализации.

### 3. Рекомендации

* Какая модель наилучшим образом подойдёт в данной ситуации и почему.
* Уточняющие вопросы при необходимости.
"""

        # Добавляем инструкции по цитированию и результаты поиска
        system_message += self.create_citation_instructions()
        system_message += f"\n\n{search_content}"

        model = ChatOpenAI(
            api_key=self.valves.OPENAI_API_KEY,
            model=self.valves.MODEL_ID,
            temperature=self.valves.TEMPERATURE,
            streaming=True,
            max_tokens=self.valves.MAX_TOKENS
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template("{user_input}")
        ])

        formatted_messages = prompt.format_messages(user_input=user_message)

        async def generate_stream() -> AsyncIterator[str]:
            async for chunk in model.astream(formatted_messages):
                content = getattr(chunk, "content", None)
                if content:
                    logging.debug(f"Model chunk: {content}")
                    yield content

        # Wrap with retry logic
        return await self.make_request_with_retry(generate_stream)
