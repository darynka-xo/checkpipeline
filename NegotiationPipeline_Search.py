import logging
import sys
import os
import asyncio
import json
import aiohttp
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class SearchResult(BaseModel):
    title: str
    link: str
    snippet: str
    position: int


class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(default_factory=list)
    error: Optional[str] = None


class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4o"
        TEMPERATURE: float = 0.7
        MAX_TOKENS: int = 1500
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")

    def __init__(self):
        self.name = "Negotiation Strategy Predictor"
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

    async def search_web(self, query: str, num_results: int = 3) -> SearchResponse:
        """
        Perform a web search with the Serper API, focusing on trusted sites.
        """
        # Generate site restriction using OR operator for trusted sites
        site_filter = " OR ".join(self.trusted_sites)
        enhanced_query = f"{query} ({site_filter})"

        logging.info(f"Searching for: {enhanced_query}")

        url = "https://google.serper.dev/search"

        headers = {
            'X-API-KEY': self.valves.SERPER_API_KEY,
            'Content-Type': 'application/json'
        }

        payload = {
            'q': enhanced_query,
            'gl': 'kz',  # Kazakhstan geolocation
            'hl': 'ru',  # Russian language
            'num': num_results
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logging.error(f"Search API returned status {response.status}: {error_text}")
                        return SearchResponse(error=f"Search API error: {response.status}")

                    data = await response.json()

                    # Parse organic results
                    results = []
                    if 'organic' in data:
                        for i, item in enumerate(data['organic']):
                            results.append(SearchResult(
                                title=item.get('title', 'No Title'),
                                link=item.get('link', ''),
                                snippet=item.get('snippet', 'No snippet available'),
                                position=i + 1
                            ))

                    return SearchResponse(results=results)

        except Exception as e:
            logging.error(f"Error during web search: {str(e)}")
            return SearchResponse(error=f"Search failed: {str(e)}")

    def format_search_results(self, search_response: SearchResponse) -> str:
        """
        Format search results for inclusion in the system prompt.
        """
        if search_response.error:
            return f"Поиск не удался с ошибкой: {search_response.error}"

        if not search_response.results:
            return "Поиск не дал результатов. Используйте только имеющиеся у вас знания."

        results_text = "### Результаты поиска из проверенных источников:\n\n"

        for i, result in enumerate(search_response.results):
            results_text += f"**Источник {i + 1}: {result.title}**\n"
            results_text += f"URL: {result.link}\n"
            results_text += f"Содержание: {result.snippet}\n\n"

        results_text += "При использовании информации из источников, обязательно укажите номер источника в скобках, например: [Источник 1].\n\n"

        return results_text

    async def pipe(self, user_message: str, model_id: str = None, messages: List[dict] = None,
                   body: dict = None) -> str:
        """
        Простая версия без стриминга, возвращающая полный ответ сразу
        """
        try:
            # Выполняем поиск на основе запроса пользователя
            search_response = await self.search_web(user_message)
            search_results_text = self.format_search_results(search_response)

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

**Источники информации:**
Используйте предоставленные результаты поиска для обогащения ваших ответов. Обязательно цитируйте источники в вашем ответе, используя формат [Источник X], где X - номер источника.

{search_results}

**Структура ответа:**

### 1. Возможные компромиссные решения
- Конкретные, практически реализуемые предложения, учитывающие интересы обеих сторон.

### 2. Прогноз эффективности
- Оценка каждой предложенной модели или решения по таким критериям, как: устойчивость, масштабируемость, риски, потенциал реализации.

### 3. Рекомендации
- Какая модель наилучшим образом подойдёт в данной ситуации и почему.
- Уточняющие вопросы при необходимости.

### 4. Использованные источники
- Перечислите номера и названия использованных источников.
"""
            # Вставляем результаты поиска в системное сообщение
            system_message = system_message.format(search_results=search_results_text)

            # Инициализируем модель без стриминга
            model = ChatOpenAI(
                api_key=self.valves.OPENAI_API_KEY,
                model=self.valves.MODEL_ID if not model_id else model_id,
                temperature=self.valves.TEMPERATURE,
                max_tokens=self.valves.MAX_TOKENS,
                streaming=False  # Отключаем стриминг
            )

            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_message),
                HumanMessagePromptTemplate.from_template("{user_input}")
            ])

            formatted_messages = prompt.format_messages(user_input=user_message)

            # Получаем полный ответ сразу
            response = model.invoke(formatted_messages)
            full_response = response.content

            logging.info("Got complete response")
            return full_response

        except Exception as e:
            logging.error(f"Error in pipeline: {str(e)}")
            return f"Извините, произошла ошибка при обработке запроса: {str(e)}"


# Эта функция нужна для правильной обработки pipeline в OpenWebUI
async def filter_inlet(body, context):
    pipeline = Pipeline()
    await pipeline.on_startup()

    user_message = body.get("prompt", "")
    model_id = body.get("model", None)
    messages = body.get("messages", [])

    try:
        # Просто получаем полный ответ
        response = await pipeline.pipe(user_message, model_id, messages, body)
        logging.info("Full response generated")
        return response
    except Exception as e:
        logging.error(f"Error in filter_inlet: {str(e)}")
        return f"Error: {str(e)}"
