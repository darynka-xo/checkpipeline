import logging
import sys
import os
import asyncio
import json
import re
from typing import List, Iterator, Callable, Any, Dict, Optional
from pydantic import BaseModel, Field
import aiohttp
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


class SerperSearchResults(BaseModel):
    results: List[SearchResult] = Field(default_factory=list)
    query: str = ""
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

    async def make_request_with_retry(self, fn: Callable[[], Iterator[str]], retries=3) -> Iterator[str]:
        for attempt in range(retries):
            try:
                return fn()
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt + 1 == retries:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def search_serper(self, query: str, num_results: int = 5) -> SerperSearchResults:
        """
        Perform a web search using Serper API with trusted sites
        """
        if not self.valves.SERPER_API_KEY:
            return SerperSearchResults(
                query=query,
                error="SERPER_API_KEY is not set in environment variables"
            )

        # Add trusted sites to the query
        site_query = " OR ".join(self.trusted_sites)
        enhanced_query = f"{query} ({site_query})"
        logging.debug(f"Enhanced search query: {enhanced_query}")

        headers = {
            "X-API-KEY": self.valves.SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "q": enhanced_query,
            "num": num_results
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://google.serper.dev/search", 
                    headers=headers, 
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract organic search results
                        results = []
                        if "organic" in data:
                            for i, item in enumerate(data["organic"]):
                                results.append(
                                    SearchResult(
                                        title=item.get("title", ""),
                                        link=item.get("link", ""),
                                        snippet=item.get("snippet", ""),
                                        position=i + 1
                                    )
                                )
                        
                        return SerperSearchResults(
                            results=results,
                            query=query
                        )
                    else:
                        error_text = await response.text()
                        return SerperSearchResults(
                            query=query,
                            error=f"Error: {response.status}, {error_text}"
                        )
        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return SerperSearchResults(
                query=query,
                error=f"Exception during search: {str(e)}"
            )

    def format_search_results(self, search_results: SerperSearchResults) -> str:
        """Format search results for inclusion in the prompt"""
        if search_results.error:
            return f"Search error: {search_results.error}"
        
        if not search_results.results:
            return "No relevant search results found."
        
        formatted_results = "### Результаты поиска из официальных источников:\n\n"
        
        for i, result in enumerate(search_results.results, 1):
            formatted_results += f"{i}. **Источник**: {result.title}\n"
            formatted_results += f"   **Ссылка**: {result.link}\n"
            formatted_results += f"   **Фрагмент**: {result.snippet}\n\n"
        
        return formatted_results

    def create_citation_instructions(self) -> str:
        """Create instructions for the model to properly cite sources"""
        return """
**Инструкции по цитированию:**
1. Используйте информацию из предоставленных источников для обоснования ваших рекомендаций.
2. При использовании информации из конкретного источника, укажите номер источника в квадратных скобках, например: [1].
3. Если вы комбинируете информацию из нескольких источников, укажите все источники: [1, 3, 5].
4. В конце ответа приведите список использованных источников с указанием их названий и ссылок.

**Пример цитирования:**
"Согласно законодательству РК о медиации, стороны имеют право самостоятельно выбирать медиатора [2]."

**Список источников:**
1. [Название источника 1](ссылка1)
2. [Название источника 2](ссылка2)
"""

    async def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Iterator[str]:
        # Perform web search to gather relevant information
        search_results = await self.search_serper(user_message)
        formatted_search_results = self.format_search_results(search_results)
        citation_instructions = self.create_citation_instructions()

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

{citation_instructions}

**Структура ответа:**

### 1. Возможные компромиссные решения
- Конкретные, практически реализуемые предложения, учитывающие интересы обеих сторон.

### 2. Прогноз эффективности
- Оценка каждой предложенной модели или решения по таким критериям, как: устойчивость, масштабируемость, риски, потенциал реализации.

### 3. Рекомендации
- Какая модель наилучшим образом подойдёт в данной ситуации и почему.
- Уточняющие вопросы при необходимости.

### 4. Использованные источники
- Список всех источников, которые вы использовали в своем ответе, с указанием названий и ссылок.

**Информация из официальных источников:**
{formatted_search_results}
"""

        model = ChatOpenAI(
            api_key=self.valves.OPENAI_API_KEY,
            model=self.valves.MODEL_ID if model_id is None else model_id,
            temperature=self.valves.TEMPERATURE,
            max_tokens=self.valves.MAX_TOKENS,
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
        return await self.make_request_with_retry(generate_stream)

    def process_citations(self, response: str, search_results: SerperSearchResults) -> str:
        """
        Process the response to ensure citations are properly formatted and linked
        """
        if not search_results.results:
            return response
            
        # Replace citation numbers with hyperlinks
        pattern = r'\[(\d+(?:,\s*\d+)*)\]'
        
        def replace_citation(match):
            citation_numbers = [int(num.strip()) for num in match.group(1).split(',')]
            links = []
            
            for num in citation_numbers:
                if 1 <= num <= len(search_results.results):
                    result = search_results.results[num-1]
                    links.append(f'<a href="{result.link}" target="_blank">[{num}]</a>')
                else:
                    links.append(f'[{num}]')
                    
            return ' '.join(links)
            
        return re.sub(pattern, replace_citation, response)
        
    async def process_request(self, user_message: str, model_id: str = None) -> str:
        """
        Process a request end-to-end including search, generation, and citation processing
        """
        # Perform search
        search_results = await self.search_serper(user_message)
        
        # Generate response stream
        response_stream = await self.pipe(user_message, model_id, [], {})
        
        # Collect full response
        full_response = ''.join(list(response_stream))
        
        # Process citations in the response
        processed_response = self.process_citations(full_response, search_results)
        
        return processed_response
