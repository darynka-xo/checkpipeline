import logging
import sys
import os
import asyncio
from typing import List, Iterator, Callable, Any, AsyncGenerator
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

    async def search_trusted_web(self, query: str, max_sources=3) -> List[tuple[str, str]]:
        """
        Searches trusted Kazakhstani government websites for information related to the query.
        Uses async requests for better performance and proper error handling.

        Args:
            query: The search query
            max_sources: Maximum number of sources to retrieve

        Returns:
            List of tuples containing (url, content)
        """
        import aiohttp
        import asyncio
        from urllib.parse import quote_plus

        # Format the search query with trusted sites
        search_query = quote_plus(f'{query} ' + ' OR '.join(self.trusted_sites))
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

        trusted_domains = [s.split(":")[1] for s in self.trusted_sites]
        urls = []

        try:
            async with aiohttp.ClientSession() as session:
                # Search for URLs
                async with session.get(
                        f"https://www.google.com/search?q={search_query}&num=20",
                        headers=headers,
                        timeout=10
                ) as response:
                    if response.status != 200:
                        logging.warning(f"Google search returned status code {response.status}")
                        return []

                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")

                    # Find search result links - more specific selector
                    for div in soup.select("div.yuRUbf"):
                        a_tag = div.find('a')
                        if a_tag and a_tag.get('href'):
                            url = a_tag.get('href')
                            # Verify it's from a trusted domain
                            if any(domain in url for domain in trusted_domains):
                                urls.append(url)
                                if len(urls) >= max_sources:
                                    break

                    # Fallback to broader search if specific selector fails
                    if not urls:
                        for a in soup.select("a"):
                            href = a.get("href")
                            if href and "/url?q=" in href:
                                url = href.split("/url?q=")[1].split("&")[0]
                                if any(domain in url for domain in trusted_domains):
                                    urls.append(url)
                                    if len(urls) >= max_sources:
                                        break

                if not urls:
                    logging.warning("No trusted URLs found in search results")
                    return []

                # Fetch content from found URLs
                async def fetch_url(url):
                    try:
                        async with session.get(url, headers=headers, timeout=15) as resp:
                            if resp.status != 200:
                                logging.warning(f"Failed to fetch {url}: Status {resp.status}")
                                return None

                            content = await resp.text()
                            page_soup = BeautifulSoup(content, "html.parser")

                            # Remove unwanted elements
                            for tag in page_soup(["script", "style", "nav", "footer", "header"]):
                                tag.extract()

                            # Get main content
                            main_content = page_soup.find("main") or page_soup.find("article") or page_soup
                            text = main_content.get_text(separator="\n")

                            # Clean and process the text
                            lines = text.splitlines()
                            cleaned_lines = []
                            for line in lines:
                                line = line.strip()
                                # Keep informative lines (not too short, not just punctuation)
                                if len(line) > 20 and any(c.isalpha() for c in line):
                                    cleaned_lines.append(line)

                            cleaned_text = "\n".join(cleaned_lines)

                            # Smartly limit text length to avoid cutting in the middle of sentences
                            if len(cleaned_text) > 1500:
                                # Find a good breakpoint
                                breakpoint = cleaned_text[:1500].rfind('.')
                                if breakpoint == -1:
                                    breakpoint = cleaned_text[:1500].rfind('\n')
                                if breakpoint == -1:
                                    breakpoint = 1500

                                cleaned_text = cleaned_text[:breakpoint + 1]

                            return url, cleaned_text
                    except asyncio.TimeoutError:
                        logging.warning(f"Timeout while fetching {url}")
                        return None
                    except Exception as e:
                        logging.warning(f"Failed to parse {url}: {str(e)}")
                        return None

                # Process URLs in parallel with rate limiting
                tasks = []
                for url in urls:
                    tasks.append(fetch_url(url))
                    # Add a small delay between requests to avoid rate limiting
                    await asyncio.sleep(0.5)

                results = await asyncio.gather(*tasks)

                # Filter out None results and return
                return [r for r in results if r is not None]

        except Exception as e:
            logging.error(f"Error in search_trusted_web: {str(e)}")
            return []

    async def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> AsyncGenerator[str, None]:
        logging.info(f"Searching trusted web for: {user_message}")
        search_results = await self.search_trusted_web(user_message)

        if not search_results:
            yield "Не удалось найти релевантную информацию на доверенных сайтах. Пожалуйста, уточните запрос или опишите ситуацию подробнее."
            return

        web_context_parts = []
        for idx, (url, text) in enumerate(search_results):
            domain = url.split("//")[-1].split("/")[0]
            web_context_parts.append(f"Источник {idx + 1} ({domain}):\n{text}")

        web_context = "\n\n".join(web_context_parts)

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
            model=model_id or self.valves.MODEL_ID,
            temperature=self.valves.TEMPERATURE,
            max_tokens=self.valves.MAX_TOKENS,
            streaming=True
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template("{user_input}")
        ])

        formatted_messages = prompt.format_messages(user_input=user_message)

        # Use proper async streaming
        async def generate_stream():
            async for chunk in model.astream(formatted_messages):
                content = getattr(chunk, "content", None)
                if content:
                    logging.debug(f"Model chunk: {content}")
                    yield content

        # Return the stream directly
        async for chunk in generate_stream():
            yield chunk
