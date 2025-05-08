import logging
import sys
import os
import asyncio
from typing import List, Iterator, Callable, Any
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

trusted_sites = [
    "site:senate.parlam.kz", "site:akorda.kz", "site:primeminister.kz",
    "site:otyrys.prk.kz", "site:senate-zan.prk.kz",
    "site:lib.prk.kz", "site:online.zakon.kz", "site:adilet.zan.kz",
    "site:legalacts.egov.kz", "site:egov.kz", "site:eotinish.kz"
]

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
        self.search_tool = DuckDuckGoSearchRun()

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

    async def _search_and_cite(self, query: str) -> List[Document]:
        search_results = await asyncio.to_thread(self.search_tool.run, f"{query} {' '.join(trusted_sites)}")
        documents = []
        if search_results:
            for i, res in enumerate(search_results.split("\n")):
                if res.startswith("[") and "]" in res and " - " in res:
                    try:
                        source_num_str = res[res.find("[") + 1:res.find("]")]
                        source_num = int(source_num_str)
                        content_start = res.find(" - ") + 3
                        content = res[content_start:]
                        link_start = content.find("http")
                        link = content[link_start:] if link_start != -1 else "No link found"
                        page_content = content[:link_start].strip() if link_start != -1 else content.strip()
                        metadata = {"source": link}
                        documents.append(Document(page_content=page_content, metadata=metadata))
                    except ValueError:
                        logging.warning(f"Could not parse search result: {res}")
                    except IndexError:
                        logging.warning(f"Could not parse search result: {res}")
        return documents

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Iterator[str]:
        system_message_template = """
        **Роль:** Вы — эксперт по переговорам и стратегическому управлению. Ваша специализация — анализ и прогнозирование успешности различных моделей ведения переговоров, каналов коммуникации, а также построение компромиссных стратегий для разрешения конфликтов. Вы работаете строго в рамках переговорной аналитики, без отклонений в политические, бытовые, философские или технические темы. При необходимости, вы можете использовать предоставленные контекстные документы для формирования аргументов и цитирования источников.

        **Область ответственности:**
        Вы не отвечаете на вопросы, которые не касаются переговорных ситуаций, конфликтов интересов между сторонами, выбора стратегий убеждения, построения аргументов или оценки компромиссных решений. Если пользователь задаёт нерелевантный запрос, вы мягко перенаправляете его и предлагаете сформулировать переговорную ситуацию или проблему, с которой вы можете помочь.

        **Пример реакции на нерелевантный вопрос:**
        > «Прошу прощения, моя компетенция ограничена анализом переговоров, стратегиями компромисса и оценки успешности коммуникационных моделей. Могу ли я помочь вам с анализом конкретной переговорной ситуации или конфликта интересов?»

        **Цель:**
        1. Анализировать представленные конфликтные или сложные переговорные ситуации.
        2. Предлагать эффективные, взвешенные и компромиссные решения, подкрепленные аргументами из авторитетных источников.
        3. Прогнозировать успешность каждой переговорной модели с точки зрения устойчивости, выгод для сторон и долгосрочного эффекта.
        4. Цитировать источники, подтверждающие ваши аргументы, в следующем формате: ([номер источника]: краткое описание источника). Полный список источников приводится в конце ответа.

        **Структура ответа:**
        ### 1. Возможные компромиссные решения
        - Конкретные, практически реализуемые предложения, учитывающие интересы обеих сторон, с аргументацией и цитатами.
        ### 2. Прогноз эффективности
        - Оценка каждой предложенной модели или решения по таким критериям, как: устойчивость, масштабируемость, риски, потенциал реализации.
        ### 3. Рекомендации
        - Какая модель наилучшим образом подойдёт в данной ситуации и почему.
        - Уточняющие вопросы при необходимости.
        ### 4. Источники
        - Полный список использованных источников в формате: [номер источника]: [полная ссылка на источник] - [краткое описание].
        """
        model = ChatOpenAI(
            api_key=self.valves.OPENAI_API_KEY,
            model=self.valves.MODEL_ID,
            temperature=self.valves.TEMPERATURE,
            streaming=True
        )
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_message_template),
            HumanMessagePromptTemplate.from_template("{user_input}")
        ])

        async def enhanced_generate(user_input: str) -> Iterator[str]:
            search_results = await self._search_and_cite(user_input)
            context = "\n".join([f"{doc.page_content} (Источник: {doc.metadata['source']})" for doc in search_results])

            enhanced_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_message_template + "\n\n**Контекст:**\n{context}"),
                HumanMessagePromptTemplate.from_template("{user_input}")
            ])
            formatted_messages = await enhanced_prompt.ainvoke({"user_input": user_input, "context": context})

            def generate_stream() -> Iterator[str]:
                full_response = ""
                for chunk in model.stream(formatted_messages):
                    content = getattr(chunk, "content", None)
                    if content:
                        logging.debug(f"Model chunk: {content}")
                        nonlocal full_response
                        full_response += content
                        yield content

                # After the stream is complete, append the sources
                if search_results:
                    yield "\n\n### 4. Источники\n"
                    for i, doc in enumerate(search_results):
                        yield f"[{i+1}]: {doc.metadata['source']} - {doc.page_content[:100]}...\n" # Add a brief description

            # Wrap with retry logic
            return self.make_request_with_retry(generate_stream)

        return enhanced_generate(user_message)
