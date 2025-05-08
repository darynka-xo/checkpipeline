import logging
import sys
import os
import asyncio
from typing import List, Iterator, Callable, Dict
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import requests

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Доверенные домены для поиска
trusted_sites: List[str] = [
    "site:senate.parlam.kz",
    "site:akorda.kz",
    "site:primeminister.kz",
    "site:otyrys.prk.kz",
    "site:senate-zan.prk.kz",
    "site:lib.prk.kz",
    "site:online.zakon.kz",
    "site:adilet.zan.kz",
    "site:legalacts.egov.kz",
    "site:egov.kz",
    "site:eotinish.kz",
]


class Pipeline:
    class Valves(BaseModel):
        MODEL_ID: str = "gpt-4o"
        TEMPERATURE: float = 0.7
        MAX_TOKENS: int = 1500
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")  # ключ Serper.dev

    def __init__(self):
        self.name = "Negotiation Strategy Predictor"
        self.valves = self.Valves()

        # HTTP-сессия Serper с заголовком ключа
        self._serper = requests.Session()
        self._serper.headers.update(
            {
                "X-API-KEY": self.valves.SERPER_API_KEY,
                "Content-Type": "application/json",
            }
        )

    # --------------------------------------------------------------------- #
    #                        инфраструктурные события                       #
    # --------------------------------------------------------------------- #
    async def on_startup(self):
        logging.info("Pipeline is warming up...")

    async def on_shutdown(self):
        logging.info("Pipeline is shutting down...")

    # --------------------------------------------------------------------- #
    #                        вспомогательные функции                        #
    # --------------------------------------------------------------------- #
    async def make_request_with_retry(
        self, fn: Callable[[], Iterator[str]], retries: int = 3
    ) -> Iterator[str]:
        """Повторяет вызов генератора-стрима в случае нештатных ошибок."""
        for attempt in range(1, retries + 1):
            try:
                return fn()
            except Exception as e:
                logging.error(f"Attempt {attempt} failed: {e}")
                if attempt == retries:
                    raise
                await asyncio.sleep(2**attempt)

    # ------------------------- поиск и форматирование -------------------- #
    def _search_serper(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Делает запрос к Serper.dev и возвращает список результатов
        только с доверенных сайтов.
        """
        q = f'{query} {" OR ".join(trusted_sites)}'
        resp = self._serper.post(
            "https://google.serper.dev/search", json={"q": q}, timeout=20
        )
        resp.raise_for_status()
        data = resp.json()

        results: List[Dict[str, str]] = []
        for item in data.get("organic", [])[:limit]:
            link = item.get("link", "")
            if any(domain.replace("site:", "") in link for domain in trusted_sites):
                results.append(
                    {
                        "title": item.get("title", "").strip(),
                        "link": link,
                        "snippet": item.get("snippet", "").strip(),
                    }
                )
        return results

    def _format_evidence(self, results: List[Dict[str, str]]) -> str:
        """Преобразует результаты поиска в блок, понятный модели."""
        if not results:
            return "Нет релевантных источников из доверенного списка."

        lines: List[str] = []
        for idx, res in enumerate(results, start=1):
            lines.append(
                f"[{idx}] {res['title']} — {res['link']}\n{res['snippet']}"
            )
        return "\n\n".join(lines)

    # --------------------------------------------------------------------- #
    #                               основной API                            #
    # --------------------------------------------------------------------- #
    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Iterator[str]:
        # 1) Собираем доказательства
        evidence = self._format_evidence(self._search_serper(user_message))

        # 2) Системное сообщение с инструкцией по цитированию
        system_template = """
**Роль:** Вы — эксперт по переговорам и стратегическому управлению.  
Анализируйте ситуации, предлагайте компромиссы и прогнозируйте результат.

**Обязательные требования:**
• Используйте представленные ниже источники, цитируя их в тексте в формате [n].  
• Сошлитесь минимум на 2 разных источника.  
• В конце добавьте раздел **«Список источников»** с полными ссылками.

**Источники:**  
{evidence_block}

**Структура ответа:**  
### 1. Возможные компромиссные решения  
### 2. Прогноз эффективности  
### 3. Рекомендации
"""

        model = ChatOpenAI(
            api_key=self.valves.OPENAI_API_KEY,
            model=self.valves.MODEL_ID,
            temperature=self.valves.TEMPERATURE,
            streaming=True,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template("{user_input}"),
            ]
        )

        formatted = prompt.format_messages(
            user_input=user_message, evidence_block=evidence
        )

        # 3) Функция-стрим с логированием
        def generate_stream() -> Iterator[str]:
            for chunk in model.stream(formatted):
                content = getattr(chunk, "content", None)
                if content:
                    logging.debug(f"Model chunk: {content}")
                    yield content

        # 4) Запускаем с повторами
        return asyncio.run(self.make_request_with_retry(generate_stream))
