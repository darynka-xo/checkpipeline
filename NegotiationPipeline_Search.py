import logging
import sys
import os
import asyncio
from typing import List, Iterator, Callable, Any
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
import aiohttp
import json
from aiohttp import ClientSession, ClientTimeout
from fastapi import HTTPException

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

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

    async def make_request_with_retry(self, fn: Callable[[], Iterator[dict]], retries=3) -> Iterator[dict]:
        for attempt in range(retries):
            try:
                return fn()
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt + 1 == retries:
                    raise HTTPException(status_code=500, detail=f"Pipeline failed after {retries} attempts: {str(e)}")
                await asyncio.sleep(2 ** attempt)

    async def search_trusted_sources(self, query: str, session: ClientSession) -> List[dict]:
        """Perform a web search using Serper API, limited to trusted sites."""
        search_query = f"{query} {' '.join(self.trusted_sites)}"
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.valves.SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "q": search_query,
            "num": 10
        }

        timeout = ClientTimeout(total=10)  # 10-second timeout
        retries = 3
        for attempt in range(retries):
            try:
                async with session.post(url, headers=headers, json=payload, timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("organic", [])
                        formatted_results = [
                            {
                                "web_id": f"web:{idx}",
                                "title": result.get("title", ""),
                                "snippet": result.get("snippet", ""),
                                "link": result.get("link", "")
                            }
                            for idx, result in enumerate(results)
                        ]
                        logging.debug(f"Search results: {formatted_results}")
                        return formatted_results
                    else:
                        logging.error(f"Serper API error: {response.status}")
                        return []
            except Exception as e:
                logging.error(f"Search attempt {attempt + 1} failed: {e}")
                if attempt + 1 == retries:
                    logging.error("Max retries reached for search")
                    return []
                await asyncio.sleep(2 ** attempt)

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Iterator[dict]:
        system_message = """
**Role:** You are an expert in negotiation and strategic management, specializing in analyzing and predicting the success of negotiation models, communication channels, and building compromise strategies for conflict resolution. Your expertise is strictly limited to negotiation analytics, avoiding political, domestic, philosophical, or technical topics.

**Scope of Responsibility:**
You only respond to questions related to negotiation situations, conflicts of interest, persuasion strategies, argument construction, or compromise evaluations. For irrelevant queries, gently redirect the user to formulate a negotiation-related situation or problem.

**Example Response to Irrelevant Query:**
> "My expertise is limited to negotiation analysis, compromise strategies, and communication model success. Can I assist with a specific negotiation situation or conflict?"

**Objective:**
1. Analyze provided conflict or complex negotiation situations.
2. Propose effective, balanced, and compromise-based solutions.
3. Predict the success of each negotiation model based on sustainability, benefits, and long-term impact.

**Response Structure:**

### 1. Possible Compromise Solutions
- Specific, actionable proposals considering both parties' interests.

### 2. Effectiveness Forecast
- Evaluate each proposed model based on sustainability, scalability, risks, and implementation potential.

### 3. Recommendations
- Identify the best model for the situation and explain why.
- Include clarifying questions if needed.

**Web Search Integration:**
- Use provided search results from trusted sources to strengthen arguments and provide evidence.
- Cite sources using the format: [web:<number>]
- Only include relevant information from search results that supports the negotiation analysis.
- If no relevant search results are available, rely on your expertise without citing.
"""

        async def generate_response() -> Iterator[dict]:
            async with ClientSession() as session:
                try:
                    # Perform search
                    search_results = await self.search_trusted_sources(user_message, session)
                    search_context = "\n".join(
                        [f"Web ID: {res['web_id']}\nTitle: {res['title']}\nSnippet: {res['snippet']}\nLink: {res['link']}\n"
                         for res in search_results]
                    ) if search_results else "No relevant search results found."

                    model = ChatOpenAI(
                        api_key=self.valves.OPENAI_API_KEY,
                        model=self.valves.MODEL_ID,
                        temperature=self.valves.TEMPERATURE,
                        streaming=True
                    )

                    prompt = ChatPromptTemplate.from_messages([
                        SystemMessagePromptTemplate.from_template(system_message),
                        HumanMessagePromptTemplate.from_template(
                            "User Input: {user_input}\n\nSearch Results:\n{search_context}"
                        )
                    ])

                    formatted_messages = prompt.format_messages(
                        user_input=user_message,
                        search_context=search_context
                    )

                    # Stream response in OpenAI-compatible format
                    async for chunk in model.astream(formatted_messages):
                        content = getattr(chunk, "content", None)
                        if content:
                            logging.debug(f"Model chunk: {content}")
                            yield {
                                "choices": [
                                    {
                                        "delta": {"content": content},
                                        "index": 0,
                                        "finish_reason": None
                                    }
                                ]
                            }
                    # Signal completion
                    yield {
                        "choices": [
                            {
                                "delta": {},
                                "index": 0,
                                "finish_reason": "stop"
                            }
                        ]
                    }
                except Exception as e:
                    logging.error(f"Error in generate_response: {e}")
                    raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

        return asyncio.run(self.make_request_with_retry(generate_response))
