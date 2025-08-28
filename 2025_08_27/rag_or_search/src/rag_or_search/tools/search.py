from typing import Type, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

class SearchToolInput(BaseModel):
	"""Input schema for SearchTool."""
	topic: str = Field(..., description="Topic to search for on DuckDuckGo.")

class SearchTool(BaseTool):
	name: str = "DuckDuckGo Search Tool"
	description: str = (
		"A tool to search DuckDuckGo for a topic and return the first three results. "
		"SSL certificate verification is disabled for corporate environments."
	)
	args_schema: Type[BaseModel] = SearchToolInput

	def search_ddg(self, topic: str, n: int = 3):
		with DDGS(verify=False) as ddgs:
			return list(ddgs.text(topic, region="en-us", safesearch="off", max_results=n))

	def _run(self, topic: str) -> List[dict]:

		if not topic:
			raise SystemExit("Choose a topic.")
		risultati = self.search_ddg(topic, 3)
		for i, r in enumerate(risultati, 1):
			titolo = r.get("title", "")
			url = r.get("href") or r.get("url") or ""
			snippet = r.get("body", "")
			return (f"{i}. {titolo}\n{url}\n{snippet}\n")
