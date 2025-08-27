from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchResults


class DuckDuckGoToolInput(BaseModel):
    query: str = Field(..., description="Search query for DuckDuckGo.")

class DuckDuckGoTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = "Searches DuckDuckGo and returns the top 3 results (title, link, snippet)."
    args_schema: Type[BaseModel] = DuckDuckGoToolInput

    def _run(self, query: str) -> str:
        ddg_tool = DuckDuckGoSearchResults(name="duckduckgo_search", num_results=3)
        results = ddg_tool.run(query)
        return str(results)
