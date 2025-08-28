from typing import Type, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from .rag_utils import rag_search

class RagToolInput(BaseModel):
	"""Input schema for RagTool."""
	question: str = Field(..., description="Question to answer using RAG search.")
	k: int = Field(3, description="Number of documents to retrieve for context.")
 
class RagTool(BaseTool):
	name: str = "RAG Search Tool"
	description: str = (
		"A tool that performs a Retrieval-Augmented Generation (RAG) search given a question and a number of documents to retrieve. "
		"Uses a local vector store and LLM to retrieve and answer based on context."
		"Returns a dictionary in the form { 'source': str, 'document': str }"
	)
	args_schema: Type[BaseModel] = RagToolInput

	def _run(self, question: str, k: int) -> List[str]:
		if not question:
			raise ValueError("Please provide a question for RAG search.")
		results = rag_search(question, k=k)
		
		return results
