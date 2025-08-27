from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class SumToolInput(BaseModel):
    """Input schema for SumTool."""

    value1: float = Field(..., description="First value to sum.")
    value2: float = Field(..., description="Second value to sum.")


class SumTool(BaseTool):
    name: str = "Sum Tool"
    description: str = (
        "A tool to calculate the sum of two numeric values."
    )
    args_schema: Type[BaseModel] = SumToolInput

    def _run(self, value1: float, value2: float) -> float:
        return value1 + value2
