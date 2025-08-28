#!/usr/bin/env python
import os
os.environ["CREWAI_TELEMETRY_DISABLED"] = "1"
from random import randint

from pydantic import BaseModel
from crewai import LLM

from crewai.flow import Flow, listen, start, router

from src.rag_or_search.crews.searchcrew.searchcrew import SearchCrew
from src.rag_or_search.crews.ragcrew.ragcrew import Ragcrew
from src.rag_or_search.crews.mathcrew.mathcrew import Mathcrew


class RAGSearchState(BaseModel):
    request: str = "" 
    tool: str = ""  # "RAG" or "web"


class RAGSearchFlow(Flow[RAGSearchState]):

    @start()
    def get_user_request(self):
        
        llm = LLM(model="azure/gpt-4o")
        
        while True:
            self.state.request = input("Enter your request: ")
            
            messages = [
            {
                "role": "system",
                "content": (
                "You are an AI assistant that evaluates topics for safety and ethics. "
                "Given a topic, determine if it is dangerous, unethical, or otherwise inappropriate. "
                "Respond with 'safe' if the topic is appropriate, or 'unsafe' if it is dangerous or unethical."
                )
            },
            {
                "role": "user",
                "content": f"Is the following topic safe or unsafe? Topic: '{self.state.request}'"
            }
            ]

            response = llm.call(messages=messages)
            
            if "unsafe" in response.lower():
                print("The topic is unsafe. Please enter a different topic.")
            else:
                break
            
        print("***** USER REQUEST *****")
        print(f"Request: {self.state.request}")
            
        messages = [
            {
            "role": "system",
            "content": (
                "You are an AI assistant that classifies user requests according to the following rules: "
                "1) If the user's request is related to RAG systems, output 'RAG'. "
                "2) If the user request is to compute a mathematical formula (e.g. the area of a circle, the square root of a value), output 'math'."
                "3) If the user request is about anything else (e.g., web, general topics), output 'web'."
                "Only respond with 'RAG', 'math', or 'web'."
            )
            },
            {
            "role": "user",
            "content": f"Classify the following topic: '{self.state.request}'"
            }
        ]
        
        self.state.tool = llm.call(messages=messages)
        
        return self.state.request

    @router(get_user_request)
    def select_tool(self):
        
        if self.state.tool == "RAG":
            print("RAG selected to answer your query")
            return "RAG"
        elif self.state.tool == "web":
            print("Web search selected to answer your query")
            return "web"
        elif self.state.tool == "math":
            print("Math selected to answer your query")
            return "math"

    @listen("RAG")
    def query_RAG(self):
        print(f"Using RAG to search for topic: '{self.state.request}'")

        result = Ragcrew().crew().kickoff(
            inputs={
                "request": self.state.request
            }
        )

    @listen("web")
    def query_web(self):
        result = SearchCrew().crew().kickoff(
            inputs={
                "request": self.state.request
            }
        )
        
    @listen("math")
    def query_math(self):

        result = Mathcrew().crew().kickoff(
            inputs={
                "question": self.state.request
            }
        )
            
def kickoff():
    poem_flow = RAGSearchFlow()
    poem_flow.kickoff()


def plot():
    poem_flow = RAGSearchFlow()
    poem_flow.plot()


if __name__ == "__main__":
    kickoff()
