#!/usr/bin/env python
import os
os.environ["CREWAI_TELEMETRY_DISABLED"] = "1"
from random import randint

from pydantic import BaseModel
from crewai import LLM

from crewai.flow import Flow, listen, start, router

from src.rag_or_search.crews.searchcrew.searchcrew import SearchCrew
from src.rag_or_search.crews.ragcrew.ragcrew import Ragcrew


class RAGSearchState(BaseModel):
    topic: str = "" 
    tool: str = ""  # "RAG" or "web"


class RAGSearchFlow(Flow[RAGSearchState]):

    @start()
    def get_user_topic(self):
        
        while True:
            self.state.topic = input("Enter the topic you want to search for: ")
            
            llm = LLM(model="azure/gpt-4o")
            
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
                "content": f"Is the following topic safe or unsafe? Topic: '{self.state.topic}'"
            }
            ]

            response = llm.call(messages=messages)
            
            if "unsafe" in response.lower():
                print("The topic is unsafe. Please enter a different topic.")
            else:
                break
            
        messages = [
            {
            "role": "system",
            "content": (
                "You are an AI assistant that classifies user requests. "
                "If the user's topic is related to RAG systems, output 'RAG'. "
                "If the topic is about anything else (e.g., web, general topics), output 'web'. "
                "Only respond with 'RAG' or 'web'."
            )
            },
            {
            "role": "user",
            "content": f"Classify the following topic: '{self.state.topic}'"
            }
        ]
        
        self.state.tool = llm.call(messages=messages)
        
        return self.state.topic

    @router(get_user_topic)
    def select_tool(self):
        
        if self.state.tool == "RAG":
            print("RAG selected to answer your query")
            return "RAG"
        elif self.state.tool == "web":
            print("Web search selected to answer your query")
            return "web"

    @listen("RAG")
    def query_RAG(self):
        print(f"Using RAG to search for topic: '{self.state.topic}'")

        result = Ragcrew().crew().kickoff(
            inputs={
                "topic": self.state.topic
            }
        )

    @listen("web")
    def query_web(self):
        result = SearchCrew().crew().kickoff(
            inputs={
                "topic": self.state.topic
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
