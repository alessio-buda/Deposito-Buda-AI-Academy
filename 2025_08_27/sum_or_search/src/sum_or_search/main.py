#!/usr/bin/env python
from random import randint

from pydantic import BaseModel
from crewai import LLM

from crewai.flow import Flow, listen, start, router

from sum_or_search.crews.sumcrew.sum_crew import SumCrew
from sum_or_search.crews.searchcrew.searchcrew import SearchCrew


class SumSearchState(BaseModel):
    option: str = ""
    sum1: int = 0
    sum2: int = 0
    result: int = 0
    topic: str = ""


class SumSearchFlow(Flow[SumSearchState]):

    @start()
    def get_user_choice(self):
        
        options = ["sum", "search"]
        
        print("Select one of the available options")
        print("Options:")
        for idx, opt in enumerate(options, start=1):
            print(f"{idx} - {opt}")

        valid_choice = False
        while not valid_choice:
            choice = int(input("Enter the number of your choice: "))
            if choice == options.index("sum") + 1:
                valid_choice = True
                self.state.option = "sum"
            elif choice == options.index("search") + 1:
                valid_choice = True
                self.state.option = "search"
            else:
                print("Invalid choice. Please try again.")

    @router(get_user_choice)
    def select_option(self):
        
        if self.state.option == "sum":
            return "sum"
        elif self.state.option == "search":
            return "search"

    @listen("sum")
    def get_values(self):
        print("You selected the sum option.")
        while True:
            try:
                self.state.sum1 = int(input("Enter the first number to sum: "))
                break
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
        while True:
            try:
                self.state.sum2 = int(input("Enter the second number to sum: "))
                break
            except ValueError:
                print("Invalid input. Please enter a valid integer.")
        
        return self.state.sum1, self.state.sum2
                
    @listen(get_values)
    def calculate_sum(self):
        result = SumCrew().crew().kickoff(
            inputs={
                "value1": self.state.sum1,
                "value2": self.state.sum2
            }
        )
        
        print(result)

    @listen("search")
    def get_topic(self):
        print("You selected the search option.")
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
            
        print(f"Searching for topic: {self.state.topic}")
        return self.state.topic
        
    @listen(get_topic)
    def perform_search(self):
        result = SearchCrew().crew().kickoff(
            inputs={
                "topic": self.state.topic
            }
        )
        
        print(result)
            
def kickoff():
    poem_flow = SumSearchFlow()
    poem_flow.kickoff()


def plot():
    poem_flow = SumSearchFlow()
    poem_flow.plot()


if __name__ == "__main__":
    kickoff()
