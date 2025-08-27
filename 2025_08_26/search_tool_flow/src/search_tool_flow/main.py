#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start
from crewai import LLM

from search_tool_flow.crews.paraphrase_crew.paraphrase_crew import ParaphraseCrew

class FlowState(BaseModel):
    topic: str = ""
    summary: str = ""
    

class Flow(Flow[FlowState]):

    @start()
    def get_user_input(self):
        
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
        
        return self.state.topic

    @listen(get_user_input)
    def generate_summary(self):
        print("Generating summary...")
        result = (
            ParaphraseCrew()
            .crew()
            .kickoff(inputs={"topic": self.state.topic})
        )

        print("Summary generated", result.raw)
        self.state.summary = result.raw
        
        return result.raw

    @listen(generate_summary)
    def save_summary(self):
        print("Saving summary")
        with open("summary.txt", "w") as f:
            f.write(self.state.summary)


def kickoff():
    summary_flow = Flow()
    summary_flow.kickoff()


if __name__ == "__main__":
    kickoff()
