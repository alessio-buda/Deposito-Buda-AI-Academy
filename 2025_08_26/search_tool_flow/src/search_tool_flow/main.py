#!/usr/bin/env python
from random import randint

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from search_tool_flow.crews.paraphrase_crew.paraphrase_crew import ParaphraseCrew

class FlowState(BaseModel):
    topic: str = ""
    summary: str = ""
    

class Flow(Flow[FlowState]):

    @start()
    def get_user_input(self):
        
        self.state.topic = input("Enter a topic you want to know more about: ")
        
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
