from pydantic import BaseModel
from crewai.flow import Flow, listen, start , router , or_
from rag_flow.crews.csv_rag_crew.csv_rag_crew import CsvRagCrew
from rag_flow.crews.pdf_rag_crew.pdf_rag_crew import PdfRagCrew
from rag_flow.crews.json_rag_crew.json_rag_crew import JsonRagCrew
from rag_flow.crews.text_rag_crew.text_rag_crew import TextRagCrew
from rag_flow.crews.website_rag_crew.website_rag_crew import WebsiteRagCrew
class RagState(BaseModel):
    question: str = ""
    file_type: str = ""
    answer: str = ""

class RagFlow(Flow[RagState]):

    @start()
    def get_query(self):
        # In a real application, these details might come from user input or a database
        #self.state.question = "What challenges did England face in their Champions Trophy match against South Africa on March 1, 2025?"
        self.state.question = "What is the average salary of employees in the HR department?"
        self.state.file_type = "csv"
        print(f"question: {self.state.question} , file_type: {self.state.file_type}")
        return "query_received"

    @router(get_query)
    def rag_based_on_file_type(self):
        """Route based on file type."""
        if self.state.file_type == "csv":
            return "csv"
        elif self.state.file_type == "pdf":
            return "pdf"
        elif self.state.file_type == "json":
            return "json"
        elif self.state.file_type == "txt":
            return "txt"
        elif self.state.file_type == "website":
            return "website"
        else:
            return "NA"

    @listen("csv")
    def csv_information(self):
        """Retrieve the infomation from csv"""
        crew = CsvRagCrew().crew()
        inputs = {
            "question": self.state.question
        }
        result = crew.kickoff(inputs=inputs)
        self.state.answer = result.raw
        print(f"Answer: {self.state.answer}")
        return "information_retrieved"

    @listen("pdf")
    def pdf_information(self):
        """Retrieve the infomation from pdf"""
        crew = PdfRagCrew().crew()
        inputs = {
            "question": self.state.question
        }
        result = crew.kickoff(inputs=inputs)
        self.state.answer = result.raw
        print(f"Answer: {self.state.answer}")
        return "information_retrieved"
    
    @listen("json")
    def json_information(self):
        """Retrieve the infomation from json"""
        crew = JsonRagCrew().crew()
        inputs = {
            "question": self.state.question
        }
        result = crew.kickoff(inputs=inputs)
        self.state.answer = result.raw
        print(f"Answer: {self.state.answer}")
        return "information_retrieved"
    
    @listen("txt")
    def text_information(self):
        """Retrieve the infomation from txt"""
        crew = TextRagCrew().crew()
        inputs = {
            "question": self.state.question
        }
        result = crew.kickoff(inputs=inputs)
        self.state.answer = result.raw
        print(f"Answer: {self.state.answer}")
        return "information_retrieved"
    
    @listen("website")
    def website_information(self):
        """Retrieve the infomation from website"""
        crew = WebsiteRagCrew().crew()
        inputs = {
            "question": self.state.question
        }
        result = crew.kickoff(inputs=inputs)
        self.state.answer = result.raw
        print(f"Answer: {self.state.answer}")
        return "information_retrieved"

def kickoff():
    rag_flow = RagFlow()
    rag_flow.kickoff()

def plot():
    rag_flow = RagFlow()
    rag_flow.plot()

if __name__ == "__main__":
    kickoff()