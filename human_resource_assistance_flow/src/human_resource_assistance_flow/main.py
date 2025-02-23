from pydantic import BaseModel
from crewai.flow import Flow, listen, start , router , or_
from human_resource_assistance_flow.crews.developers_crew.developers_crew import DevelopersCrew
from human_resource_assistance_flow.crews.leads_crew.leads_crew import LeadsCrew
class EmployeeState(BaseModel):
    employeeCode: str = ""
    job_role: str = ""
    employee_details: str = ""

class HumanResourceAssistanceFlow(Flow[EmployeeState]):

    @start()
    def get_employee_details(self):
        # In a real application, these details might come from user input or a database
        self.state.employeeCode = "C5"
        self.state.job_role = "Developer"
        print(f"employeeCode: {self.state.employeeCode}, job_role: {self.state.job_role}")
        return "details_collected"

    @router(get_employee_details)
    def route_based_on_relationship(self):
        """Route based on job role."""
        if self.state.job_role == "Lead":
            return "Lead"
        elif self.state.job_role == "Developer":
            return "Developer"
        else:
            return "NA"

    @listen("Lead")
    def leads_information(self):
        """Retrieve the infomation about the lead"""
        crew = LeadsCrew().crew()
        inputs = {
            "employeeCode": self.state.employeeCode
        }
        result = crew.kickoff(inputs=inputs)
        self.state.employee_details = result.raw
        print(f"Employee Information: {self.state.employee_details}")
        return "details_retrieved"

    @listen("Developer")
    def developers_information(self):
        """Retrieve the infomation about the developer"""
        crew = DevelopersCrew().crew()
        inputs = {
            "employeeCode": self.state.employeeCode,
        }
        result = crew.kickoff(inputs=inputs)
        self.state.employee_details = result.raw
        print(f"Employee Information: {self.state.employee_details}")
        return "details_retrieved"

    

    @listen(or_(leads_information,developers_information))
    def save_details(self):
        with open("employee_details.txt", "w") as file:
            file.write(self.state.employee_details)
        print("Employee details saved to employee_details.txt")

def kickoff():
    employee_flow = HumanResourceAssistanceFlow()
    employee_flow.kickoff()


def plot():
    employee_flow = HumanResourceAssistanceFlow()
    employee_flow.plot()


if __name__ == "__main__":
    kickoff()