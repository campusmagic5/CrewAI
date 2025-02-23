from pydantic import BaseModel
from crewai.flow import Flow, listen, start , router , or_
from greeting_flow.crews.client_greeting_crew.client_greeting_crew import ClientGreetingCrew
from greeting_flow.crews.colleague_greeting_crew.colleague_greeting_crew import ColleagueGreetingCrew
from greeting_flow.crews.family_greeting_crew.family_greeting_crew import FamilyGreetingCrew
from greeting_flow.crews.friend_greeting_crew.friend_greeting_crew import FriendGreetingCrew
from greeting_flow.crews.general_greeting_crew.general_greeting_crew import GeneralGreetingCrew

class GreetingState(BaseModel):
    recipient_name: str = ""
    occasion: str = ""
    relationship_type: str = ""
    greeting_message: str = ""

class GreetingFlow(Flow[GreetingState]):

    @start()
    def get_recipient_details(self):
        # In a real application, these details might come from user input or a database
        self.state.recipient_name = "John Doe"
        self.state.occasion = "Birthday"
        self.state.relationship_type = "Family"
        print(f"Recipient: {self.state.recipient_name}, Occasion: {self.state.occasion}")
        return "details_collected"

    @router(get_recipient_details)
    def route_based_on_relationship(self):
        """Route based on relationship type to personalize greetings."""
        if self.state.relationship_type == "Family":
            return "family_greeting"
        elif self.state.relationship_type == "Friend":
            return "friend_greeting"
        elif self.state.relationship_type == "Colleague":
            return "colleague_greeting"
        elif self.state.relationship_type == "Client":
            return "client_greeting"
        else:
            return "general_greeting"

    @listen("family_greeting")
    def generate_family_greeting(self):
        """Generate a warm and affectionate greeting for family members."""
        crew = FamilyGreetingCrew().crew()
        inputs = {
            "recipient_name": self.state.recipient_name,
            "occasion": self.state.occasion,
            "additional_info": "Sending lots of love and happiness your way!"
        }
        result = crew.kickoff(inputs=inputs)
        self.state.greeting_message = result.raw
        print(f"Family Greeting: {self.state.greeting_message}")
        return "greeting_generated"

    @listen("friend_greeting")
    def generate_friend_greeting(self):
        """Generate a cheerful and casual greeting for friends."""
        crew = FriendGreetingCrew().crew()
        inputs = {
            "recipient_name": self.state.recipient_name,
            "occasion": self.state.occasion,
            "additional_info": "Hope you have an amazing day filled with fun and laughter!"
        }
        result = crew.kickoff(inputs=inputs)
        self.state.greeting_message = result.raw
        print(f"Friend Greeting: {self.state.greeting_message}")
        return "greeting_generated"

    @listen("colleague_greeting")
    def generate_colleague_greeting(self):
        """Generate a professional and respectful greeting for colleagues."""
        crew = ColleagueGreetingCrew().crew()
        inputs = {
            "recipient_name": self.state.recipient_name,
            "occasion": self.state.occasion,
            "additional_info": "Wishing you continued success in your professional journey!"
        }
        result = crew.kickoff(inputs=inputs)
        self.state.greeting_message = result.raw
        print(f"Colleague Greeting: {self.state.greeting_message}")
        return "greeting_generated"

    @listen("client_greeting")
    def generate_client_greeting(self):
        """Generate a formal and appreciative greeting for clients."""
        crew = ClientGreetingCrew().crew()
        inputs = {
            "recipient_name": self.state.recipient_name,
            "occasion": self.state.occasion,
            "additional_info": "We appreciate your trust in us and look forward to continued collaboration!"
        }
        result = crew.kickoff(inputs=inputs)
        self.state.greeting_message = result.raw
        print(f"Client Greeting: {self.state.greeting_message}")
        return "greeting_generated"

    @listen("general_greeting")
    def generate_general_greeting(self):
        """Generate a standard greeting if no specific relationship is identified."""
        crew = GeneralGreetingCrew().crew()
        inputs = {
            "recipient_name": self.state.recipient_name,
            "occasion": self.state.occasion,
            "additional_info": "Best wishes on this special occasion!"
        }
        result = crew.kickoff(inputs=inputs)
        self.state.greeting_message = result.raw
        print(f"General Greeting: {self.state.greeting_message}")
        return "greeting_generated"

    @listen(or_(generate_family_greeting,generate_friend_greeting,generate_colleague_greeting,generate_client_greeting,generate_general_greeting))
    def save_greeting(self):
        with open("greeting.txt", "w") as file:
            file.write(self.state.greeting_message)
        print("Greeting message saved to greeting.txt")

def kickoff():
    greeting_flow = GreetingFlow()
    greeting_flow.kickoff()


def plot():
    greeting_flow= GreetingFlow()
    greeting_flow.plot()


if __name__ == "__main__":
    kickoff()
