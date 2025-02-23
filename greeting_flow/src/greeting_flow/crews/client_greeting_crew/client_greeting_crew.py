from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import yaml

@CrewBase
class ClientGreetingCrew:
    """Crew for generating personalized greetings."""

    # Paths to the YAML configuration files
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def client_greeting_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['client_greeting_agent'],
            verbose=True
        )

    @task
    def generate_client_greeting(self) -> Task:
        return Task(
            config=self.tasks_config['generate_client_greeting'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Personalized Greeting crew."""
        return Crew(
            agents=self.agents,  # Automatically collected by the @agent decorator
            tasks=self.tasks,    # Automatically collected by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
