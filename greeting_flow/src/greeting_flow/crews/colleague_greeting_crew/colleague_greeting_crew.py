from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import yaml

@CrewBase
class ColleagueGreetingCrew:
    """Crew for generating personalized greetings."""

    # Paths to the YAML configuration files
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def colleague_greeting_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['colleague_greeting_agent'],
            verbose=True
        )

    @task
    def generate_colleague_greeting(self) -> Task:
        return Task(
            config=self.tasks_config['generate_colleague_greeting'],
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
