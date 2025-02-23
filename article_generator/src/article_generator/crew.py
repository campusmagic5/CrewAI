from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class ArticleGenerator:
    @agent
    def content_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['content_writer'],
            verbose=True,
            # Add necessary tools for content generation
        )

    @agent
    def summarization_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config['summarization_specialist'],
            verbose=True,
            # Add necessary tools for summarization
        )

    @task
    def write_article_task(self) -> Task:
        return Task(
            config=self.tasks_config['write_article_task'],
            # Define task-specific parameters
        )

    @task
    def summarize_article_task(self) -> Task:
        return Task(
            config=self.tasks_config['summarize_article_task'],
            # Define task-specific parameters
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
