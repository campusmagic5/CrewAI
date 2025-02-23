from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource
@CrewBase
class LeadsCrew:
    """Crew for retrieving leads information."""

    # Create a JSON knowledge source
    leads_json_source = JSONKnowledgeSource(
        file_paths=["Leads.json"]
    )

    # Paths to the YAML configuration files
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def lead_information_retriever_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['lead_information_retriever_agent'],
            verbose=True
        )

    @task
    def leads_information(self) -> Task:
        return Task(
            config=self.tasks_config['leads_information'],
        )

    @crew
    def crew(self) -> Crew:
        """Provides Employee details."""
        return Crew(
            agents=self.agents,  # Automatically collected by the @agent decorator
            tasks=self.tasks,    # Automatically collected by the @task decorator
            process=Process.sequential,
            verbose=True,
            knowledge_sources=[self.leads_json_source],
            embedder = {
                    "provider": "ollama",
                    "config": {
                    "model": "mxbai-embed-large",
                    "base_url": "http://localhost:11434"
    }
}

        )