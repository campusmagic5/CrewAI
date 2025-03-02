from crewai import Agent, Crew, Process, Task,LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import CSVSearchTool
from crewai_tools import PDFSearchTool
from crewai_tools import FileReadTool
from crewai_tools import FileWriterTool
from crewai_tools import TXTSearchTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

file_read_tool = FileReadTool(file_path="D:\\crewai_rag_tutorial\\rag_flow\\Latest_Football_News_March_1_2025.txt")
text_tool = TXTSearchTool(txt="D:\\crewai_rag_tutorial\\rag_flow\\Latest_Football_News_March_1_2025.txt",
            config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama3.1",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="ollama", # or openai, ollama, ...
            config=dict(
                model="nomic-embed-text",
            ),
        ),
    )
)
file_writer = FileWriterTool()
@CrewBase
class TextRagCrew:
    """Text RAG Crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would lik to add tools to your crew, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    @agent
    def file_read_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["file_read_agent"],
            tools=[file_read_tool],
            verbose=True,
            
        )

    @agent
    def text_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["text_agent"],
            tools=[text_tool],
            verbose=True,
            
        )
    
    @agent
    def file_write_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["file_write_agent"],
            tools=[file_writer],
            verbose=True,
            
        )
    
    

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task

    @task
    def read_file_task(self) -> Task:
        return Task(
            config=self.tasks_config["read_file_task"],
            tools=[file_read_tool],
        )

    @task
    def text_rag_task(self) -> Task:
        return Task(
            config=self.tasks_config["text_rag_task"],
            tools=[text_tool],
        )
    
    @task
    def write_file_task(self) -> Task:
        return Task(
            config=self.tasks_config["write_file_task"],
            tools=[file_writer],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Text RAG Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )