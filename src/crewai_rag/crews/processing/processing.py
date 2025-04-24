from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from ...tools.pdf_extracting import PDFTextExtractorTool
from ...tools.text_structurer import TextStructurerTool
from ...tools.chunking import SemanticChunkerTool
from ...tools.embedding import EmbedChunksTool
from ...tools.storing import StoreToPineconeTool



@CrewBase
class Processes:
    """Processing Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        super().__init__()
        self.inputs = {}
    
    @agent
    def Pdf_reader(self) -> Agent:
        return Agent(
            config=self.agents_config["Pdf_reader"],
            tools = [PDFTextExtractorTool()]
        )
    
    @agent
    def Text_preprocesser(self) -> Agent:
        return Agent(
            config=self.agents_config["Text_preprocesser"],
            tools = [TextStructurerTool()]
        )
    
    @agent
    def Text_Chunker(self) -> Agent:
        return Agent(
            config=self.agents_config["Text_Chunker"],
            tools = [SemanticChunkerTool()]
        )
    
    @agent
    def Text_Embedder(self) -> Agent:
        return Agent(
            config=self.agents_config["Text_Embedder"],
            tools = [EmbedChunksTool()]
        )
    
    @agent
    def Vector_Store(self) -> Agent:
        return Agent(
            config=self.agents_config["Vector_Store"],     
            tools = [StoreToPineconeTool()]
        )
    
    @task
    def Read_pdf(self) -> Task:
        return Task(
            config=self.tasks_config["Read_pdf"],
        )
    
    @task
    def Text_preprocessing(self) -> Task:
        return Task(
            config=self.tasks_config["Text_preprocessing"],
        )
    
    @task
    def Text_chunking(self) -> Task:
        return Task(
            config=self.tasks_config["Text_chunking"],
        )
    
    @task
    def Text_Embedding(self) -> Task:
        return Task(
            config=self.tasks_config["Text_Embedding"],
        )
    
    @task
    def Storing_Vectors(self) -> Task:
        return Task(
            config=self.tasks_config["Storing_Vectors"],
        )

    @crew
    def crew(self) -> Crew:
       """Creates the Processing Crew"""
      
       return Crew(
           agents=self.agents,
           tasks=self.tasks,
           process=Process.sequential,
        #    memory = True,
           verbose=True
       )
    def kickoff(self, inputs: dict):
        """Kickoff the crew with the given inputs"""
        if 'pdf' not in inputs or 'subject' not in inputs:
            raise ValueError("Missing required inputs: 'pdf' and 'subject' are required")
        self.inputs = inputs
        return self.crew().kickoff(inputs=inputs)

