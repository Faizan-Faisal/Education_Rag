from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from ...tools.retrieving import RetrieverTool
from ...tools.generatingAns import GeneratorTool
from ...tools.validating import ValidatorTool

# from crewai.memory.file_memory import FileMemory
# from joblib import Memory 

@CrewBase
class RAG:
    """Retrieval CREW"""


    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    #  memory = FileMemory("memory")
    def __init__(self):
        super().__init__()
        self.inputs = {}

    @agent
    def context_retrieval(self) -> Agent:
        return Agent(
            config=self.agents_config["context_retrieval"],
            tools = [RetrieverTool()] 
        )
    
    @agent
    def answer(self) -> Agent:
        return Agent(
            config=self.agents_config["answer"],
            tools = [GeneratorTool()] 
        )
    
    @agent
    def validater(self) -> Agent:
        return Agent(
            config=self.agents_config["validater"],
            tools = [ValidatorTool()] 
        )

    @task
    def retrieving_context(self) -> Task:
        return Task(
            config=self.tasks_config["retrieving_context"],
        )

    @task
    def generating_answer(self) -> Task:
        return Task(
            config=self.tasks_config["generating_answer"],
        )
    
    @task
    def validating_answer(self) -> Task:
        return Task(
            config=self.tasks_config["validating_answer"],
        )
    
    @crew
    def crew(self) -> Crew:
        """Creates the Retrieval Augmented Generation Crew"""
        # if 'question' not in self.inputs or 'subject' not in self.inputs:
        #    raise ValueError("Missing required inputs: question' and 'subject' are required")

        return Crew(
            agents=self.agents,  
            tasks=self.tasks,  
            process=Process.sequential,
            # memory = self.memory,
            verbose=True,
        )
        def kickoff(self, inputs: dict):
            """Kickoff the crew with the given inputs"""
            if 'question' not in inputs or 'subject' not in inputs:
                raise ValueError("Missing required inputs: 'question' and 'subject' are required")
            self.inputs = inputs
            return self.crew().kickoff(inputs=inputs)
