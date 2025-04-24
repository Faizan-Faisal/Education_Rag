# generator_tool.py
from pydantic import BaseModel , Field
from typing import Type
from crewai.tools import BaseTool
from crewai_rag.utils.custom_memory import CustomMemory  # ✅ Correct import
import os

# Input schema
class AnswerInput(BaseModel):
    query: str = Field(..., description="The user's original query to generate an answer for.")
    subject: str = Field(..., description="The subject name to fetch retrieved context from memory.")  # ✅ Added

class GeneratorTool(BaseTool):
    name: str = "answering"
    description: str = "Generates an answer using retrieved context from memory and a language model"
    args_schema: Type[AnswerInput] = AnswerInput


    def _run(self, query: str, subject: str) -> str:
        memory = CustomMemory(subject)
        context = memory.load("retrieved_context")

        if not context:
            return "No context was found in memory to generate an answer."

        prompt = f"""
You are an AI tutor assisting with concepts from the subject '{subject}'. Answer the query based on the given context.

Query: {query}

Context (retrieved sources):
{context}

If the context does not contain enough information, say: "I don't have sufficient information in my sources but the detail I know about the topic is: ...".

Provide a detailed yet concise answer. Add a heading for the definition and explain it properly if it's in the book.
"""

        return prompt  # This works if you're using CrewAI's built-in LLM manager
