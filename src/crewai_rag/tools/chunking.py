# src/crewai_rag/tools/semantic_chunker.py

from typing import List, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from crewai_rag.utils.custom_memory import CustomMemory  # 👈 import your memory utility

class SemanticChunkingInput(BaseModel):
    subject: str = Field(..., description="The subject name to retrieve and store chunked data.")

class SemanticChunkerTool(BaseTool):
    name: str = "chunking_text"
    description: str = "Chunks structured text semantically for better context retention and saves it under the subject."
    args_schema: Type[SemanticChunkingInput] = SemanticChunkingInput

    def _run(self, subject: str) -> str:
        memory = CustomMemory(subject)

        structured_data = memory.load("structured_text")
        if not structured_data:
            return f"❌ No structured_text found for subject '{subject}'. Please run the structurer tool first."

        chunked_data = self.chunk_text_semantically(structured_data)
        memory.save("chunked_data", chunked_data)

        return f"✅ Semantic chunking complete for subject: {subject} ({len(chunked_data)} chunks)"

    def chunk_text_semantically(self, structured_data: List[dict]) -> List[dict]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len
        )

        chunks = []
        for sec in structured_data:
            if sec["text"]:
                split_chunks = text_splitter.split_text(sec["text"])
                for chunk in split_chunks:
                    chunks.append({
                        "chapter": sec.get("chapter"),
                        "topic": sec.get("topic"),
                        "text": chunk
                    })

        return chunks
