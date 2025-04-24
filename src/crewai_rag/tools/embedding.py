# tools/embed_chunks_tool.py

import uuid
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from sentence_transformers import SentenceTransformer
from crewai_rag.utils.custom_memory import CustomMemory  # ✅ Custom memory

class EmbedChunksInput(BaseModel):
    subject: str = Field(..., description="The subject to embed and store data for.")

class EmbedChunksTool(BaseTool):
    name: str = "embed_text"
    description: str = "Embeds chunked data using BAAI/bge-large-en model and stores vectors and metadata in memory."
    args_schema: Type[EmbedChunksInput] = EmbedChunksInput

    def _run(self, subject: str) -> str:
        memory = CustomMemory(subject)

        chunked_data = memory.load("chunked_data")
        if not chunked_data:
            return f"❌ No chunked_data found for subject '{subject}'. Please run the chunker tool first."

        model = SentenceTransformer("BAAI/bge-large-en")
        embedded_data = []

        for chunk in chunked_data:
            embedding = model.encode(chunk["text"]).tolist()
            doc_id = str(uuid.uuid4())
            metadata = {
                "chapter": chunk.get("chapter", "Unknown Chapter"),
                "topic": chunk.get("topic", "Unknown Topic"),
                "text": chunk["text"]
            }
            embedded_data.append({
                "id": doc_id,
                "vector": embedding,
                "metadata": metadata
            })

        memory.save("embedded_data", embedded_data)
        return f"✅ Generated and stored embeddings for {len(embedded_data)} chunks under subject '{subject}'."
