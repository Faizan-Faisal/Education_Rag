# retriever_tool.py
from pydantic import BaseModel, Field
from typing import Type
from crewai.tools import BaseTool
from sentence_transformers import SentenceTransformer
import os
from pinecone import Pinecone, ServerlessSpec
from crewai_rag.utils.custom_memory import CustomMemory

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

class QueryInput(BaseModel):
    question: str = Field(..., description="The user's search question.")
    subject: str = Field(..., description="The subject name used as namespace in Pinecone.") 

# Load embedding model
embedding_model = SentenceTransformer(os.getenv("EMBEDDING_MODEL"))

class RetrieverTool(BaseTool):
    name: str = "context_retrieving"
    description: str = "Retrieves relevant chunks from Pinecone based on the user question for the given subject and stores them in memory"
    args_schema: Type[QueryInput] = QueryInput

    # def __init__(self):
    #     super().__init__()
    #     self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    #     self.index = self.pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    def _run(self, question: str, subject: str) -> str:

        formatted_refs, raw_chunks = self.retrieve_relevant_material(question, subject)

        # Save raw chunks (not formatted) to memory for the next tool (LLM)
        memory = CustomMemory(subject)
        memory.save("retrieved_context", raw_chunks)

        return f"Retrieved and saved {len(raw_chunks)} chunks for subject '{subject}'."

    def retrieve_relevant_material(self, question: str, subject: str):
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
        namespace = f"{subject}_notes"
        query_vector = embedding_model.encode(question).tolist()

        search_results = index.query(
            vector=query_vector,
            namespace=namespace,
            top_k=6,
            include_metadata=True
        )

        sorted_results = sorted(search_results["matches"], key=lambda x: x['score'], reverse=True)

        formatted_refs = []
        raw_chunks = []

        for match in sorted_results[:5]:
            metadata = match["metadata"]
            ref_text = metadata["text"]
            chapter = metadata.get("chapter", "Unknown Chapter")
            topic = metadata.get("topic", "Unknown Topic")
            page_number = metadata.get("page_number", "Unknown Page")

            formatted_ref = f"[{chapter} - {topic} (Page {page_number})]: {ref_text}"
            formatted_refs.append(formatted_ref)
            raw_chunks.append(ref_text)  # only the clean text for LLM

        return formatted_refs, raw_chunks
