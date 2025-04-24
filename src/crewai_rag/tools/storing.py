# # tools/store_to_pinecone_tool.py

# import os
# import json
# from pydantic import BaseModel
# from crewai_tools import BaseTool
# from pinecone import Pinecone, ServerlessSpec

# from ...utils.custom_memory import CustomMemory
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
# class StoreToPineconeInput(BaseModel):
#     dummy: str = "trigger"
#     subject: str = Field(..., description="The subject name of the pdf which is being store in the database")

# class StoreToPineconeTool(BaseTool):
#     name: str = "vector_storing"
#     description: str = "Stores embedded data to Pinecone from memory"
#     args_schema: type = StoreToPineconeInput

#     def _run(self, dummy: str , subject: str) -> str:
#         if not os.path.exists(MEMORY_PATH):
#             return "No memory found. Please run the embedder first."

#         with open(MEMORY_PATH, "r") as file:
#             memory_data = json.load(file)

#         embedded_data = memory_data.get("embedded_data")
#         if not embedded_data:
#             return "No embedded data found. Please run the embedder first."

#         # Initialize Pinecone

#         pc = Pinecone(api_key=pinecone_api_key)
#         index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
#         namespace = f"{subject}_notes"
#         vectors = []
#         batch_size = 100
#         for i, item in enumerate(embedded_data):
#             vectors.append((item["id"], item["vector"], item["metadata"]))

#             if (i + 1) % batch_size == 0 or i == len(embedded_data) - 1:
#                 index.upsert(vectors=vectors, namespace = namespace)
#                 vectors = []

#         # return f"Stored {len(embedded_data)} vectors in Pinecone."
        
#         deleted = memory.clear()
#         if deleted:
#             return f"✅ Stored {len(embedded_data)} vectors in Pinecone and cleared local memory for subject '{subject}'."
#         else:
#             return f"✅ Stored {len(embedded_data)} vectors in Pinecone, but failed to delete local memory."

# tools/store_to_pinecone_tool.py

import os
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from pinecone import Pinecone , ServerlessSpec
from crewai_rag.utils.custom_memory import CustomMemory

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

class StoreToPineconeInput(BaseModel):
    subject: str = Field(..., description="The subject name of the PDF to store in Pinecone.")

class StoreToPineconeTool(BaseTool):
    name: str = "vector_storing"
    description: str = "Stores embedded data to Pinecone from memory and deletes local memory"
    args_schema: Type[StoreToPineconeInput] = StoreToPineconeInput
    # def __init__(self):
    #     super().__init__()
    #     self.pc = Pinecone(api_key=pinecone_api_key)
    #     self.index = self.pc.Index(pinecone_index_name)
    
    def _run(self, subject: str) -> str:
        # Load subject-specific memory
        memory = CustomMemory(subject)
        embedded_data = memory.load("embedded_data")

        if not embedded_data:
            return f"❌ No embedded_data found for subject '{subject}'. Please run the embedder first."
        
        pc = Pinecone(api_key=pinecone_api_key)
        # Initialize Pinecone
        if not pinecone_api_key or not pinecone_index_name:
            return "❌ Pinecone API key or index name not set in environment."

        # pc = Pinecone(api_key=pinecone_api_key)
        
        if pinecone_index_name not in pc.list_indexes().names():
           pc.create_index(
           name=pinecone_index_name,
           dimension=1024,
           metric='cosine',
           spec=ServerlessSpec(
           cloud='aws',
           region='us-west-2'
        )
    )

# Always initialize the index here
        index = pc.Index(pinecone_index_name)
        namespace = f"{subject}_notes"

        batch_size = 100
        vectors = []

        for i, item in enumerate(embedded_data):
            vectors.append((item["id"], item["vector"], item["metadata"]))

            if (i + 1) % batch_size == 0 or i == len(embedded_data) - 1:
                index.upsert(vectors=vectors, namespace=namespace)
                vectors = []

        # Delete local memory file
        deleted = memory.clear()
        if deleted:
            return f"✅ Stored {len(embedded_data)} vectors in Pinecone and cleared local memory for subject '{subject}'."
        else:
            return f"✅ Stored {len(embedded_data)} vectors in Pinecone, but failed to delete local memory."

