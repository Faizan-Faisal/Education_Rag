from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import fitz
from crewai_rag.utils.custom_memory import CustomMemory  # 👈 import here

class PDFExtractionInput(BaseModel):
    pdf_path: str = Field(..., description="Path to the PDF file.")
    subject: str = Field(..., description="The subject name for storing the memory.")

class PDFTextExtractorTool(BaseTool):
    name: str = "pdf_extractor"
    description: str = "Extracts raw text from a PDF and stores it in subject memory."
    args_schema: Type[BaseModel] = PDFExtractionInput

    def _run(self, pdf_path: str, subject: str) -> str:
        doc = fitz.open(pdf_path)
        raw_text = "\n".join(page.get_text("text") for page in doc).strip()

        memory = CustomMemory(subject)
        memory.save("raw_text", raw_text)  # 👈 Save using utility

        return f"✅ Raw text stored for subject: {subject}"
