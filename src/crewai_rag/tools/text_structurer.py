# src/crewai_rag/tools/text_structurer.py

import re
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from typing import Type
from crewai_rag.utils.custom_memory import CustomMemory  # 👈 import your memory utility

class TextStructuringInput(BaseModel):
    subject: str = Field(..., description="The subject name to retrieve and store structured text.")

class TextStructurerTool(BaseTool):
    name: str = "preprocess_text"
    description: str = "Structures raw PDF text by chapters and topics and stores it under the same subject."
    args_schema: Type[TextStructuringInput] = TextStructuringInput

    def _run(self, subject: str) -> str:
        memory = CustomMemory(subject)

        raw_text = memory.load("raw_text")
        if not raw_text:
            return f"❌ No raw_text found in memory for subject '{subject}'. Please run the extractor first."

        structured = self.structure_text(raw_text)
        memory.save("structured_text", structured)

        return f"✅ Structured text saved for subject: {subject}"

    def structure_text(self, raw_text: str):
        lines = raw_text.split("\n")
        structured_text = []
        current_chapter = "Unknown Chapter"
        current_topic = "Uknown Topic"
        current_text = ""

        for line in lines:
            line = line.strip()

            if re.match(r"^(Chapter\s\d+[:.])", line, re.IGNORECASE):
                if current_text:
                    structured_text.append({
                        "chapter": current_chapter,
                        "topic": current_topic,
                        "text": current_text.strip()
                    })
                    current_text = ""
                current_chapter = line
                current_topic = "Unknown Topic"
            elif re.match(r"^\d+\.\d+\s+[A-Za-z]", line):
                if current_text:
                    structured_text.append({
                        "chapter": current_chapter,
                        "topic": current_topic,
                        "text": current_text.strip()
                    })
                    current_text = ""
                current_topic = line

            else:
                current_text += " " + line

        if current_text:
            structured_text.append({
                "chapter": current_chapter or "Unknown Chapter",
                "topic": current_topic or "Unknown Topic",
                "text": current_text.strip()
            })

        return structured_text
