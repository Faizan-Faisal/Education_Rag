# from random import randint

# from pydantic import BaseModel

# from crewai.flow import Flow, listen, start
# from crewai_rag.crews.processing.processing import Processes
# from crewai_rag.crews.Retrieval.retrieval import RAG
# class Processing_Crew(Flow): 
#     processing_inputs = {
#         'pdf': 'path/to/document.pdf',
#         'subject': 'AI'
#     }
#     @start()
#     processing_result = Processes().crew().kickoff(inputs=processing_inputs)
    
#     # Answer a question
#     retrieval_inputs = {
#         'question': 'What is machine learning?',
#         'subject': 'AI'
#     }
#     retrieval_result = RAG().crew().kickoff(inputs=retrieval_inputs)
    




# def kickoff():
#     generate = Processing_Crew()
#     result = generate.kickoff()
#     print(result)

#!/usr/bin/env python
import os
from dotenv import load_dotenv
from crewai_rag.crews.processing.processing import Processes
from crewai_rag.crews.retrieval.retrieval import RAG

# def process_course_material(pdf: str, subject: str):
#     """
#     Instructor workflow: Process and store course material.
    
#     Args:
#         pdf: Path to the course material PDF
#         subject: Subject name (used as namespace in vector store)
#     """
#     try:
#         crew = Processes()
#         result = crew.crew().kickoff(
#             inputs={
#                 "pdf": pdf,
#                 "subject": subject
#             }
#         )
#         print(f"\n=== Course Material Processing Complete ===\n")
#         print(f"Successfully processed {pdf_path} for subject {subject}")
#         return result
#     except Exception as e:
#         print(f"Error processing course material: {str(e)}")
#         raise

def process_course_material(pdf: str, subject: str):
    """
    Instructor workflow: Process and store course material.
    
    Args:
        pdf: Path to the course material PDF
        subject: Subject name (used as namespace in vector store)
    """
    try:
        crew = Processes()
        result = crew.kickoff(
            inputs={
                "pdf": pdf,
                "subject": subject
            }
        )
        print(f"\n=== Course Material Processing Complete ===\n")
        print(f"Successfully processed {pdf} for subject {subject}")
        return result
    except Exception as e:
        print(f"Error processing course material: {str(e)}")
        raise

def answer_student_question(question: str, subject: str):
    """
    Student workflow: Answer questions about course material.
    
    Args:
        question: Student's question
        subject: Subject to search in (matches instructor's uploaded content)
    """
    try:
        crew = RAG()
        result = crew.crew().kickoff(
            inputs={
                "question": question,
                "subject": subject
            }
        )
        print("\n=== Answer ===\n")
        print(result.raw)
        return result
    except Exception as e:
        print(f"Error answering question: {str(e)}")
        raise

def store():
    load_dotenv()
    
    # Example: Instructor uploading course material
    process_course_material(
        pdf="src//crewai_rag/resources/Software_Engineering.pdf",
        subject="software engineering"
    )
def retrieve():
    load_dotenv()
    # Example: Student asking a question
    answer_student_question(
        question="What is waterfall model?",
        subject="software engineering"
    )