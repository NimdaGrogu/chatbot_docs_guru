from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import Pinecone
from langchain_core.prompts import PromptTemplate
import logging


def prompt_template(question: str) -> str:
    prompt = PromptTemplate.from_template(f"""
        You are a helpful assistant, Perform the following task:
                 
        1- Answer the question: {question}
                    
    """)

    return prompt.format(question=question)


def aumented_prompt_template(query: str, source_knowledge: str) -> str:
    logging.info(f"Augmented prompt, searching for: {query} in vectorstore")

    # feed into an augmented prompt
    augmented_prompt = PromptTemplate.from_template(f"""
    You are a helpful assistant, Perform the following task:
    1- Only Consider the context below to answer
        Context: {source_knowledge}        
    2- Answer the question: {query}

    """).format(source_knowledge=source_knowledge, query=query)
    logging.info(augmented_prompt)
    return augmented_prompt


def rag_prompt_template():
    template = """
        You are a helpful assistant, follow the following rules:
            1 - Answer the question {question} based only on the following context: {context}
            2 - if you don't know the answer, Don't try to make up the answer, reply I don't know the answer to that question
        
    """

    rag_prompt = PromptTemplate.from_template(template)
    return rag_prompt


if __name__ == "__main__":
    print(rag_prompt_template())
