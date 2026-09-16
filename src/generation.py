from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import (
    LLM_MODEL,
    OPENROUTER_API_KEY,
)


def create_llm():

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )

    return llm


def create_prompt():

    prompt = ChatPromptTemplate.from_template(
        """
You are the OrionWorks Knowledge Base Assistant.

Your job is to answer questions using ONLY the provided context.

Rules:

1. Use only information from the provided context.
2. Do not use outside knowledge.
3. Do not invent policies, numbers, dates, contacts, or exceptions.
4. If the context does not contain enough information, say:
   "The information is not stated in the OrionWorks knowledge base."
5. Give a concise but complete answer.
6. Do not mention information that is not supported by the context.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    return prompt


def format_documents(documents):

    formatted_documents = []

    for document in documents:

        page = document.metadata.get("page", 0) + 1

        text = document.page_content

        formatted_documents.append(
            f"[Page {page}]\n{text}"
        )

    return "\n\n".join(formatted_documents)


def generate_answer(question, documents, llm, prompt):

    context = format_documents(documents)

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return answer


def get_sources(documents):

    pages = []

    for document in documents:

        page = document.metadata.get("page", 0) + 1

        if page not in pages:
            pages.append(page)

    return pages