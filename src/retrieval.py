from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    EMBEDDING_MODEL,
    VECTORSTORE_PATH,
    RETRIEVAL_K,
)

from src.reranking import (
    create_reranker,
    rerank_documents,
)


def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    return embeddings


def load_vectorstore(embeddings):

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vectorstore


def retrieve_documents(query, vectorstore):

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": RETRIEVAL_K
        }
    )

    documents = retriever.invoke(query)

    return documents


def main():

    print("Loading embedding model...")

    embeddings = load_embeddings()

    print("Loading FAISS vector store...")

    vectorstore = load_vectorstore(embeddings)

    print("Loading reranker...")

    ranker = create_reranker()

    question = "How many annual leave days does a full-time employee receive?"

    print("\nRetrieving documents with FAISS...")

    documents = retrieve_documents(
        question,
        vectorstore
    )

    print(f"FAISS retrieved {len(documents)} documents.")

    print("\nApplying reranker...")

    reranked_documents = rerank_documents(
        question,
        documents,
        ranker
    )

    print(
        f"Reranker selected {len(reranked_documents)} documents."
    )

    for i, document in enumerate(reranked_documents, 1):

        page = document.metadata.get("page")

        score = document.metadata.get(
            "rerank_score"
        )

        print("\n" + "=" * 60)
        print(f"Reranked Result {i}")
        print(f"Page: {page}")
        print(f"Score: {score}")
        print("=" * 60)

        print(document.page_content)


if __name__ == "__main__":
    main()