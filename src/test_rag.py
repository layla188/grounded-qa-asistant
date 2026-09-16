
from src.retrieval import (
    load_embeddings,
    load_vectorstore,
    retrieve_documents,
)

from src.reranking import (
    create_reranker,
    rerank_documents,
)

from src.generation import (
    create_llm,
    create_prompt,
    generate_answer,
    get_sources,
)

from src.config import (
    RETRIEVAL_K,
)


def main():

    question = (
        "How many annual leave days does a "
        "full-time employee receive?"
    )

    # ========================================================
    # 1. Load embeddings
    # ========================================================

    print("Loading embedding model...")

    embeddings = load_embeddings()

    # ========================================================
    # 2. Load FAISS
    # ========================================================

    print("Loading FAISS vector store...")

    vectorstore = load_vectorstore(
        embeddings
    )

    # ========================================================
    # 3. Retrieve
    # ========================================================

    print("Retrieving documents...")

    documents = retrieve_documents(
        question,
        vectorstore
    )

    print(
        f"Retrieved {len(documents)} documents."
    )

    # ========================================================
    # 4. Rerank
    # ========================================================

    print("Loading reranker...")

    ranker = create_reranker()

    print("Reranking documents...")

    reranked_documents = rerank_documents(
        question,
        documents,
        ranker
    )

    print(
        f"Selected {len(reranked_documents)} documents."
    )

    # ========================================================
    # 5. Load LLM
    # ========================================================

    print("Loading LLM...")

    llm = create_llm()

    prompt = create_prompt()

    # ========================================================
    # 6. Generate answer
    # ========================================================

    print("\nGenerating answer...")

    answer = generate_answer(
        question,
        reranked_documents,
        llm,
        prompt,
    )

    # ========================================================
    # 7. Sources
    # ========================================================

    sources = get_sources(
        reranked_documents
    )

    print("\n" + "=" * 60)
    print("QUESTION")
    print("=" * 60)

    print(question)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(answer)

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    print(
        ", ".join(
            f"Page {page}"
            for page in sources
        )
    )


if __name__ == "__main__":
    main()