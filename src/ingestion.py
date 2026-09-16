from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import (
    PDF_PATH,
    VECTORSTORE_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    return chunks


def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    return embeddings


def build_vectorstore(chunks, embeddings):
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


def save_vectorstore(vectorstore):
    vectorstore.save_local(
        VECTORSTORE_PATH
    )


def main():

    print("Loading PDF...")

    documents = load_documents(PDF_PATH)

    print(f"Loaded {len(documents)} pages.")

    print("Splitting documents...")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Loading embedding model...")

    embeddings = create_embeddings()

    print("Building FAISS vector store...")

    vectorstore = build_vectorstore(
        chunks,
        embeddings
    )

    print("Saving vector store...")

    save_vectorstore(vectorstore)

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    main()