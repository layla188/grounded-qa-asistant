# OrionWorks Grounded Q&A Assistant

A Retrieval-Augmented Generation (RAG) application for answering questions about the **OrionWorks Employee Handbook & Operations Knowledge Base**.

The system retrieves relevant information from the source PDF, reranks the retrieved passages, and generates a grounded answer using an LLM. Each answer includes the relevant source pages from the original document.

---

## Project Overview

Traditional keyword search may fail when a user's question uses different wording from the source document.

For example, a user might ask:

> Can I work from home during my first two months?

While the handbook may state:

> Employees within their first 60 days are not eligible for remote work.

A semantic retrieval system can understand that these two statements refer to the same concept.

This project uses **Retrieval-Augmented Generation (RAG)** to connect a language model with a specific knowledge base.

Instead of relying only on the LLM's general knowledge, the system:

1. Loads the OrionWorks PDF.
2. Splits the document into smaller chunks.
3. Converts the chunks into embeddings.
4. Stores the embeddings in a FAISS vector store.
5. Retrieves relevant chunks for a user question.
6. Reranks the retrieved chunks using FlashRank.
7. Sends the most relevant context to the LLM.
8. Generates a grounded answer.
9. Displays the source pages used for the answer.

---

# System Architecture

```text
                    OrionWorks PDF
                    Employee Handbook
                           |
                           v
                  +-------------------+
                  |    PDF Loader     |
                  |    PyPDFLoader    |
                  +---------+---------+
                            |
                            v
                  +-------------------+
                  |   Text Chunking   |
                  | Recursive Splitter|
                  +---------+---------+
                            |
                            v
                  +-------------------+
                  |    Embeddings     |
                  | BGE-small-en-v1.5 |
                  +---------+---------+
                            |
                            v
                  +-------------------+
                  |    FAISS Index    |
                  |   Vector Store    |
                  +---------+---------+
                            |
                     User Question
                            |
                            v
                  +-------------------+
                  | Semantic Retrieval|
                  |      Top-K        |
                  +---------+---------+
                            |
                            v
                  +-------------------+
                  |    FlashRank      |
                  |     Reranker      |
                  +---------+---------+
                            |
                            v
                  +-------------------+
                  |  Grounded Prompt  |
                  | Context + Query   |
                  +---------+---------+
                            |
                            v
                  +-------------------+
                  |       LLM         |
                  |    OpenRouter     |
                  +---------+---------+
                            |
                            v
                  +-------------------+
                  |   Final Answer    |
                  |  + Source Pages   |
                  +-------------------+
```

---

# Features

## Grounded Question Answering

The assistant answers questions using information retrieved from the OrionWorks knowledge base.

The generation prompt instructs the model to:

- Use only the provided context.
- Avoid outside knowledge.
- Avoid inventing policies.
- Avoid inventing numbers.
- Avoid inventing dates.
- Avoid inventing contacts or exceptions.
- State when the requested information is not available.

This helps reduce hallucinations and keeps generated answers grounded in the source document.

---

## Semantic Search

The system uses embeddings instead of relying only on exact keyword matching.

Both the document chunks and the user question are converted into vector representations.

FAISS then searches for chunks that are semantically similar to the question.

---

## Reranking

The first retrieval stage returns multiple potentially relevant chunks.

The retrieved chunks are then passed to **FlashRank**, which reranks them according to their relevance to the user's specific question.

The pipeline is:

```text
User Question
      |
      v
FAISS Retrieval
      |
      v
Top 10 Candidates
      |
      v
FlashRank Reranking
      |
      v
Top 4 Relevant Chunks
      |
      v
LLM
```

---

## Source Citations

Each generated answer displays the PDF pages containing the supporting information.

Example:

```text
Sources

Page 3
Page 7
Page 4
```

Users can also expand:

```text
View supporting information
```

to inspect the retrieved passages.

---

## Conversational Interface

The Streamlit application supports multiple questions in the same conversation.

Users can:

- Ask a question.
- Receive an answer.
- Ask another question.
- Ask follow-up questions.
- Choose suggested questions.
- Continue the conversation without starting a new chat.
- Start a new conversation when needed.

---

# Knowledge Base

The project uses a synthetic document:

**OrionWorks Employee Handbook & Operations Knowledge Base**

The knowledge base contains information about:

- Company profile and contacts
- Working hours and attendance
- Annual leave and time off
- Hybrid and remote work
- Business travel and expenses
- Procurement and purchasing rules
- IT accounts and passwords
- Devices and software
- Information security
- Performance and learning
- Internal mobility
- Workplace safety
- Incident reporting
- Escalation procedures
- Frequently asked questions
- Glossary

The document also contains questions specifically designed to test RAG retrieval and grounding.

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | User interface |
| LangChain | RAG pipeline components |
| PyPDF | PDF document loading |
| RecursiveCharacterTextSplitter | Document chunking |
| Hugging Face | Embedding model |
| BAAI/bge-small-en-v1.5 | Text embeddings |
| FAISS | Vector similarity search |
| FlashRank | Document reranking |
| OpenRouter | LLM API |
| python-dotenv | Environment variable management |

---

# Embedding Model

The project uses:

```text
BAAI/bge-small-en-v1.5
```

The model was selected because:

- The current knowledge base is in English.
- It provides semantic text embeddings.
- It is relatively lightweight.
- It can run locally.
- It does not require an embedding API key.

The generated embeddings have:

```text
384 dimensions
```

The ingestion script verifies the embedding dimension during execution.

---

# LLM

The generation model is accessed through **OpenRouter** using an OpenAI-compatible API.

The model is configured through the `.env` file:

```env
LLM_MODEL=your_model_here
```

This allows the generation model to be changed without modifying the main application code.

The API key is also stored in `.env` and is never included directly in the source code.

---

# RAG Pipeline

## 1. Document Loading

The PDF is loaded using:

```python
PyPDFLoader
```

The loader preserves page information in the document metadata.

This is important because the application uses this metadata to display source page citations.

---

## 2. Document Chunking

The loaded document is split into smaller chunks using:

```python
RecursiveCharacterTextSplitter
```

Current configuration:

```text
Chunk Size: 800
Chunk Overlap: 150
```

The overlap helps preserve context when information is located near a chunk boundary.

For the current OrionWorks PDF:

```text
Pages Loaded: 9
Chunks Created: 33
```

---

## 3. Embedding Generation

Each document chunk is converted into a vector using:

```text
BAAI/bge-small-en-v1.5
```

The vector represents the semantic meaning of the text.

---

## 4. Vector Store

The generated embeddings are stored in:

```text
FAISS
```

FAISS allows efficient similarity search between the user's question and the document chunks.

The local vector store is saved as:

```text
data/vectorstore/
├── index.faiss
└── index.pkl
```

---

## 5. Retrieval

When a user asks a question, the question is converted into an embedding.

The embedding is compared with the vectors stored in FAISS.

The system retrieves:

```text
Top 10 candidate chunks
```

These chunks are passed to the reranking stage.

---

## 6. Reranking

The retrieved chunks are passed to:

```text
FlashRank
```

FlashRank evaluates the relevance of each retrieved passage to the user's question.

The system keeps:

```text
Top 4 reranked chunks
```

These chunks are then used as the main context for the LLM.

---

## 7. Grounded Generation

The retrieved passages are formatted with their page numbers and inserted into the prompt.

The prompt also contains the user's question.

The LLM is instructed to answer only using the provided context.

If the context does not contain enough information, the assistant is instructed to respond:

```text
The information is not stated in the OrionWorks knowledge base.
```

This helps prevent the model from filling missing information with unsupported assumptions.

---

# Example

### Question

```text
How many annual leave days does a full-time employee receive?
```

### Answer

```text
A full-time employee receives 24 business days of paid annual
leave per calendar year. New employees accrue leave monthly
from their start date, and up to 5 unused annual leave days
may be carried into the next calendar year, expiring on 31 March.
```

### Sources

```text
Page 3
Page 7
Page 4
```

The answer is generated using the retrieved handbook information.

---

# Project Structure

```text
grounded-qa-assistant/
│
├── data/
│   ├── documents/
│   │   └── orionworks_handbook.pdf
│   │
│   └── vectorstore/
│       ├── index.faiss
│       └── index.pkl
│
├── src/
│   ├── config.py
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── reranking.py
│   └── generation.py
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

# File Responsibilities

## `src/config.py`

Contains the main project configuration:

- Embedding model
- LLM model
- OpenRouter API key
- Chunk size
- Chunk overlap
- Retrieval size
- Reranking size
- File paths

Example:

```python
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

RETRIEVAL_K = 10
RERANK_TOP_N = 4
```

---

## `src/ingestion.py`

Responsible for preparing the knowledge base.

Main workflow:

```text
PDF
 |
 v
Load Documents
 |
 v
Split into Chunks
 |
 v
Generate Embeddings
 |
 v
Build FAISS Vector Store
 |
 v
Save Vector Store
```

Run this script whenever the source PDF changes.

---

## `src/retrieval.py`

Responsible for:

- Loading the embedding model.
- Loading the FAISS vector store.
- Retrieving relevant document chunks.

Main workflow:

```text
Question
   |
   v
Question Embedding
   |
   v
FAISS Similarity Search
   |
   v
Retrieved Documents
```

---

## `src/reranking.py`

Responsible for reranking the initial retrieved documents using FlashRank.

Main workflow:

```text
Retrieved Documents
        |
        v
     FlashRank
        |
        v
Top Relevant Documents
```

---

## `src/generation.py`

Responsible for:

- Creating the LLM.
- Creating the grounding prompt.
- Formatting retrieved documents.
- Generating the final answer.
- Extracting source pages.

---

## `app.py`

The main Streamlit application.

It combines:

```text
Retrieval
+
Reranking
+
Generation
+
Conversation State
+
User Interface
```

The UI is designed for normal employees rather than exposing technical implementation details such as FAISS, embedding dimensions, or reranking scores.

---

# Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd grounded-qa-assistant
```

---

## 2. Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a file named:

```text
.env
```

Add:

```env
OPENROUTER_API_KEY=your_api_key_here
LLM_MODEL=your_model_here
```

Do not commit `.env` to GitHub.

The API key should remain private.

---

# Build the Vector Store

Before running the application for the first time, build the FAISS vector store.

Run:

```bash
python -m src.ingestion
```

Expected output:

```text
Loading PDF...
Loaded 9 pages.

Splitting documents...
Created 33 chunks.

Loading embedding model...

Embedding dimension: 384

Building FAISS vector store...

Saving vector store...

Ingestion completed successfully.
```

This creates:

```text
data/vectorstore/
├── index.faiss
└── index.pkl
```

---

# Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser.

Users can then ask questions about the OrionWorks knowledge base.

---

# Example Questions

### Annual Leave

```text
How many annual leave days do I get?
```

### Carried-Over Leave

```text
Can I carry unused leave into the next year?
```

### Remote Work

```text
Can I work remotely during my first 60 days?
```

### Procurement

```text
How many quotations are required for a $12,000 purchase?
```

### Purchase Approval

```text
Who approves purchases above $25,000?
```

### Supplier Rules

```text
Do I need to contact a supplier before creating a purchase order?
```

### Travel

```text
What is the hotel reimbursement limit?
```

### Security

```text
What should I do after entering my credentials into a phishing site?
```

### Information Security

```text
Can I paste confidential client information into a public generative AI tool?
```

### Performance

```text
How often are formal performance reviews conducted?
```

### Internal Mobility

```text
How long must an employee wait before applying for an internal role?
```

---

# Handling Unknown Information

A major requirement of the application is avoiding unsupported answers.

If the user asks something that is not covered by the knowledge base, the assistant should not guess.

For example:

```text
What is the company policy for working from Mars?
```

If the knowledge base contains no relevant information, the assistant should respond:

```text
The information is not stated in the OrionWorks knowledge base.
```

This behavior is important because the goal is to build a **grounded knowledge-base assistant**, not a general-purpose chatbot.

---

# Conversation Support

The application maintains conversation history using Streamlit session state.

Users can ask multiple questions without starting a new chat.

Example:

```text
User:
How many annual leave days do I get?

Assistant:
A full-time employee receives 24 business days...

User:
Can I carry unused days into next year?

Assistant:
Up to 5 unused annual leave days may be carried over...

User:
When do they expire?

Assistant:
Carried-over days expire on 31 March.
```

The **New conversation** button clears the current conversation when the user wants to start again.

---

# Retrieval Configuration

The current configuration is:

```python
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

RETRIEVAL_K = 10
RERANK_TOP_N = 4
```

## Chunk Size

`CHUNK_SIZE` controls the approximate amount of text contained in each document chunk.

A larger chunk provides more surrounding context but may contain more irrelevant information.

A smaller chunk can provide more precise retrieval but may lose important surrounding context.

---

## Chunk Overlap

`CHUNK_OVERLAP` controls how much text is shared between neighboring chunks.

The overlap helps preserve information that may otherwise be split between two chunks.

---

## Retrieval K

```python
RETRIEVAL_K = 10
```

The system initially retrieves 10 candidate chunks from FAISS.

The purpose of this stage is to provide a sufficiently broad candidate set for the reranker.

---

## Rerank Top N

```python
RERANK_TOP_N = 4
```

After reranking, the system keeps the four most relevant chunks for the generation stage.

---

# Why RAG?

RAG is useful when the information being queried is:

- Specific to an organization.
- Stored in external documents.
- Frequently updated.
- Not necessarily present in the LLM's training data.
- Required to be traceable to source documents.

With RAG, the underlying documents can be updated and re-indexed without retraining the language model.

---

# RAG vs Fine-Tuning

This project uses **RAG rather than fine-tuning** because the main goal is retrieving factual information from a knowledge base.

### RAG

```text
Documents
    |
    v
Embeddings
    |
    v
Vector Store
    |
    v
Retrieve Relevant Context
    |
    v
LLM
```

### Fine-Tuning

```text
Training Dataset
    |
    v
Model Training
    |
    v
Fine-Tuned Model
```

Fine-tuning is generally more useful for changing model behavior, style, or task-specific patterns.

RAG is more suitable when the application needs to answer questions based on external documents and provide references to those documents.

---

# Grounding Strategy

The project uses several techniques to improve answer grounding.

## 1. Retrieved Context

The LLM receives relevant passages retrieved from the knowledge base.

## 2. Explicit Prompt Instructions

The prompt instructs the LLM:

```text
Use only information from the provided context.
```

## 3. No Outside Knowledge

The model is explicitly instructed not to use information outside the supplied context.

## 4. Unknown Answer Handling

The model is instructed to state when the information is not available.

## 5. Reranking

FlashRank improves the ordering of retrieved passages before generation.

## 6. Source Pages

The application displays the PDF pages associated with the retrieved information.

---

# Security Considerations

The project uses an API key for the LLM provider.

The key is stored in:

```text
.env
```

and excluded from Git using:

```gitignore
.env
```

The API key should never be included in:

- Source code
- GitHub repositories
- README files
- Screenshots
- Public configuration files
- Public deployment code

The local vector store is also excluded from Git by default:

```gitignore
data/vectorstore/
```

---

# `.gitignore`

The project uses the following `.gitignore`:

```gitignore
.env
__pycache__/
*.pyc
.venv/
data/vectorstore/
```

This prevents sensitive environment variables, Python cache files, the virtual environment, and the generated local vector store from being committed to the repository.

---

# `requirements.txt`

The project dependencies are:

```text
streamlit
langchain
langchain-community
langchain-text-splitters
langchain-huggingface
langchain-openai
sentence-transformers
faiss-cpu
pypdf
flashrank
python-dotenv
pandas
```

Install them using:

```bash
pip install -r requirements.txt
```

---

# Limitations

This project is designed as a RAG learning and demonstration project.

Current limitations include:

- The knowledge base is a synthetic document.
- The system depends on retrieval quality.
- Incorrect retrieval can affect the generated answer.
- Reranking improves relevance but does not guarantee perfect retrieval.
- The assistant can only answer from information available in the knowledge base.
- The LLM requires a valid API configuration.
- Page citations correspond to the original PDF pages containing retrieved chunks.
- The current application uses a single knowledge base.

---

# Future Improvements

Possible future improvements include:

- Supporting multiple PDFs.
- Adding metadata filtering.
- Adding hybrid keyword + semantic search.
- Improving citation formatting.
- Adding query rewriting.
- Adding conversation-aware retrieval.
- Adding document version management.
- Adding authentication.
- Improving the Streamlit interface.
- Comparing different embedding models.
- Comparing different chunking strategies.
- Comparing retrieval with and without reranking.
- Adding a larger evaluation dataset.
- Deploying the application for real users.

---

# Project Workflow

The complete system follows this workflow:

```text
                     INGESTION
                         |
                         v
              Load OrionWorks PDF
                         |
                         v
                  Split into Chunks
                         |
                         v
                Generate Embeddings
                         |
                         v
                  Build FAISS Index
                         |
                         v
                  Save Vector Store
                         |
                         |
                         v
                       QUERY
                         |
                         v
                   User Question
                         |
                         v
                Semantic Retrieval
                         |
                         v
                   Top 10 Chunks
                         |
                         v
                     FlashRank
                         |
                         v
                    Top 4 Chunks
                         |
                         v
                  Grounded Prompt
                         |
                         v
                        LLM
                         |
                         v
                   Final Answer
                         |
                         v
                    Source Pages
```

---

# Project Status

The current implementation includes:

- [x] PDF document loading
- [x] Document chunking
- [x] Local embedding generation
- [x] FAISS vector store
- [x] Semantic retrieval
- [x] FlashRank reranking
- [x] Grounded LLM generation
- [x] Source page extraction
- [x] Streamlit interface
- [x] Conversation history
- [x] Suggested questions
- [x] Supporting information section
- [x] Environment variable configuration
- [x] End-to-end RAG pipeline

---

# Key Learning Outcomes

This project demonstrates practical understanding of the main components of a Retrieval-Augmented Generation system.

### Document Processing

Loading a source document and transforming it into smaller searchable chunks.

### Embeddings

Representing text as numerical vectors that capture semantic meaning.

### Vector Search

Using FAISS to retrieve document chunks that are semantically related to a user's question.

### Retrieval

Selecting potentially relevant information before sending context to the language model.

### Reranking

Using a dedicated reranker to improve the relevance ordering of retrieved passages.

### Grounded Generation

Providing retrieved source context to an LLM so that the generated answer is based on the knowledge base.

### Citations

Connecting the generated answer back to the original PDF pages.

### Application Development

Turning the RAG pipeline into a usable conversational application using Streamlit.

---

# Conclusion

The **OrionWorks Grounded Q&A Assistant** demonstrates a complete end-to-end Retrieval-Augmented Generation workflow.

Instead of relying on an LLM alone, the system combines:

```text
Document Processing
        +
Semantic Embeddings
        +
Vector Search
        +
Reranking
        +
Grounded Generation
        +
Source Citations
```

This approach provides a practical way to build question-answering applications over specific knowledge bases while keeping generated answers grounded in retrieved source information.

The project provides a foundation for building more advanced RAG systems over larger and more diverse document collections.#   g r o u n d e d - q a - a s i s t a n t  
 