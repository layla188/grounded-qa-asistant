# OrionWorks Grounded Q&A Assistant

A complete **Retrieval-Augmented Generation (RAG)** application that answers employee questions using the **OrionWorks Employee Handbook & Operations Knowledge Base**.

The system retrieves relevant information from the handbook, reranks the retrieved passages, generates an answer grounded only in the retrieved context, and displays the supporting PDF pages.

---

## Overview

The OrionWorks Grounded Q&A Assistant was built to demonstrate an end-to-end RAG workflow over a company knowledge base.

Instead of relying on the language model's general knowledge, the application searches the OrionWorks handbook first and provides the relevant information to the LLM before generating an answer.

This makes the assistant more suitable for questions involving:

- Company policies
- Annual leave
- Hybrid and remote work
- Procurement
- Business travel
- Expenses
- IT support
- Password and access policies
- Information security
- Learning and development
- Internal mobility
- Workplace safety
- Escalation procedures

---

## How It Works

The application follows this pipeline:

```text
OrionWorks PDF
      |
      v
Document Loading
      |
      v
Text Chunking
      |
      v
BGE Embeddings
      |
      v
FAISS Vector Store
      |
      v
User Question
      |
      v
Semantic Retrieval
      |
      v
Top 10 Candidate Chunks
      |
      v
FlashRank Reranking
      |
      v
Top 4 Relevant Chunks
      |
      v
Grounded Prompt
      |
      v
OpenRouter LLM
      |
      v
Answer + Source Pages
```

---

# Features

## Grounded Question Answering

The assistant is instructed to answer using only the information retrieved from the OrionWorks knowledge base.

The generation prompt prevents the model from intentionally relying on outside knowledge and instructs it not to invent:

- Policies
- Numbers
- Dates
- Approval requirements
- Contacts
- Exceptions

If the answer is not available in the retrieved context, the assistant is instructed to respond:

```text
The information is not stated in the OrionWorks knowledge base.
```

---

## Semantic Retrieval

The application uses semantic search rather than relying only on exact keyword matches.

For example, a user may ask:

```text
Can I work from home during my first month?
```

while the handbook states:

```text
Hybrid eligibility starts after 60 calendar days.
```

The embedding model represents both texts as vectors, allowing the retrieval system to identify that they refer to the same topic.

---

## Reranking

The initial retrieval stage retrieves the 10 most semantically similar chunks from FAISS.

These chunks are then passed through **FlashRank**, which reranks them based on their relevance to the exact user question.

```text
User Question
      |
      v
FAISS
      |
      v
Top 10
      |
      v
FlashRank
      |
      v
Top 4
      |
      v
LLM
```

This helps reduce irrelevant context before the answer is generated.

---

## Source Citations

The system preserves page metadata from the original PDF.

Each answer displays the PDF pages associated with the retrieved evidence.

Example:

```text
Sources

Page 3
Page 4
Page 7
```

Users can also open the **View supporting information** section to inspect the retrieved document passages.

---

## Conversational Interface

The Streamlit interface allows users to continue asking questions in the same conversation.

Users can:

- Ask an initial question
- Receive a grounded answer
- Ask another question
- Select suggested questions
- Review source pages
- Inspect supporting passages
- Start a new conversation when needed

---

# Knowledge Base

The system uses the:

**OrionWorks Employee Handbook & Operations Knowledge Base**

OrionWorks is a fictional technology and professional-services company used for RAG testing.

The handbook includes information about:

- Company profile and contacts
- Working hours and attendance
- Annual leave
- Sick leave
- Personal days
- Hybrid and remote work
- Business travel
- Hotel and meal limits
- Procurement thresholds
- Purchase Orders
- Sole-source purchasing
- IT accounts
- Password rules
- Multi-factor authentication
- Phishing incidents
- IT support priorities
- Lost devices
- Information classification
- Generative AI usage
- Performance reviews
- Learning allowance
- Internal job applications
- Workplace safety
- Incident escalation
- Frequently asked questions

---

# Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| LangChain | RAG components and orchestration |
| PyPDFLoader | PDF loading |
| RecursiveCharacterTextSplitter | Document chunking |
| Hugging Face | Embedding model integration |
| BAAI/bge-small-en-v1.5 | Semantic embeddings |
| FAISS | Vector similarity search |
| FlashRank | Passage reranking |
| OpenRouter | LLM API |
| python-dotenv | Environment variable management |

---

# Embedding Model

The project uses:

```text
BAAI/bge-small-en-v1.5
```

The model was selected because the OrionWorks knowledge base is written in English and the model provides strong lightweight semantic embeddings.

The model produces:

```text
384-dimensional vectors
```

Embeddings are generated locally.

This means an external embedding API is not required.

---

# Document Processing

The OrionWorks PDF contains:

```text
9 pages
```

After chunking with the current configuration, the system creates:

```text
33 chunks
```

The chunking configuration is:

```python
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
```

---

## Why Chunk the Document?

Passing the entire PDF to the language model for every question would be inefficient.

Instead, the document is split into smaller pieces.

This allows the retrieval system to search only for the passages that are relevant to the user's question.

---

## Chunk Overlap

The project uses:

```python
CHUNK_OVERLAP = 150
```

This allows neighboring chunks to share some text.

Overlap helps preserve meaning when a policy, exception, or rule appears near a chunk boundary.

---

# Vector Search

The project uses:

```text
FAISS
```

FAISS stores the embeddings generated from the handbook chunks.

When a user submits a question:

1. The question is converted into an embedding.
2. FAISS compares the query vector with the stored document vectors.
3. The most semantically similar chunks are returned.

The project initially retrieves:

```python
RETRIEVAL_K = 10
```

---

# Reranking

The project uses FlashRank to perform a second retrieval stage.

FAISS first returns 10 candidate chunks.

FlashRank then evaluates the relationship between the question and each candidate passage.

The final context contains:

```python
RERANK_TOP_N = 4
```

Therefore:

```text
FAISS Top 10
     |
     v
FlashRank
     |
     v
Best 4 Chunks
```

---

# Grounded Generation

The four reranked chunks are formatted into a context block and sent to the LLM together with the user's question.

The prompt follows rules similar to:

```text
Use only the provided context.

Do not use outside knowledge.

Do not invent policies, numbers, dates,
contacts, or exceptions.

If the answer is not contained in the
knowledge base, say that the information
is not stated in the OrionWorks knowledge base.
```

This is the grounding mechanism used by the application.

---

# LLM

The application accesses the generation model through **OpenRouter**.

OpenRouter provides an OpenAI-compatible API, allowing the application to use LangChain's `ChatOpenAI` interface.

The model is configured through the environment:

```env
LLM_MODEL=openrouter/free
```

or another supported OpenRouter model.

This means the model can be changed without changing the main RAG code.

---

# Example Question

### User

```text
How many annual leave days does a full-time employee receive?
```

### Assistant

```text
A full-time employee receives 24 business days of paid
annual leave per calendar year.

New employees accrue annual leave monthly from their
employment start date.
```

### Sources

```text
Page 3
Page 4
Page 7
```

Users can expand the supporting information section to inspect the retrieved passages directly.

---

# Project Structure

```text
grounded-qa-assistant/
│
├── data/
│   │
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

Contains the main RAG configuration.

Example:

```python
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

RETRIEVAL_K = 10
RERANK_TOP_N = 4

PDF_PATH = "data/documents/orionworks_handbook.pdf"

VECTORSTORE_PATH = "data/vectorstore"
```

It also loads:

```python
OPENROUTER_API_KEY
LLM_MODEL
```

from the environment.

---

## `src/ingestion.py`

Responsible for preparing the knowledge base.

```text
PDF
 |
 v
Load Pages
 |
 v
Create Chunks
 |
 v
Generate Embeddings
 |
 v
Build FAISS Index
 |
 v
Save Vector Store
```

Run this script when the PDF changes or when a new vector store needs to be generated.

---

## `src/retrieval.py`

Responsible for loading the embedding model and FAISS index and retrieving relevant chunks.

```text
Question
   |
   v
Embedding
   |
   v
FAISS Search
   |
   v
Top 10 Chunks
```

---

## `src/reranking.py`

Responsible for reranking the documents retrieved from FAISS.

```text
Top 10 FAISS Results
        |
        v
     FlashRank
        |
        v
     Best 4
```

---

## `src/generation.py`

Responsible for:

- Creating the LLM
- Creating the grounded prompt
- Formatting retrieved context
- Generating answers
- Extracting source pages

---

## `app.py`

Provides the Streamlit user interface.

The application includes:

- Search input
- Conversational history
- Suggested questions
- Follow-up questions
- New conversation button
- Source pages
- Expandable supporting information

---

# Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project:

```bash
cd grounded-qa-assistant
```

---

## 2. Create a Virtual Environment

### Windows

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

# Requirements

The project uses the following dependencies:

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

---

# Environment Variables

Create a `.env` file in the project root.

```env
OPENROUTER_API_KEY=your_openrouter_api_key
LLM_MODEL=openrouter/free
```

Do not upload the `.env` file to GitHub.

---

# Build the Knowledge Base

Before running the application for the first time, build the FAISS vector store.

```bash
python -m src.ingestion
```

Example output:

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

This generates:

```text
data/vectorstore/
├── index.faiss
└── index.pkl
```

---

# Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

---

# Example Questions

## Leave

```text
How many annual leave days do I receive?
```

```text
Can I carry unused annual leave into the next year?
```

```text
When does carried-over leave expire?
```

---

## Remote Work

```text
Can I work remotely during my first month?
```

```text
How many remote-working days are allowed per week?
```

```text
Can I work remotely from another country?
```

---

## Procurement

```text
How many quotations are required for a $12,000 purchase?
```

```text
Who approves a purchase of $25,000 or more?
```

```text
Can a supplier begin work before the Purchase Order is issued?
```

```text
When is a sole-source justification required?
```

---

## Business Travel

```text
What is the standard hotel cap?
```

```text
When is a receipt required?
```

```text
How long do I have to submit an expense report?
```

```text
Can I claim mileage when using my own car?
```

---

## IT and Security

```text
What is the minimum password length?
```

```text
What should I do if I entered my password into a phishing website?
```

```text
How quickly must a stolen laptop be reported?
```

```text
Can confidential client information be pasted into a public AI chatbot?
```

---

## Learning and Internal Mobility

```text
What is the annual learning allowance?
```

```text
When can I apply for another internal role?
```

```text
When are promotion reviews held?
```

---

# Unknown Information Handling

The assistant is designed not to invent answers when the knowledge base does not contain enough information.

For example:

```text
What is the OrionWorks policy for company cars?
```

If no relevant company-car policy exists in the handbook, the expected response is:

```text
The information is not stated in the OrionWorks knowledge base.
```

---

# Conversation Support

The Streamlit application stores messages using session state.

This allows users to ask multiple questions during the same session.

Example:

```text
User:
How many annual leave days do I receive?

Assistant:
Employees receive 24 business days of annual leave.

User:
Can I carry some into next year?

Assistant:
Yes. Up to 5 unused annual leave days may be carried over.

User:
When do they expire?

Assistant:
Carried-over annual leave expires on 31 March.
```

The user can select **New conversation** when they want to reset the current chat.

---

# Retrieval Configuration

The current system uses:

```python
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

RETRIEVAL_K = 10
RERANK_TOP_N = 4
```

### `CHUNK_SIZE`

Controls the approximate amount of text in each searchable unit.

### `CHUNK_OVERLAP`

Allows neighboring chunks to share context.

### `RETRIEVAL_K`

Controls how many candidate chunks FAISS initially retrieves.

### `RERANK_TOP_N`

Controls how many reranked chunks are sent to the LLM.

---

# Why Use RAG?

RAG is useful for organization-specific information because the LLM does not need to memorize the entire knowledge base.

Instead:

```text
Knowledge Base
      |
      v
Retrieval
      |
      v
Relevant Information
      |
      v
LLM
```

This provides several benefits:

- Company information can be updated without retraining the LLM.
- Answers can be linked to source documents.
- Less irrelevant context needs to be passed to the model.
- The system can work with private or organization-specific information.
- Hallucination risk can be reduced through grounding.

---

# RAG vs Fine-Tuning

This project uses RAG instead of fine-tuning because the goal is to provide access to factual information stored in a document.

## RAG

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
Retrieval
   |
   v
LLM
```

## Fine-Tuning

```text
Training Examples
      |
      v
Model Training
      |
      v
Modified Model
```

Fine-tuning is generally better suited for changing model behavior or teaching task patterns.

RAG is better suited for accessing changing external knowledge.

---

# Grounding Strategy

Several techniques are used to improve answer grounding.

## Retrieved Context

Only relevant document passages are passed to the LLM.

## Explicit Prompt Rules

The prompt tells the LLM to use only the supplied information.

## Unknown-Answer Instruction

The model is instructed not to guess when the answer is unavailable.

## Reranking

The strongest retrieved passages are prioritized before generation.

## Page Metadata

Retrieved documents retain their original PDF page numbers.

## Supporting Information

Users can inspect the passages used by the system.

---

# Security

The OpenRouter API key is stored in:

```text
.env
```

and should never be committed to the repository.

Recommended `.gitignore`:

```gitignore
.env
__pycache__/
*.pyc
.venv/
data/vectorstore/
```

Never expose API keys in:

- Source code
- GitHub repositories
- Screenshots
- README files
- Public configuration

---

# Limitations

The current project has several limitations:

- OrionWorks is a fictional company.
- The project currently uses one PDF knowledge base.
- Retrieval quality depends on the embedding model and chunking strategy.
- Reranking improves relevance but does not guarantee perfect retrieval.
- The LLM may still produce imperfect responses.
- The assistant is limited to the information contained in the knowledge base.
- Page citations identify retrieved source pages rather than sentence-level citations.
- The current conversation interface stores history for the active Streamlit session.

---

# Possible Future Improvements

Future versions could include:

- Multiple PDF support
- Automatic document ingestion
- Hybrid keyword and semantic search
- Metadata filtering
- Conversation-aware query rewriting
- Improved source citation formatting
- Highlighted supporting sentences
- Authentication
- Role-based document access
- Persistent chat history
- Document versioning
- Larger evaluation datasets
- Automated RAG evaluation
- Different embedding model comparisons
- Different chunking experiments
- Retrieval performance dashboards
- Cloud deployment

---

# Project Status

Current implementation:

- [x] PDF ingestion
- [x] PDF page metadata
- [x] Recursive text chunking
- [x] BGE embeddings
- [x] FAISS vector search
- [x] Semantic retrieval
- [x] FlashRank reranking
- [x] Top-4 context selection
- [x] OpenRouter LLM generation
- [x] Grounded prompt
- [x] Unknown-information handling
- [x] PDF page citations
- [x] Supporting source excerpts
- [x] Streamlit application
- [x] Conversation history
- [x] Suggested questions
- [x] New conversation functionality

---

# Key Learning Outcomes

This project demonstrates the main components of a practical RAG application.

## Document Ingestion

Loading raw documents and preserving useful metadata.

## Chunking

Breaking large documents into retrieval-friendly units.

## Embeddings

Representing semantic meaning using numerical vectors.

## Vector Databases

Using FAISS to perform efficient similarity search.

## Retrieval

Finding document passages related to a user's question.

## Reranking

Improving the ordering of retrieved passages before generation.

## Grounded Generation

Providing retrieved evidence to an LLM instead of asking it to answer from memory alone.

## Citations

Connecting generated answers back to source pages.

## Application Development

Turning a RAG pipeline into a user-facing conversational application with Streamlit.

---

# Conclusion

The **OrionWorks Grounded Q&A Assistant** demonstrates a complete Retrieval-Augmented Generation pipeline:

```text
PDF
 |
 v
Chunking
 |
 v
Embeddings
 |
 v
FAISS Retrieval
 |
 v
FlashRank Reranking
 |
 v
Grounded LLM Generation
 |
 v
Answer + Sources
```

The project shows how retrieval, reranking, grounding, and source attribution can be combined to create a practical knowledge-base assistant.

Rather than relying on an LLM alone, the system connects the model to a specific source of trusted information and provides users with the evidence behind each answer.