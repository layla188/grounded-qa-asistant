import streamlit as st

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


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="OrionWorks Assistant",
    page_icon="O",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
<style>

    /* ========================================================
       Global
       ======================================================== */

    .stApp {
        background-color: #f8fafc;
    }

    .main .block-container {
        max-width: 1000px;
        padding-top: 1.5rem;
        padding-bottom: 6rem;
    }


    /* ========================================================
       Sidebar
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #172554 !important;
    }

    section[data-testid="stSidebar"] p {
        color: #64748b !important;
        line-height: 1.6;
    }


    /* ========================================================
       Main Text
       ======================================================== */

    .stMarkdown p {
        color: #475569;
    }

    h1 {
        color: #172554 !important;
        font-weight: 750 !important;
    }

    h2 {
        color: #172554 !important;
    }

    h3 {
        color: #1e293b !important;
    }


    /* ========================================================
       Welcome Header
       ======================================================== */

    .welcome-title {
        text-align: center;
        font-size: 2.7rem;
        font-weight: 750;
        color: #172554;
        margin-top: 4rem;
        margin-bottom: 0.8rem;
        letter-spacing: -1px;
    }

    .welcome-subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 2.5rem;
    }


    /* ========================================================
       Main Question Input
       ======================================================== */

    div[data-testid="stTextInput"] input {
        height: 3.8rem;
        border-radius: 14px;
        border: 1px solid #cbd5e1;
        background-color: #ffffff;
        color: #1e293b;
        font-size: 1rem;
        padding-left: 1rem;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #2563eb;
        box-shadow: 0 0 0 1px #2563eb;
    }


    /* ========================================================
       Suggestion Buttons
       ======================================================== */

    .stButton > button {
        min-height: 3rem;
        border-radius: 11px;
        border: 1px solid #e2e8f0;
        background-color: #ffffff;
        color: #334155;
        font-size: 0.9rem;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #93c5fd;
        background-color: #eff6ff;
        color: #1d4ed8;
    }


    /* ========================================================
       Chat Messages
       ======================================================== */

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        margin-bottom: 1rem;
    }


    /* ========================================================
       Assistant Answer
       ======================================================== */

    .answer-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }


    /* ========================================================
       Source Cards
       ======================================================== */

    .source-label {
        font-size: 0.8rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.6rem;
    }


    /* ========================================================
       Expander (Fixed Contrast & Text Colors)
       ======================================================== */

    [data-testid="stExpander"] {
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        background-color: #ffffff !important;
    }

    [data-testid="stExpander"] summary {
        color: #1e293b !important;
        font-weight: 600 !important;
    }

    [data-testid="stExpander"] summary:hover {
        color: #2563eb !important;
    }

    [data-testid="stExpander"] p,
    [data-testid="stExpander"] div,
    [data-testid="stExpander"] span,
    [data-testid="stExpander"] label {
        color: #334155 !important;
    }


    /* ========================================================
       Chat Input
       ======================================================== */

    [data-testid="stChatInput"] {
        border-radius: 13px;
    }


    /* ========================================================
       Footer
       ======================================================== */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        margin-top: 3rem;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# Load RAG Resources
# ============================================================

@st.cache_resource
def load_resources():
    embeddings = load_embeddings()
    vectorstore = load_vectorstore(embeddings)
    ranker = create_reranker()
    llm = create_llm()
    prompt = create_prompt()

    return (
        vectorstore,
        ranker,
        llm,
        prompt,
    )


# ============================================================
# Session State
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None


suggestions = [
    "How many annual leave days do I get?",
    "Can I work remotely during my first 60 days?",
    "Who approves purchases above $25,000?",
    "What should I do after entering credentials into a phishing site?",
]


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("OrionWorks")
    st.caption("Employee Knowledge Assistant")
    st.divider()

    if st.button("New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.selected_question = None
        st.rerun()

    st.divider()

    st.subheader("Quick Questions")
    for idx, sug in enumerate(suggestions):
        if st.button(sug, key=f"sidebar_sug_{idx}", use_container_width=True):
            st.session_state.selected_question = sug
            st.rerun()

    st.divider()

    st.subheader("About")
    st.write(
        "Get quick answers about OrionWorks policies "
        "and workplace procedures."
    )

    st.subheader("You can ask about")
    st.write(
        "• Leave and time off\n\n"
        "• Remote work\n\n"
        "• Business travel\n\n"
        "• Purchasing and expenses\n\n"
        "• IT and account security\n\n"
        "• Workplace policies"
    )

    st.divider()

    st.caption(
        "Answers are based on the OrionWorks "
        "Employee Handbook."
    )


# ============================================================
# Capture Selected Question
# ============================================================

user_question = None

if st.session_state.selected_question:
    user_question = st.session_state.selected_question
    st.session_state.selected_question = None


# ============================================================
# Screen Rendering (Welcome or Chat)
# ============================================================

if not st.session_state.messages:

    # Welcome Screen Header
    st.markdown(
        """
        <div class="welcome-title">
            How can we help you today?
        </div>

        <div class="welcome-subtitle">
            Ask a question about the OrionWorks Employee Handbook.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Search Box
    initial_question = st.text_input(
        "Ask a question",
        placeholder=(
            "Ask about leave, remote work, expenses, "
            "security, or company policies..."
        ),
        label_visibility="collapsed",
        key="initial_question",
    )

    # Ask Button
    ask_col1, ask_col2, ask_col3 = st.columns([1, 2, 1])

    with ask_col2:
        ask_clicked = st.button(
            "Ask OrionWorks",
            use_container_width=True,
            type="primary",
        )

    # Suggested Questions Grid
    st.write("")
    st.markdown("#### Try asking")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(suggestions[0], key="suggestion_0", use_container_width=True):
            st.session_state.selected_question = suggestions[0]
            st.rerun()

        if st.button(suggestions[2], key="suggestion_2", use_container_width=True):
            st.session_state.selected_question = suggestions[2]
            st.rerun()

    with col2:
        if st.button(suggestions[1], key="suggestion_1", use_container_width=True):
            st.session_state.selected_question = suggestions[1]
            st.rerun()

        if st.button(suggestions[3], key="suggestion_3", use_container_width=True):
            st.session_state.selected_question = suggestions[3]
            st.rerun()

    # Determine Question from landing input
    if not user_question and ask_clicked and initial_question.strip():
        user_question = initial_question.strip()

else:

    # Chat Screen Header
    st.title("OrionWorks Assistant")
    st.caption("Ask follow-up questions about the Employee Handbook.")

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                st.divider()

                st.markdown("**Sources**")
                sources = message["sources"]

                if sources:
                    source_cols = st.columns(min(len(sources), 4))
                    for index, page in enumerate(sources):
                        with source_cols[index % len(source_cols)]:
                            st.info(f"Page {page}")
                else:
                    st.caption("No source page was available.")

                if message.get("documents"):
                    with st.expander("View supporting information"):
                        for doc_idx, document in enumerate(message["documents"]):
                            page = document.metadata.get("page", 0) + 1

                            st.markdown(f"**Page {page}**")
                            st.write(document.page_content)

                            if doc_idx < len(message["documents"]) - 1:
                                st.divider()

    # Chat Input for Follow-up
    chat_input_val = st.chat_input("Ask another question...")
    if not user_question and chat_input_val:
        user_question = chat_input_val


# ============================================================
# Process New Question
# ============================================================

if user_question:
    user_question = user_question.strip()

    if user_question:

        # Save User Message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        # Render current user input
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Preparing the assistant..."):
                    (
                        vectorstore,
                        ranker,
                        llm,
                        prompt,
                    ) = load_resources()

                with st.spinner("Searching the handbook..."):
                    documents = retrieve_documents(
                        user_question,
                        vectorstore,
                    )

                with st.spinner("Finding the most relevant information..."):
                    reranked_documents = rerank_documents(
                        user_question,
                        documents,
                        ranker,
                    )

                with st.spinner("Preparing your answer..."):
                    answer = generate_answer(
                        user_question,
                        reranked_documents,
                        llm,
                        prompt,
                    )

                sources = get_sources(reranked_documents)

                # Display Answer
                st.markdown(
                    '<div class="answer-title">Answer</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(answer)

                # Display Sources
                st.divider()
                st.markdown(
                    '<div class="source-label">Sources</div>',
                    unsafe_allow_html=True,
                )

                if sources:
                    source_cols = st.columns(min(len(sources), 4))
                    for index, page in enumerate(sources):
                        with source_cols[index % len(source_cols)]:
                            st.info(f"Page {page}")
                else:
                    st.caption("No source page was available.")

                # Display Supporting Information
                if reranked_documents:
                    with st.expander("View supporting information"):
                        for index, document in enumerate(reranked_documents):
                            page = document.metadata.get("page", 0) + 1

                            st.markdown(f"**Page {page}**")
                            st.write(document.page_content)

                            if index < len(reranked_documents) - 1:
                                st.divider()

                # Save Assistant Response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "documents": reranked_documents,
                    }
                )

                # Rerun to cleanly transition layout to multi-turn conversation
                st.rerun()

            except Exception as error:
                st.error("Sorry, I couldn't process that question.")
                st.caption(f"Technical details: {error}")


# ============================================================
# Footer
# ============================================================

if not st.session_state.messages:
    st.markdown(
        """
        <div class="footer">
            OrionWorks Employee Knowledge Assistant
        </div>
        """,
        unsafe_allow_html=True,
    )