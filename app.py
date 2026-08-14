import os
import shutil

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent


# ============================================================
# 1. Load Environment Variables
# ============================================================

load_dotenv()


# ============================================================
# 2. Streamlit Session State
# ============================================================

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# 3. Page UI
# ============================================================

st.title("PDF Knowledge Assistant")

st.subheader("AI-Powered Retrieval-Augmented Generation (RAG)")

st.caption(
    "Upload one or more PDF files and ask questions about their content."
)


# ============================================================
# 4. Sidebar
# ============================================================

with st.sidebar:

    st.header("How to Use")

    st.markdown("""
    **Steps:**

    1. Upload one or more PDF files.
    2. Wait until the PDFs are processed.
    3. Ask a question about the PDF.
    4. The AI retrieves relevant information.
    5. The AI generates an answer from the PDF.
    """)

    st.divider()

    st.info("Supported file format: PDF")

    st.caption(
        "Built with Streamlit • LangChain • Hugging Face • Groq • LangGraph"
    )


# ============================================================
# 5. Process PDF
# ============================================================

def process_document(path):

    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    loader = PyPDFDirectoryLoader(path)

    documents = loader.load()

    if not documents:
        st.error("No readable text was found in the PDF.")
        return

    # --------------------------------------------------------
    # Debug information
    # --------------------------------------------------------

    print("Number of pages:", len(documents))

    for i, doc in enumerate(documents[:3]):

        print(f"\n--- PAGE {i + 1} ---")

        print(doc.page_content[:1000])

    # --------------------------------------------------------
    # Split documents
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print("Number of chunks:", len(chunks))

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # --------------------------------------------------------
    # Create vector store
    # --------------------------------------------------------

    vector_db = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # Save vector store in session
    st.session_state.vector_store = vector_db

    # --------------------------------------------------------
    # Check API key
    # --------------------------------------------------------

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:

        st.error(
            "GROQ_API_KEY was not found. "
            "Please add it to your .env file or Streamlit Secrets."
        )

        return

    # --------------------------------------------------------
    # Load LLM
    # --------------------------------------------------------

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        temperature=0
    )

    # --------------------------------------------------------
    # Retrieval Tool
    # --------------------------------------------------------

    @tool
    def retrieve_context(query: str) -> str:
        """
        Search the uploaded PDF and return the most relevant information.
        """

        docs = vector_db.similarity_search(
            query,
            k=6
        )

        if not docs:
            return "No relevant information was found in the PDF."

        context_parts = []

        for doc in docs:

            page = doc.metadata.get("page", "unknown")

            source = doc.metadata.get(
                "source",
                "uploaded PDF"
            )

            context_parts.append(
                f"Source: {source}\n"
                f"Page: {page}\n"
                f"Content:\n{doc.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # Debugging
        print("\nUSER QUERY:")
        print(query)

        print("\nRETRIEVED CONTEXT:")
        print(context)

        return context

    # --------------------------------------------------------
    # System Prompt
    # --------------------------------------------------------

    system_prompt = """
    You are a PDF question-answering assistant.

    Your job is to answer questions using ONLY information
    retrieved from the uploaded PDF.

    IMPORTANT RULES:

    1. Always use the retrieve_context tool for questions
       about the uploaded PDF.

    2. Do not guess information.

    3. Do not invent patient names, ages, dates, diagnoses,
       or other information.

    4. If the requested information is not available in
       the retrieved PDF context, clearly say:

       "I could not find this information in the uploaded PDF."

    5. If the information is available, answer directly
       and clearly.

    6. For patient information, copy the information
       exactly from the retrieved PDF context whenever possible.
    """

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    memory = InMemorySaver()

    # --------------------------------------------------------
    # Create Agent
    # --------------------------------------------------------

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    # Save agent
    st.session_state.agent = agent

    st.session_state.document_uploaded = True


# ============================================================
# 6. PDF Upload UI
# ============================================================

if not st.session_state.document_uploaded:

    uploaded = st.file_uploader(
        label="Select PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:

        with st.spinner("Processing PDFs..."):

            path = "./doc_files"

            # Create folder
            os.makedirs(path, exist_ok=True)

            # Remove old PDFs
            for filename in os.listdir(path):

                file_path = os.path.join(
                    path,
                    filename
                )

                if os.path.isfile(file_path):

                    os.remove(file_path)

            # Save new PDFs
            for file in uploaded:

                file_path = os.path.join(
                    path,
                    file.name
                )

                with open(file_path, "wb") as f:

                    f.write(file.getvalue())

            # Process documents
            process_document(path)

        st.rerun()


# ============================================================
# 7. Chat UI
# ============================================================

if (
    st.session_state.document_uploaded
    and st.session_state.agent
):

    # Display previous messages
    for message in st.session_state.messages:

        role = message["role"]

        content = message["content"]

        st.chat_message(role).markdown(content)

    # Chat input
    query = st.chat_input(
        "Ask anything about the uploaded PDF..."
    )

    if query:

        # ----------------------------------------------------
        # Display user message
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        st.chat_message("user").markdown(query)

        # ----------------------------------------------------
        # Ask Agent
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner("Searching the PDF..."):

                response = st.session_state.agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": query
                            }
                        ]
                    },
                    {
                        "configurable": {
                            "thread_id": "pdf-chat-1"
                        }
                    }
                )

                result = response["messages"][-1].content

                st.markdown(result)

        # ----------------------------------------------------
        # Save assistant message
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result
            }
        )
