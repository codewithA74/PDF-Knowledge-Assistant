# PDF Knowledge Assistant

An AI-powered **PDF Knowledge Assistant** built with **Python, Streamlit, LangChain, LangGraph, Hugging Face, and Groq**.

This project uses **Retrieval-Augmented Generation (RAG)** to allow users to upload PDF documents and ask questions about their content in natural language.

## Features

* Upload one or multiple PDF files
* Extract text from PDFs
* Split documents into smaller chunks
* Generate embeddings using Hugging Face
* Store embeddings in an in-memory vector store
* Retrieve relevant information from uploaded PDFs
* Generate answers using Groq LLM
* Chat with the uploaded documents
* Maintain chat history during the Streamlit session
* Simple and beginner-friendly Streamlit interface

## How RAG Works

The basic flow of this project is:

```text
PDF Files
   ↓
Load PDF
   ↓
Split Text into Chunks
   ↓
Create Embeddings
   ↓
Store in Vector Store
   ↓
User Asks Question
   ↓
Similarity Search
   ↓
Retrieve Relevant Chunks
   ↓
Groq LLM
   ↓
AI Answer
```

### Example

Suppose you upload a PDF containing a company's leave policy.

You ask:

```text
How many paid leaves does an employee get?
```

The RAG system searches the uploaded PDF, finds the relevant information, and provides an answer based on the document.

## Technologies Used

* **Python** - Main programming language
* **Streamlit** - Web interface
* **LangChain** - LLM and RAG components
* **LangGraph** - Agent and memory workflow
* **Hugging Face** - Text embeddings
* **Groq** - Large Language Model
* **InMemoryVectorStore** - Vector storage
* **PyPDFDirectoryLoader** - PDF document loading

## Project Structure

```text
PDF-Knowledge-Assistant/
│
├── app.py
├── doc_files/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd PDF-Knowledge-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv env
```

Activate it on Windows:

```bash
env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not upload your `.env` file to GitHub.

Add this to `.gitignore`:

```text
.env
env/
__pycache__/
doc_files/
```

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the URL shown in your terminal, usually:

```text
http://localhost:8501
```

## How to Use

1. Start the Streamlit application.
2. Upload one or more PDF files.
3. The application processes the PDFs.
4. The documents are split into smaller chunks.
5. Embeddings are created for the chunks.
6. The embeddings are stored in the vector store.
7. Ask a question about the uploaded PDF.
8. The retrieval tool finds relevant chunks.
9. Groq generates the final answer.
10. The answer is displayed in the chat interface.

## Main Components

### 1. PDF Loader

```python
loader = PyPDFDirectoryLoader(path)
documents = loader.load()
```

This loads PDF files from the `doc_files` directory.

### 2. Text Splitter

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

The PDF text is divided into smaller pieces.

This is important because sending a very large document directly to the LLM is inefficient.

### 3. Embeddings

```python
embeddings = HuggingFaceEmbeddings(
    model="BAAI/bge-small-en-v1.5"
)
```

Embeddings convert text into numerical representations so that similar pieces of text can be found during retrieval.

### 4. Vector Store

```python
vector_db = InMemoryVectorStore.from_documents(
    documents=documents,
    embedding=embeddings
)
```

The vector store keeps the document embeddings and allows similarity searches.

### 5. Retrieval Tool

```python
@tool
def retrieve_context(query: str) -> str:
    docs = vector_db.similarity_search(query, k=4)
```

This searches for the four most relevant chunks for the user's question.

### 6. LLM

```python
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)
```

Groq provides the language model that generates the final response.

### 7. Agent

```python
agent = create_agent(
    model=llm,
    tools=[retrieve_context],
    system_prompt=system_prompt,
    checkpointer=memory
)
```

The agent can use the retrieval tool before answering questions about the uploaded documents.

## Important Note

This project uses:

```text
InMemoryVectorStore
```

This means the vector data is stored in memory while the application is running.

If the application restarts, the vector store is lost and the PDFs need to be processed again.

For a larger production application, you can replace it with a persistent vector database such as:

* FAISS
* Chroma
* Qdrant
* Pinecone
* Weaviate

## Future Improvements

Possible improvements for the next version:

* Add PDF page citations to answers
* Add source document names
* Add page numbers
* Support DOCX and TXT files
* Add persistent vector database
* Add document deletion
* Add multiple conversation sessions
* Add authentication
* Add streaming responses
* Improve hallucination control
* Deploy the application online

## Learning Outcomes

Through this project, you can learn:

* How RAG works
* PDF document loading
* Text chunking
* Embeddings
* Vector databases
* Similarity search
* LLM integration
* LangChain tools
* LangGraph agents
* Streamlit session state
* Environment variables
* Building an AI application with Python

## Author

**Code with Akhil**
Built as a practical Generative AI project to learn and implement a complete PDF-based RAG application.
