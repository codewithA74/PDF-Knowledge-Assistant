# 1. Load Environment Variables
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader   
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit as st 
import os 


## data in st session 

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None 
       
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    
## comnication B|w user and ai 
if "messages" not in st.session_state:
    st.session_state.messages = []
   
    
    
st.title("PDF Knowledge Assistant")

st.subheader("AI-Powered Retrieval-Augmented Generation (RAG)")

st.caption(
    "Upload one or more PDF files and ask questions in natural language."
)

with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    **Steps:**
    1. Upload one or more PDF files.
    2. Wait until the PDFs are processed.
    3. Type your question in the chat box.
    4. The AI will answer using the uploaded documents.
    5. Upload new PDFs anytime to start a new session.
    """)

    st.divider()

    st.info("Supported file format: PDF")

    st.caption("Built with Streamlit • LangChain • Hugging Face • Groq • LangGraph")
     
    
def process_document(path):    
    
    # 2. Load PDF
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    
    # 3. Split the Documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    documents = splitter.split_documents(documents)
    
    # 4. Create Embeddings
    embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
    
    # 5. Store Embeddings in Vector Database
    vector_db= InMemoryVectorStore.from_documents(
    documents=documents,
    embedding=embeddings
    )
    
        # 6. Load LLM
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    # 7. Create Tool
    @tool
    def retrieve_context(query: str) -> str:
        """
        Retrieve relevant context from the PDF.
        """
        docs = vector_db.similarity_search(query, k=4)
        context = ""
        for doc in docs:
            context += doc.page_content + "\n\n"
        return context
    
    # 8. System Prompt
    system_prompt = """
    You are a helpful AI assistant.
    Always use the retrieve_context tool
    before answering questions related to
    the uploaded PDF.
    """
    
    # 9. Memory
    memory = InMemorySaver()
    
    
    # 10. Create Agent
    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )


# 11. Chat Loop
#while True:

#   if query.lower() == "quit":
#     break
#  response = agent.invoke(
#      {
#        "messages": [
#             {
#                "role": "user",
#              "content": query
#           }
#      ]
#   },
#   {
#       "configurable": {
#          "thread_id": "1"
#      }
#  }
# ) 
# result =  response["messages"][-1].content
#   print("AI:", result)
    
    st.session_state.agent = agent
    st.session_state.document_uploaded = True 
    
    
## upload ui      ##upload ui section show tab tak ho ga jab pdf uploade nahi ke hai
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Select PDF files",type=["pdf"],accept_multiple_files= True) 
    if uploaded:
        with st.spinner("Processing...."):
            path = "./doc_files/"
            os.makedirs(path, exist_ok=True)
            
            for file in uploaded:
             with open(os.path.join(path, file.name), "wb") as f:
                f.write(file.getvalue())
            
            process_document(path) ## it process our document
            st.rerun()  ##this is not refress ouver page , it refress our variable and  reload   UI 


## chat ui  
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content =message.get("content")
        st.chat_message(role).markdown(content)
        

    query =  st.chat_input("Ask anything related to uploaded do")
    if query:
        st.session_state.messages.append({"role":"user","content":query}) 
    
        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages":[{"role":"user","content":query}]},
            {"configurable":{"thread_id":1}}
        )
        result =  response["messages"][-1].content  
        st.chat_message("assistant").markdown(result)
        st.session_state.messages.append({"role":"assistant","content":result}) 
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
