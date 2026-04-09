import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# --- Configuration ---
DB_PATH = "chroma_db_medical"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"
# ---------------------

# Verify the database directory exists
if not os.path.exists(DB_PATH):
    st.error(f"Vector database not found at {DB_PATH}.")
    st.stop()

# --- RAG Prompt Template ---
# This prompt guides the LLM to act as a helpful pharmacy agent [cite: 826]
RAG_PROMPT_TEMPLATE = """
You are an AI assistant for a hospital pharmacy.
Your task is to answer user queries about medicines based *only* on the provided context.

CONTEXT:
{context}

QUERY:
{question}

Follow these instructions:
1.  Analyze the user's query.
2.  Use the retrieved data in the CONTEXT to find the most relevant information.
3.  Answer in a clear, human-like, and helpful manner[cite: 826].
4.  **Stock Check**: If the query is about a specific medicine, check its 'Stock' status.
5.  **Alternative Suggestion**: If the 'Stock' is 'No', state it is 'out of stock' and *always* suggest the 'Alternative' medicine listed[cite: 835, 840].
6.  **Dosage**: If the medicine is available, provide the 'Dosage_Instruction'[cite: 826, 838].
7.  **FAQ**: If the query is a general question (e.g., "medicine for fever"), list medicines from the CONTEXT whose 'Use_Case' matches[cite: 826, 835].
8.  If the context does not contain an answer, state that you do not have that information.
"""

def format_docs(docs):
    """Helper function to format retrieved documents into a string."""
    return "\n\n".join(
        f"Record {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)
    )

@st.cache_resource
def get_rag_chain():
    """Initializes and returns the RAG chain (cached for performance)."""
    try:
        # Initialize LLM and Embeddings from Ollama 
        llm = OllamaLLM(model=LLM_MODEL)
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)

        # Load the persistent ChromaDB 
        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )

        # Create the retriever 
        retriever = vectorstore.as_retriever(k=3) # Retrieve top 3 results

        # Create the RAG prompt
        rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        # Create the RAG chain using LangChain Expression Language (LCEL)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain

    except Exception as e:
        st.error(f"Error initializing RAG chain: {e}")
        return None

# --- Streamlit User Interface --- 
st.title("🏥 AI Medical Shop Assistant")
st.caption(f"Powered by {LLM_MODEL}, LangChain, and ChromaDB")

# Initialize RAG chain
try:
    rag_chain = get_rag_chain()
    if rag_chain is None:
        st.stop()
except Exception as e:
    st.error(f"Failed to load AI model. Is Ollama running? Error: {e}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! How can I help you with your medicine query today?"}
    ]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("E.g., 'Do you have Paracetamol 500mg?'"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the RAG chain to get an answer 
                response = rag_chain.invoke(user_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")