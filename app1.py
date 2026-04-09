import streamlit as st
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Dual AI Chat Assistant",
    layout="wide"
)

# --- Configuration ---
DB_PATH = "chroma_db_medical_5k"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama-3.1-8b-instant"


os.environ["GROQ_API_KEY"] = "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"
# ---------------------------------

# --- NEW: WhatsApp-style Chat CSS ---
st.markdown("""
<style>
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        max-width: 80%;
        word-wrap: break-word;
    }
    .chat-message.user {
        background-color: #dcf8c6; /* WhatsApp user green */
        align-self: flex-end;
        color: #000;
    }
    .chat-message.assistant {
        background-color: #ffffff; /* WhatsApp assistant white */
        align-self: flex-start;
        border: 1px solid #eee;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

# --- RAG Prompt Template ---
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
3.  Answer in a clear, human-like, and helpful manner.
4.  **Stock Check**: If the query is about a specific medicine, check its 'Stock' status.
5.  **Alternative Suggestion**: If the 'Stock' is 'No', state it is 'out of stock' and *always* suggest the 'Alternative' medicine listed.
6.  **Dosage**: If the medicine is available, provide the 'Dosage_Instruction'.
7.  **FAQ**: If the query is a general question (e.g., "medicine for fever"), list medicines from the CONTEXT whose 'Use_Case' matches.
8.  If the context does not contain an answer, state that you do not have that information.
"""

# --- General AI Prompt Template ---
GENERAL_PROMPT_TEMPLATE = """
You are a helpful general AI assistant. Answer the user's question concisely.
User: {question}
Assistant:
"""

# --- Helper Functions (Cached) ---

def format_docs(docs):
    return "\n\n".join(
        f"Record {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)
    )

@st.cache_resource
def get_llm():
    """Initializes the Groq LLM."""
    return ChatGroq(model_name=LLM_MODEL)

@st.cache_resource
def get_rag_chain(_llm):
    """Initializes the RAG chain."""
    try:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(k=3)
        rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        return (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | _llm
            | StrOutputParser()
        )
    except Exception as e:
        st.error(f"Error initializing RAG chain: {e}")
        return None

@st.cache_resource
def get_general_chain(_llm):
    """Initializes the General AI chain."""
    prompt = PromptTemplate.from_template(GENERAL_PROMPT_TEMPLATE)
    return (
        {"question": RunnablePassthrough()}
        | prompt
        | _llm
        | StrOutputParser()
    )

# --- Initialize Chains ---
if os.environ.get("GROQ_API_KEY") == "gsk_YOUR_NEW_API_KEY_GOES_HERE":
    st.error("Please paste your new Groq API key into the app.py file (line 21).")
    st.stop()

if not os.path.exists(DB_PATH):
    st.error(f"Vector database not found at {DB_PATH}. Please run 'ingest_data.py' first.")
    st.stop()

llm = get_llm()
rag_chain = get_rag_chain(llm)
general_chain = get_general_chain(llm)

if rag_chain is None:
    st.stop()

# --- Initialize Session State ---
if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = [{"role": "assistant", "content": "Welcome! How can I help you with your medicine query today?"}]
if "general_messages" not in st.session_state:
    st.session_state.general_messages = [{"role": "assistant", "content": "I am a general AI assistant. You can ask me anything!"}]

# --- Main App Layout (Two Columns) ---
col1, col2 = st.columns(2)

# --- Column 1: Medical RAG Assistant ---
with col1:
    st.header("🏥 Medical RAG Assistant")
    st.caption(f"Powered by {LLM_MODEL} & Local Database")

    # Display chat history
    with st.container(height=600):
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.rag_messages:
            st.markdown(f'<div class="chat-message {msg["role"]}">{msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    if user_query_rag := st.chat_input("Ask about medicines...", key="rag_input"):
        # Add user message
        st.session_state.rag_messages.append({"role": "user", "content": user_query_rag})

        # Get assistant response
        with st.spinner("Searching database..."):
            try:
                response = rag_chain.invoke(user_query_rag)
                st.session_state.rag_messages.append({"role": "assistant", "content": response})
                st.rerun() # Rerun to display new messages
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Column 2: General AI Assistant ---
with col2:
    st.header("🤖 General AI Assistant")
    st.caption(f"Powered by {LLM_MODEL}")

    # Display chat history
    with st.container(height=600):
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.general_messages:
            st.markdown(f'<div class="chat-message {msg["role"]}">{msg["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    if user_query_general := st.chat_input("Ask me anything...", key="general_input"):
        # Add user message
        st.session_state.general_messages.append({"role": "user", "content": user_query_general})

        # Get assistant response
        with st.spinner("Thinking..."):
            try:
                response = general_chain.invoke(user_query_general)
                st.session_state.general_messages.append({"role": "assistant", "content": response})
                st.rerun() # Rerun to display new messages
            except Exception as e:
                st.error(f"An error occurred: {e}")
