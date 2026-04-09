import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

# --- Configuration ---
# MODIFIED: Updated to use your new 5000-record dataset
DATA_PATH = "customized_dataset_5000_fixed.csv"
DB_PATH = "chroma_db_medical_5k" # New DB path to avoid conflicts
EMBED_MODEL = "nomic-embed-text"
# ---------------------

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        print(f"Please make sure '{DATA_PATH}' is in the same directory.")
        return

    print(f"Loading medicine data from {DATA_PATH}...")
    loader = CSVLoader(
        file_path=DATA_PATH, 
        encoding="utf-8",
        csv_args={'delimiter': ','}
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} medicine records.")

    print(f"Initializing embedding model: {EMBED_MODEL}...")
    print("This may take a while. Please ensure Ollama is running with 'nomic-embed-text'.")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    print(f"Creating and persisting vector store at: {DB_PATH}...")
    print(f"This will take some time for {len(documents)} records...")
    db = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=DB_PATH
    )
    
    print("\n--- Ingestion Complete ---")
    print(f"Successfully loaded {len(documents)} records into {DB_PATH}.")
    print("You can now run the 'app.py' file to start the agent.")

if __name__ == "__main__":
    main()
