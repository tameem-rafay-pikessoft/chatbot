import os
from langchain_community.vectorstores import Chroma

PERSIST_DIRECTORY = "chroma_db"  # Local directory to store Chroma DB


def create_chroma_connection(embedding_function=None):
    """Create a connection to the Chroma vector store."""
    # Create persist directory if it doesn't exist
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    # Initialize Chroma
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_function
    )
