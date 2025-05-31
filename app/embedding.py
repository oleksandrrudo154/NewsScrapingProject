import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "chroma_news"  # Folder for saving the vector database

# Load environment variables from a .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment. Add it to .env file.")

# Initialize the embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
# Initialize the vector database (Chroma)
vector_db = Chroma(embedding_function=embedding_model, persist_directory=CHROMA_DIR)


def store_in_vector_db(content, metadata):
    """
    Stores the given text and its metadata (e.g., source) in the vector database.
    """
    try:
        vector_db.add_texts([content], metadatas=[metadata])  # Add text to the database
        vector_db.persist()  # Save changes (for persistent storage)
        print(f"Document successfully stored: {metadata}")
    except Exception as e:
        print(f"Error while storing document in vector database: {e}")


def search_vector_db(query, top_k=3):
    """
    Performs semantic search in the database, returning the top K results.
    """
    try:
        results = vector_db.similarity_search(query, k=top_k)  # Semantic search
        return results
    except Exception as e:
        print(f"Error during semantic search: {e}")
        return []
