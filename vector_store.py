# vector_store.py
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
from embeddings import embedding_function  # Use the embedding function
from config import PINECONE_API_KEY, PINECONE_INDEX
from langchain.schema import Document
from datetime import datetime

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Global variable to store the vector store instance
vector_store_instance = None

def get_vector_store():
    """
    Initialize and return the Pinecone vector store instance.
    """
    global vector_store_instance
    if vector_store_instance is None:
        try:
            # Initialize the vector store
            vector_store_instance = PineconeStore.from_existing_index(
                embedding=embedding_function,  # Use the embedding function
                index_name=PINECONE_INDEX,  # Pass the index name as a string
            )
            print("Vector store initialized successfully")
        except Exception as e:
            print(f"Failed to initialize vector store: {e}")
            raise e
    return vector_store_instance

def add_text_to_vector_store(text: str, metadata: dict = None):
    """
    Add text to the vector store.
    """
    if metadata is None:
        metadata = {}

    # Create a document
    doc = Document(
        page_content=text,
        metadata=metadata,
    )

    # Add to vector store
    vector_store = get_vector_store()
    vector_store.add_documents([doc])

    print(f"Added text to the vector store: {text[:5000]}...")
    return doc

def get_pinecone_index():
    return pc.Index(PINECONE_INDEX)