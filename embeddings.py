# embeddings.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GOOGLE_API_KEY

# Initialize the Google Generative AI embeddings model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Use the correct embedding model name
    task_type="retrieval_document",  # Equivalent to TaskType.RETRIEVAL_DOCUMENT
    google_api_key=GOOGLE_API_KEY,
)

def get_embeddings(text: str):
    """
    Get embeddings for a given text using the Google Generative AI embedding model.
    """
    try:
        # Generate embeddings
        embeddings = embedding_model.embed_query(text)
        return embeddings
    except Exception as e:
        print(f"Failed to generate embeddings: {e}")
        raise e

# Create a callable embedding function
def embedding_function(text: str):
    """
    A callable function that takes a text input and returns its embedding.
    """
    return get_embeddings(text)