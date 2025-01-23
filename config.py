import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Google Generative AI configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_FLASH_MODEL_NAME= "gemini-2.0-flash-exp"
GEMINI_GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
} 

