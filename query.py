# query.py
import google.generativeai as genai
from config import GOOGLE_API_KEY, GEMINI_FLASH_MODEL_NAME, GEMINI_GENERATION_CONFIG
from vector_store import get_vector_store

# Initialize Gemini Pro
genai.configure(api_key=GOOGLE_API_KEY)
generation_model = genai.GenerativeModel(
    GEMINI_FLASH_MODEL_NAME,
    generation_config=GEMINI_GENERATION_CONFIG
)

RAG_PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks in Nepali. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say you don't know. Keep the answer concise and in Nepali.

Question: {question}

Context: {context}

Answer in Nepali:
"""

def rag_query(question: str, k: int = 3):
    """
    Perform RAG query: retrieve relevant docs + generate answer.
    """
    # 1. Retrieve relevant documents
    vector_store = get_vector_store()
    relevant_docs = vector_store.similarity_search(question, k=k)
    
    # 2. Prepare context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # 3. Generate answer
    prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)
    response = generation_model.generate_content(prompt)
    
    return {
        "answer": response.text,
        "relevant_docs": relevant_docs
    }