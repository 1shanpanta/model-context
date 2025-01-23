# rag_query.py
import google.generativeai as genai
from flask import Flask, request, jsonify
from langchain_community.vectorstores import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import GOOGLE_API_KEY, PINECONE_INDEX, GEMINI_FLASH_MODEL_NAME

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Disable ASCII encoding for JSON responses


class RAGQuerySystem:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            task_type="retrieval_document"
        )
        self.vector_store = Pinecone.from_existing_index(
            index_name=PINECONE_INDEX,
            embedding=self.embeddings
        )
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_FLASH_MODEL_NAME)
        self.rag_prompt = """
        नेपालीमा उत्तर दिनुहोस्। प्रश्न: {question}
        संदर्भ: {context}
        उत्तर:
        """

    def ask(self, question: str, k: int = 3):
        try:
            docs = self.vector_store.similarity_search(question, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            response = self.model.generate_content(
                self.rag_prompt.format(question=question, context=context)
            )
            return {
                "answer": response.text,
                "sources": [doc.metadata for doc in docs]
            }
        except Exception as e:
            return {"error": str(e)}

# Initialize RAG system
rag_system = RAGQuerySystem()@app.route('/ask', methods=['POST'])
def handle_query():
    if not request.json or 'question' not in request.json:
        return jsonify({"त्रुटि": "प्रश्न नभएको"}), 400
    
    question = request.json['question']
    result = rag_system.ask(question)
    
    if 'error' in result:
        return jsonify({"प्रश्न": question, "त्रुटि": result["error"]}), 500
    
    response_data = {
        "प्रश्न": question,
        "उत्तर": result["answer"],
        "स्रोतहरू": result["sources"]
    }
    
    # Debugging: Print the response data
    print("Response Data:", response_data)
    
    response = jsonify(response_data)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response

app.run(host='127.0.0.1', port=5000)