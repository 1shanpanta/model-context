# main.py
from image_processing import extract_text_from_pdf
from vector_store import add_text_to_vector_store
from datetime  import datetime
from query import rag_query


# Example usage
if __name__ == "__main__":
    # Step 1: Extract text from PDF by treating each page as an image
    pdf_path = "documents.pdf"
    extracted_texts = extract_text_from_pdf(pdf_path)

    # Step 2: Add extracted text to the vector store
    for i, text in enumerate(extracted_texts):
        add_text_to_vector_store(
            text=text,
            metadata={
                "source": pdf_path,
                "page": i + 1,
                "createdAt": datetime.now().isoformat(),
            },
        )
        print(f"Added text from page {i + 1} to the vector store.")

    print("Text extracted from PDF and added to the vector store.")
    question = "मोही लगत कट्टाको सिफारिस गर्न कुन कागजातहरु चाहिन्छ?"
    result = rag_query(question)
    
    print("\nQuestion:", question)
    print("Answer:", result["answer"])
    print("\nRelevant Documents:")
    for doc in result["relevant_docs"]:
        print(f"- {doc.page_content[:10000]}...")