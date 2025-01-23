# image_processing.py
import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import io
import time
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import GOOGLE_API_KEY, GEMINI_FLASH_MODEL_NAME
from google.api_core import exceptions

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini Flash model
model = genai.GenerativeModel(GEMINI_FLASH_MODEL_NAME)

# Retry configuration with timeout handling (PRESERVED)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((exceptions.ResourceExhausted, exceptions.DeadlineExceeded)),
)
def safe_generate_content(prompt, image):
    """Wrapper with timeout and retry logic (PRESERVED)"""
    return model.generate_content(
        contents=prompt,
        request_options={"timeout": 30},
    )

def pdf_page_to_image(pdf_path: str, page_num: int, zoom: int = 2):
    """Convert PDF page to image (PRESERVED)"""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return Image.open(io.BytesIO(pix.tobytes()))

def extract_text_from_pdf(pdf_path: str):
    """Full version with Nepali validation ADDED (NOT REMOVED)"""
    doc = fitz.open(pdf_path)
    extracted_texts = []
    nepali_pattern = re.compile(r'[\u0900-\u097F\s]+') 
    
    for page_num in range(len(doc)):
        try:
            # PRESERVED image conversion
            pil_image = pdf_page_to_image(pdf_path, page_num)
            
            # PRESERVED retry logic with enhanced prompt
            response = safe_generate_content(
                [
                    "Extract only the visible Nepali text from this image exactly as it appears.",
                    "DO NOT: Add translations/explanations/notes",  
                    "DO NOT: Include any English text",  
                    "Output format: Raw Nepali text only",
                    pil_image
                ],
                pil_image
            )
            
           
            raw_text = response.text
            cleaned_text = ''.join(nepali_pattern.findall(raw_text)).strip()
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            
            
            if len(cleaned_text) > 10:
                extracted_texts.append(cleaned_text)
                print(f"Page {page_num+1}: Valid text")
            else:
                extracted_texts.append("")
                print(f"Page {page_num+1}: No text")
            
           
            time.sleep(1)
            
        except Exception as e:
            # PRESERVED error handling
            print(f"Error processing page {page_num + 1}: {str(e)}")
            extracted_texts.append("")
    
    return extracted_texts