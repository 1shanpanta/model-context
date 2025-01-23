# image_processing.py
import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import io
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import GOOGLE_API_KEY, GEMINI_FLASH_MODEL_NAME
from google.api_core import exceptions

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini Flash model
model = genai.GenerativeModel(GEMINI_FLASH_MODEL_NAME)

# Retry configuration with timeout handling
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((exceptions.ResourceExhausted, exceptions.DeadlineExceeded)),
)
def safe_generate_content(prompt, image):
    """Wrapper with timeout and retry logic"""
    return model.generate_content(
        contents=prompt,
        request_options={"timeout": 30},  # 30 seconds timeout
    )

def pdf_page_to_image(pdf_path: str, page_num: int, zoom: int = 2):
    """
    Convert a PDF page to an image.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    mat = fitz.Matrix(zoom, zoom)  # Zoom factor for higher resolution
    pix = page.get_pixmap(matrix=mat)
    return Image.open(io.BytesIO(pix.tobytes()))  # Return as PIL Image

def extract_text_from_pdf(pdf_path: str):
    """
    Extract text from a PDF by treating each page as an image.
    """
    doc = fitz.open(pdf_path)
    extracted_texts = []
    
    for page_num in range(len(doc)):
        try:
            # Convert the page to an image
            pil_image = pdf_page_to_image(pdf_path, page_num)
            
            # Extract text with retry and timeout
            response = safe_generate_content(
                ["Extract the Nepali text from this image and don't generate any language other than Nepali in the output:", pil_image],
                pil_image
            )
            
            extracted_texts.append(response.text)
            
            # Rate limit protection
            time.sleep(1)  # Add delay between pages
            
        except Exception as e:
            print(f"Error processing page {page_num + 1}: {str(e)}")
            extracted_texts.append("")  # Add empty string for failed pages
    
    return extracted_texts