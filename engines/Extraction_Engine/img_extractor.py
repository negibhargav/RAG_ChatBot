import os
import shutil
from PIL import Image
import pytesseract
from .utils import normalize_text

def extract_image(file_path):
    """Extracts and returns normalized text from an image using Tesseract OCR."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Check if Tesseract executable is found on system PATH
    if not shutil.which("tesseract"):
        # Fallback to standard Windows installation path
        tesseract_win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_win_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_win_path
        else:
            raise RuntimeError(
                "Tesseract-OCR is not installed or not in your system PATH.\n"
                "Please download and install Tesseract-OCR (e.g. from https://github.com/UB-Mannheim/tesseract/wiki)\n"
                "and ensure it is added to your environment variables."
            )
            
    try:
        with Image.open(file_path) as image:
            raw_text = pytesseract.image_to_string(image)
        
        normalized_text = normalize_text(raw_text)
        if not normalized_text:
            raise ValueError("No text could be recognized/extracted from the image.")
            
        return normalized_text
    except Exception as e:
        raise RuntimeError(f"OCR extraction failed: {e}")
