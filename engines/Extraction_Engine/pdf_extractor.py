import os 
# pyrefly: ignore [missing-import]
from pypdf import PdfReader

def extract_pdf(file_path):
    """Extracts and returns text from a PDF file (.pdf)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        
        # Validation: Check if PDF is encrypted/password-protected
        if reader.is_encrypted:
            raise ValueError("PDF is encrypted/password-protected and cannot be read.")
            
        text_content = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                text_content.append(text.strip())
                
        extracted_text = "\n".join(text_content).strip()
        
        # Validation: Check if text is empty (scanned PDF check)
        if not extracted_text:
            if len(reader.pages) > 0:
                raise ValueError("No extractable text found. The PDF might contain only scanned images.")
            else:
                raise ValueError("The PDF has 0 pages.")
                
        return extracted_text
