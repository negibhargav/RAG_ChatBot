import os
# pyrefly: ignore [missing-import]
import docx
from .utils import normalize_text

def extract_docx(file_path):
    """Extracts and returns normalized text from a Word document (.docx)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    doc = docx.Document(file_path)
    text_content = []
    
    # Extract text from paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            text_content.append(para.text.strip())
            
    # Extract text from tables
    for table in doc.tables:
        for row in table.rows:
            # Join cells in a row with a separator
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            # Filter unique non-empty cells
            if row_text:
                text_content.append(" | ".join(row_text))
                
    raw_text = "\n".join(text_content)
    return normalize_text(raw_text)
