import re
import unicodedata

def normalize_text(text):
    """Clean and normalize raw extracted text for ingestion into a RAG pipeline."""
    if not text:
        return ""
        
    # 1. Normalize Unicode characters to canonical composition (NFKC)
    # This standardizes ligatures, accented letters, and special symbols
    text = unicodedata.normalize("NFKC", text)
    
    # 2. Replace smart/curly quotes and dashes with standard ASCII equivalents
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-")
    
    # 3. Stitch hyphenated line endings (e.g., retriev-\n-augmented -> retrieval-augmented)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # 4. Collapse consecutive spaces and tabs to a single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 5. Limit excessive newlines to a maximum of 2 newlines (preserves paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()
