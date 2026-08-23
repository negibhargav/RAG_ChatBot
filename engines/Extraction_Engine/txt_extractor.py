import os
from .utils import normalize_text

def extract_text(file_path):
    """Reads and returns normalized text from a text file (.txt)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_text = f.read()
        
    return normalize_text(raw_text)
