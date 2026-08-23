# Extraction Engine

The **Extraction Engine** is responsible for reading raw binary/text files and extracting clean, normalized plain text for downstream chunking and vector storage.

---

## 📁 File Structure

- `txt_extractor.py`: Reads plain text (`.txt`) files with UTF-8 encoding.
- `pdf_extractor.py`: Parses PDF files (`.pdf`) using `pypdf.PdfReader` with stream management.
- `doc_extractor.py`: Extracts text from Microsoft Word documents (`.docx`) using `python-docx`.
- `img_extractor.py`: Performs Optical Character Recognition (OCR) on `.png`, `.jpg`, `.jpeg` images using `PIL` and `pytesseract`.
- `utils.py`: Contains text cleaning utilities (Unicode normalization, whitespace stripping, artifact cleanup).

---

## ⚙️ Workflow

1. File path and file format extension are passed to `extract_text()`.
2. Format router dispatches to the corresponding extractor script.
3. Extracted text is processed through `utils.py` to clean control characters and normalize spacing.
4. Returns clean string output.
