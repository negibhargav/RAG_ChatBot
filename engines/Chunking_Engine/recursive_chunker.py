import re

class RecursiveCharacterChunker:
    """
    Pure-Python recursive character chunker with native sliding-window overlap,
    minimum chunk size enforcement, and orphan-header prevention.
    """

    def __init__(self, chunk_size=500, chunk_overlap=50, min_chunk_size=100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if min_chunk_size >= chunk_size:
            raise ValueError("min_chunk_size must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        self.separators = [
            "\n\n",   # paragraph
            "\n",     # line
            ". ",     # sentence-like
            "? ",     # sentence-like
            "! ",     # sentence-like
            " ",      # word
            ""        # character-level fallback
        ]

    def split_text(self, text):
        if not text:
            return []

        text = text.strip()
        if not text:
            return []

        chunks = self._split(text, self.separators)
        return [chunk for chunk in chunks if chunk.strip()]

    def _split(self, text, separators):
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._hard_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return self._hard_split(text)

        escaped_separator = re.escape(separator)
        splits = re.split(f"({escaped_separator})", text)

        items = []
        for i in range(0, len(splits), 2):
            text_part = splits[i]
            sep_part = splits[i+1] if i + 1 < len(splits) else ""
            if text_part or sep_part:
                items.append(text_part + sep_part)

        chunks = []
        i = 0
        last_j = 0 
        
        while i < len(items):
            current_chunk_parts = []
            current_len = 0
            
            j = i
            while j < len(items):
                item = items[j]
                item_len = len(item)
                
                # Header Look-Ahead Logic
                is_header_like = item_len < 60 and ("\n" in item)
                
                if is_header_like and (current_len >= self.min_chunk_size):
                    next_item_len = len(items[j+1]) if j + 1 < len(items) else 0
                    if current_len + item_len + next_item_len > self.chunk_size:
                        break

                if current_len + item_len <= self.chunk_size:
                    current_chunk_parts.append(item)
                    current_len += item_len
                    j += 1
                else:
                    if current_len == 0:
                        recursive_chunks = self._split(item, remaining_separators)
                        chunks.extend(recursive_chunks)
                        last_j = j + 1
                        j += 1
                    break
            
            if current_chunk_parts:
                if j > last_j:
                    is_end_of_doc = j >= len(items)
                    
                    if current_len >= self.min_chunk_size or is_end_of_doc:
                        chunks.append("".join(current_chunk_parts))
                        last_j = j
                    else:
                        chunks.append("".join(current_chunk_parts))
                        last_j = j
                else:
                    i = j
                    continue
                
            if j >= len(items):
                break
                
            if j > i:
                overlap_len = 0
                backtrack_count = 0
                for k in range(j - 1, i - 1, -1):
                    item_len = len(items[k])
                    if overlap_len + item_len <= self.chunk_overlap:
                        overlap_len += item_len
                        backtrack_count += 1
                    else:
                        break
                
                next_i = j - backtrack_count
                if next_i <= i:
                    next_i = i + 1
                i = next_i
            else:
                i += 1

        return chunks

    def _hard_split(self, text):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end
        return chunks


def split_document(document, chunk_size=500, chunk_overlap=50):
    """
    Split a Document dictionary into Chunk dictionaries with inherited provenance.
    """
    if not isinstance(document, dict):
        raise TypeError("document must be a dictionary")

    document_id = document.get("document_id")
    if not document_id:
        raise ValueError("document must contain a document_id")

    text = document.get("text", "")
    doc_metadata = document.get("metadata", {})

    if not isinstance(doc_metadata, dict):
        raise TypeError("document metadata must be a dictionary")

    # Create chunker with default min_chunk_size logic
    chunker = RecursiveCharacterChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=int(chunk_size * 0.2) # Defaults to 20% of chunk_size
    )

    # Split document text
    text_chunks = chunker.split_text(text)

    chunks = []
    for chunk_index, chunk_text in enumerate(text_chunks):
        # Copy parent metadata
        chunk_metadata = dict(doc_metadata)

        # Set chunk-level indexing and parent mapping
        chunk_metadata["document_id"] = document_id
        chunk_metadata["chunk_index"] = chunk_index

        chunks.append({
            "chunk_id": f"{document_id}_chunk_{chunk_index}",
            "text": chunk_text,
            "metadata": chunk_metadata
        })

    return chunks