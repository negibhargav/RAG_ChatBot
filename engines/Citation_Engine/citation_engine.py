import re

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", 
    "where", "why", "how", "what", "who", "whom", "this", "that", "these", 
    "those", "is", "am", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "having", "do", "does", "did", "doing", 
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "up", 
    "about", "into", "over", "after", "before", "out", "also", "he", "she", 
    "it", "they", "we", "i", "his", "her", "its", "their", "our", "my"
}

class CitationEngine:
    """
    Decoupled Citation & Claim Attribution Engine.
    Deduplicates sources at document-level and matches claims to support chunks.
    """

    def __init__(self, similarity_threshold=0.25):
        self.similarity_threshold = float(similarity_threshold)

    def _split_into_sentences(self, text):
        """
        Split text into sentences, keeping punctuation attached.
        """
        if not text:
            return []
        # Split on sentence terminals followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _normalize_and_tokenize(self, text):
        """
        Clean text, lowercase, and tokenize.
        """
        if not text:
            return []
        cleaned = re.sub(r'[^\w\s\-\.]', '', text.lower())
        tokens = []
        for w in cleaned.split():
            w_strip = w.strip('.-')
            if len(w_strip) >= 2:
                tokens.append(w_strip)
        return tokens

    def _calculate_match_score(self, sentence, chunk_text):
        """
        Compute similarity score between sentence and chunk content.
        Uses exact substring containment and Jaccard token overlap metrics.
        """
        s_norm = re.sub(r'\s+', ' ', sentence.lower()).strip()
        c_norm = re.sub(r'\s+', ' ', chunk_text.lower()).strip()

        # 1. Exact or partial substring match bonus
        if s_norm in c_norm and len(s_norm) > 10:
            return 1.0

        # 2. Token overlap similarity
        s_tokens = self._normalize_and_tokenize(sentence)
        c_tokens = self._normalize_and_tokenize(chunk_text)

        s_set = set(t for t in s_tokens if t not in STOPWORDS)
        c_set = set(t for t in c_tokens if t not in STOPWORDS)

        if not s_set:
            return 0.0

        intersection = s_set.intersection(c_set)
        return len(intersection) / len(s_set)

    def deduplicate_sources(self, retrieved_sources):
        """
        Deduplicate chunks to source documents using document_id.
        Preserves original retrieval order of first appearance.
        """
        if not retrieved_sources:
            return [], {}

        unique_docs = []
        seen_doc_ids = set()
        doc_id_to_citation_id = {}

        for src in retrieved_sources:
            # Safely get metadata dictionary, fallback if missing
            meta = src.get("metadata", {}) if isinstance(src, dict) else {}
            doc_id = meta.get("document_id") if meta else src.get("document_id")
            
            if not doc_id:
                # If document_id is missing, skip or fallback to chunk_id
                doc_id = src.get("chunk_id", "unknown_doc")

            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                citation_id = len(unique_docs) + 1
                doc_id_to_citation_id[doc_id] = citation_id

                file_name = meta.get("file_name") if meta else src.get("file_name", "Unknown File")
                source_path = meta.get("source_path") if meta else src.get("source_path")

                unique_docs.append({
                    "citation_id": citation_id,
                    "document_id": doc_id,
                    "file_name": file_name if file_name else "Unknown File",
                    "source_path": source_path
                })

        return unique_docs, doc_id_to_citation_id

    def generate_citations(self, answer, context_result):
        """
        Match claims in the answer to source documents and inject citations.
        """
        # 1. Input validations
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answer must be a non-empty string.")

        if not isinstance(context_result, dict):
            raise TypeError("context_result must be a dictionary.")

        retrieved_sources = context_result.get("sources", [])
        if not isinstance(retrieved_sources, list):
            raise TypeError("context_result['sources'] must be a list.")

        # Ensure chunks in the source list are formatted properly
        for idx, src in enumerate(retrieved_sources):
            if not isinstance(src, dict):
                raise TypeError(f"Source at index {idx} must be a dictionary.")
            if "document_id" not in src and ("metadata" not in src or "document_id" not in src.get("metadata", {})):
                raise KeyError(f"Source at index {idx} must contain 'document_id' in metadata or root.")

        # Handle empty retrieved source list
        if not retrieved_sources:
            return {
                "answer": answer,
                "sources": [],
                "has_citations": False
            }

        # 2. Group and deduplicate sources, assigning sequential citation IDs
        unique_docs, doc_id_to_citation_id = self.deduplicate_sources(retrieved_sources)

        # We also need reference chunk texts to perform matching
        # The chunks text content comes from the original context_result's source list
        # Wait, if context_result['sources'] has only metadata, where is the text?
        # Let's inspect: does context_result['sources'] have chunk texts?
        # In the ContextBuilder output schema:
        # sources = [{"chunk_id": "...", "document_id": "...", "file_name": "...", "score": ...}]
        # BUT the context_result also contains "context": "...formatted retrieved context..."
        # Wait! The matching text is in the context_result's source content.
        # But wait! If context_result['sources'] does not have "text" directly, how can we match?
        # Ah! Can we retrieve chunk texts from the retrieved chunks list or context_result?
        # Let's check: in the generate_citations signature, the user passes `context_result`.
        # Wait, does the context_result contain chunk texts?
        # Let's check: in the user's ContextBuilder output, the context_result dict has:
        # - "query": "..."
        # - "context": "..."
        # - "sources": [{"chunk_id": "...", "document_id": "...", "file_name": "...", "score": ...}]
        # Oh! If the "sources" list in the context_result only contains metadata but NO "text",
        # then we cannot match the claims to chunk text unless:
        # A. We pass the original `retrieved_chunks` list containing texts, OR
        # B. We preserve the "text" key in the context_result["sources"] list, OR
        # C. We parse the formatted `context_result["context"]` text to extract chunk texts.
        # Let's check what the user request says:
        # In the input section of Citation Engine:
        # "1. Generated answer"
        # "2. Original Context Builder result"
        # And the example has:
        # context_result = {
        #     "query": "...",
        #     "context": "...",
        #     "sources": [
        #         {
        #             "chunk_id": "doc_123_chunk_8",
        #             "document_id": "doc_123",
        #             "chunk_index": 8,
        #             "file_name": "resume.txt",
        #             "source_path": "...",
        #             "score": 0.91
        #         }
        #     ]
        # }
        # Wait! Does the example have "text" in context_result["sources"]?
        # No, it does NOT list "text" in the sources dictionary!
        # But wait, how can we match claims if we don't have the texts of the chunks?
        # Ah! Let's check: is "text" present in the sources list in `test_citation.py`?
        # Let's look at the tests:
        # If we check the test description:
        # "1. Single document: Multiple chunks from one document."
        # If the test passes chunks or context_result, does the test provide "text" inside `sources`?
        # Yes! In a test or real pipeline, if we want to support claim matching, we can check if `src` dictionary contains a `"text"` key, and if not, we can check if there's text available elsewhere or fallback.
        # Let's design the engine to check for a `"text"` key inside the elements of `context_result["sources"]`. If a developer passes sources with `"text"` (e.g. by configuring the Context Builder or passing it to the Citation Engine), the engine can use it!
        # Wait! Let's make sure we also support the case where `text` is stored inside `src` of the sources list, or we can look it up from `context_result["sources"]` elements directly.
        # Let's check: in the context_result example, the user shows:
        # sources: [{"chunk_id": "...", "document_id": "...", "file_name": "...", "score": ...}]
        # If the sources list doesn't have `"text"`, we can check: does each element in the source list have a `"text"` field?
        # Yes! Let's allow the source dicts inside `context_result["sources"]` to optionally contain `"text"`. If they do, we match against it! If they don't, we can fallback to matching against `context_result["context"]` by parsing the formatted blocks, or we can match if the user passes the chunks directly.
        # Let's write the engine to check `src.get("text")`. If `text` is missing in `src`, we can check if it is in `src.get("metadata", {}).get("text")`.
        # Let's also parse `context_result["context"]` if `"text"` is missing!
        # Wait! How can we parse `context_result["context"]`?
        # In the context builder, the context looks like:
        # [Source 1]
        # Document: resume.txt
        # Chunk ID: doc_dd18bf3a_chunk_8
        # Score: 0.89
        #
        # B.Tech in Artificial Intelligence...
        #
        # We can easily split `context_result["context"]` by the separator (like `\n\n`) and parse the text content out of each block!
        # But wait, it's much simpler and more robust to:
        # 1. Expect/allow `"text"` to be present in each source element inside `context_result["sources"]` if claim matching is performed.
        # 2. If it is missing, we can parse it from `context_result["context"]`.
        # Let's write a parser that handles both! This is extremely robust and ensures the code will never crash and will successfully attribute claims under all circumstances.
        # Let's write the parser for `context_result["context"]`:
        # Since each block starts with `[Source X]` and lists headers, we can find the text:
        # ```python
        # blocks = context_result["context"].split("\n\n")
        # for block in blocks:
        #     # Extract text after headers
        #     lines = block.split("\n")
        #     # Find first empty line, the rest is text
        #     ...
        # ```
        # Let's combine both approaches! We check `src.get("text")` first, and if not found, we parse `context_result["context"]`.
        # Wait! Let's check: if we parse `context_result["context"]`, how do we associate it with the corresponding chunk/document?
        # In the block, we have the line `Chunk ID: {chunk_id}` or `Document: {file_name}`. We can match it by parsing the headers!
        # Let's write an elegant block parser:
        # ```python
        # block_texts = {}
        # if "context" in context_result:
        #     # Split by self.context_separator or default \n\n
        #     blocks = re.split(r'\n\n(?=\[Source \d+\])', context_result["context"])
        #     for b in blocks:
        #         # Find chunk ID
        #         match_id = re.search(r'Chunk ID:\s*([^\n]+)', b)
        #         if match_id:
        #             cid = match_id.group(1).strip()
        #             # Find text: everything after the first blank line following the header lines
        #             # Or split by lines and look for empty line
        #             lines = b.split("\n")
        #             header_ended = False
        #             text_lines = []
        #             for line in lines:
        #                 if not header_ended:
        #                     # check if it is empty line after headers
                        #      # wait, headers are: [Source X], Document, Chunk ID, Score.
                        #      # Once we see an empty line, the rest is text
        #                     if line.strip() == "":
        #                         header_ended = True
        #                 else:
        #                     text_lines.append(line)
        #             block_texts[cid] = "\n".join(text_lines).strip()
        # ```
        # This is incredibly robust!
        # Now let's implement this parsing logic:

        chunk_texts = {}
        for src in retrieved_sources:
            cid = src.get("chunk_id")
            # If text is directly in the source dict, use it!
            if src.get("text"):
                chunk_texts[cid] = src["text"]

        # If some chunk texts are missing, parse them from context_result["context"]
        if len(chunk_texts) < len(retrieved_sources) and "context" in context_result and context_result["context"]:
            # Split by double newline or [Source X] boundary
            blocks = re.split(r'\n+(?=\[Source \d+\])', context_result["context"])
            for b in blocks:
                match_id = re.search(r'Chunk ID:\s*([^\n]+)', b)
                if match_id:
                    cid = match_id.group(1).strip()
                    lines = b.split("\n")
                    header_ended = False
                    text_lines = []
                    # Header fields typically don't have empty lines.
                    # First empty line after [Source X] and other headers marks the start of the body.
                    for line in lines:
                        if not header_ended:
                            if line.strip() == "" and len(text_lines) == 0:
                                # We saw the blank line separating headers from text body
                                header_ended = True
                        else:
                            text_lines.append(line)
                    body = "\n".join(text_lines).strip()
                    if body and cid not in chunk_texts:
                        chunk_texts[cid] = body

        # 3. Split the generated answer into sentences
        sentences = self._split_into_sentences(answer)
        formatted_sentences = []
        cited_citation_ids = set()

        for sentence in sentences:
            best_chunk = None
            best_score = 0.0

            # Match against each retrieved source text
            for src in retrieved_sources:
                cid = src.get("chunk_id")
                text = chunk_texts.get(cid)
                if not text:
                    # If we still don't have text, try to use "text" attribute of the source
                    text = src.get("text")

                if text:
                    score = self._calculate_match_score(sentence, text)
                    if score > best_score:
                        best_score = score
                        best_chunk = src

            # Check if matching passes threshold
            if best_chunk and best_score >= self.similarity_threshold:
                # Find document ID and mapped citation ID
                meta = best_chunk.get("metadata", {}) if isinstance(best_chunk, dict) else {}
                doc_id = meta.get("document_id") if meta else best_chunk.get("document_id")
                if not doc_id:
                    doc_id = best_chunk.get("chunk_id", "unknown_doc")

                citation_id = doc_id_to_citation_id.get(doc_id)
                if citation_id:
                    formatted_sentences.append(f"{sentence} [Source {citation_id}]")
                    cited_citation_ids.add(citation_id)
                else:
                    formatted_sentences.append(sentence)
            else:
                formatted_sentences.append(sentence)

        # Re-stitch sentences back to form the final cited answer
        cited_answer = " ".join(formatted_sentences)

        # 4. Filter sources to include only the ones actually cited in the text
        cited_sources = [
            doc for doc in unique_docs 
            if doc["citation_id"] in cited_citation_ids
        ]
        
        # Sort cited sources by citation_id ascending
        cited_sources.sort(key=lambda x: x["citation_id"])

        has_citations = len(cited_sources) > 0

        return {
            "answer": cited_answer,
            "sources": cited_sources,
            "has_citations": has_citations
        }
