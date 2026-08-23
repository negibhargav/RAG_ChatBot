import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../engines")))

import unittest
import Citation_Engine

class TestCitationEngine(unittest.TestCase):
    def setUp(self):
        self.engine = Citation_Engine.CitationEngine(similarity_threshold=0.25)
        
        # Standard mock data for test contexts
        self.sample_sources = [
            {
                "chunk_id": "doc_123_chunk_8",
                "text": "B.Tech in Artificial Intelligence and Data Science.",
                "document_id": "doc_123",
                "file_name": "resume.txt",
                "source_path": "/path/to/resume.txt"
            },
            {
                "chunk_id": "doc_123_chunk_9",
                "text": "coursework in linear algebra and calculus.",
                "document_id": "doc_123",
                "file_name": "resume.txt",
                "source_path": "/path/to/resume.txt"
            },
            {
                "chunk_id": "doc_456_chunk_2",
                "text": "Founding ML Engineer at Tecraki Tech Solutions.",
                "document_id": "doc_456",
                "file_name": "work_history.docx",
                "source_path": "/path/to/work_history.docx"
            }
        ]
        
        self.context_result = {
            "query": "Educational and work history",
            "context": "Context text...",
            "sources": self.sample_sources
        }

    def test_01_single_document(self):
        """1. Single document: Multiple chunks from one document group into one unique source."""
        chunks = [self.sample_sources[0], self.sample_sources[1]]
        context = {"sources": chunks}
        
        unique_docs, doc_map = self.engine.deduplicate_sources(chunks)
        self.assertEqual(len(unique_docs), 1)
        self.assertEqual(unique_docs[0]["document_id"], "doc_123")
        self.assertEqual(unique_docs[0]["citation_id"], 1)

    def test_02_multiple_documents(self):
        """2. Multiple documents: Verify different documents receive distinct citation IDs."""
        unique_docs, doc_map = self.engine.deduplicate_sources(self.sample_sources)
        
        self.assertEqual(len(unique_docs), 2)
        self.assertEqual(unique_docs[0]["document_id"], "doc_123")
        self.assertEqual(unique_docs[1]["document_id"], "doc_456")

    def test_03_duplicate_source(self):
        """3. Duplicate source: Multiple sequential chunk hits to same doc keep single index."""
        answer = "Bhupender completed a B.Tech in Artificial Intelligence. He did linear algebra and calculus."
        res = self.engine.generate_citations(answer, self.context_result)
        
        # Verify both claims are cited to the same document [Source 1]
        self.assertIn("Artificial Intelligence. [Source 1]", res["answer"])
        self.assertIn("calculus. [Source 1]", res["answer"])
        self.assertEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["citation_id"], 1)

    def test_04_mixed_sources(self):
        """4. Mixed sources: Order of first appearance determines deterministic index."""
        mixed_sources = [
            {"chunk_id": "c1", "document_id": "doc_A", "file_name": "A.txt"},
            {"chunk_id": "c2", "document_id": "doc_B", "file_name": "B.txt"},
            {"chunk_id": "c3", "document_id": "doc_A", "file_name": "A.txt"},
            {"chunk_id": "c4", "document_id": "doc_C", "file_name": "C.txt"}
        ]
        
        unique_docs, doc_map = self.engine.deduplicate_sources(mixed_sources)
        self.assertEqual(len(unique_docs), 3)
        self.assertEqual(unique_docs[0]["document_id"], "doc_A")
        self.assertEqual(unique_docs[0]["citation_id"], 1)
        self.assertEqual(unique_docs[1]["document_id"], "doc_B")
        self.assertEqual(unique_docs[1]["citation_id"], 2)
        self.assertEqual(unique_docs[2]["document_id"], "doc_C")
        self.assertEqual(unique_docs[2]["citation_id"], 3)

    def test_05_citation_numbering(self):
        """5. Citation numbering: Verify citation numbers follow first appearance sequence."""
        unique_docs, doc_map = self.engine.deduplicate_sources(self.sample_sources)
        self.assertEqual(doc_map["doc_123"], 1)
        self.assertEqual(doc_map["doc_456"], 2)

    def test_06_no_sources(self):
        """6. No sources: Verify no fake citations are generated on empty retrieval."""
        res = self.engine.generate_citations("Some answer string", {"sources": []})
        self.assertEqual(res["sources"], [])
        self.assertFalse(res["has_citations"])
        self.assertEqual(res["answer"], "Some answer string")

    def test_07_missing_optional_metadata(self):
        """7. Missing optional metadata: Works correctly when optional fields are empty."""
        stripped_source = [
            {
                "chunk_id": "chunk_99",
                "document_id": "doc_99"
                # file_name and source_path are missing
            }
        ]
        unique_docs, doc_map = self.engine.deduplicate_sources(stripped_source)
        self.assertEqual(len(unique_docs), 1)
        self.assertEqual(unique_docs[0]["file_name"], "Unknown File")
        self.assertIsNone(unique_docs[0]["source_path"])

    def test_08_multiple_claims(self):
        """8. Multiple claims: Different claims map to different documents."""
        answer = "He completed a B.Tech in Artificial Intelligence. He worked as a Founding ML Engineer at Tecraki."
        res = self.engine.generate_citations(answer, self.context_result)
        
        self.assertIn("Artificial Intelligence. [Source 1]", res["answer"])
        self.assertIn("ML Engineer at Tecraki. [Source 2]", res["answer"])
        self.assertEqual(len(res["sources"]), 2)

    def test_09_reused_citations(self):
        """9. Reused citations: Reuses same source citation index when claims originate from same doc."""
        answer = "Linear algebra coursework. Artificial Intelligence background."
        res = self.engine.generate_citations(answer, self.context_result)
        
        self.assertIn("coursework. [Source 1]", res["answer"])
        self.assertIn("background. [Source 1]", res["answer"])
        # Only doc_123 is cited, doc_456 is not
        self.assertEqual(len(res["sources"]), 1)
        self.assertEqual(res["sources"][0]["document_id"], "doc_123")

    def test_10_no_hallucinated_source_ids(self):
        """10. No hallucinated source IDs: Every citation ID appearing in answer is present in sources list."""
        answer = "Bhupender completed a B.Tech in Artificial Intelligence. Unrelated sentence with no supporting chunk."
        res = self.engine.generate_citations(answer, self.context_result)
        
        # Verify citation matches [Source 1]
        self.assertIn("[Source 1]", res["answer"])
        self.assertNotIn("[Source 2]", res["answer"])
        
        cited_ids = [s["citation_id"] for s in res["sources"]]
        self.assertIn(1, cited_ids)
        self.assertNotIn(2, cited_ids)
        self.assertEqual(len(cited_ids), 1)

if __name__ == "__main__":
    unittest.main()
