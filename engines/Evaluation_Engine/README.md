# Evaluation Engine

The **Evaluation Engine** contains the benchmark metrics and automated evaluation scripts for testing RAG quality offline.

---

## 📁 File Structure

- `dataset_loader.py`: Loads golden test dataset JSON files.
- `retrieval_evaluator.py`: Phase 1 — Evaluates retrieval recall, precision, and gate thresholds across K values.
- `chunking_evaluator.py`: Phase 2 — Sweeps chunk sizes and overlaps to measure chunking efficiency.
- `answer_accuracy_evaluator.py`: Phase 3 — Evaluates factual answer accuracy against reference standard answers.
- `grounding_evaluator.py`: Phase 4 — Measures groundedness and detects model hallucination.
- `citation_evaluator.py`: Phase 5 — Validates citation presence, mapping correctness, and support.
- `metric_calculator.py`: Precision, Recall, F1, and Jaccard distance calculation helper functions.
- `evaluation_report.py`: Generates human-readable text reports and structured JSON result logs.
