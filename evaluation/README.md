# Evaluation Runner & Reports

This directory contains the dataset definitions, evaluation test execution scripts, and generated benchmark reports.

---

## 📁 File Structure

- `datasets/rag_test_cases.json`: Golden test suite containing questions, expected document sources, answerability flags, and reference answers.
- `run_evaluation.py`: Full evaluation pipeline orchestrator executing Phases 1 through 5.
- `reports/`: Target output directory where text summary reports (`.txt`) and JSON raw metrics (`.json`) are automatically saved after each run.

---

## 🚀 How to Run Evaluation

```bash
python evaluation/run_evaluation.py
```
