# Generation Engine

The **Generation Engine** encapsulates local Small Language Model (SLM) provider connections and execution.

---

## 📁 File Structure

- `generator.py`: `GenerationEngine` interface class.
- `providers/base.py`: Abstract provider interface definition.
- `providers/ollama_provider.py`: HTTP client connector to local Ollama API (`http://localhost:11434`).

---

## ⚙️ Model Provider

- Default Model: `qwen2.5:3b`
- Temperature: `0.0` (Deterministic factual generation)
