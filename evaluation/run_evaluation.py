import os
import sys
import json
from datetime import datetime

# Reconfigure stdout to UTF-8 to handle Unicode report characters on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Setup load paths for engines
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../engines")))

import Extraction_Engine
import Chunking_Engine
import Embedding_Engine
import Vector_Store_Engine
import Retrieval_Engine
import Context_Engine
import Generation_Engine
from Generation_Engine.providers.ollama_provider import OllamaProvider
from Evaluation_Engine import (
    EvaluationDatasetLoader,
    RetrievalEvaluator,
    EvaluationReportGenerator,
    ChunkingEvaluator,
    AnswerAccuracyEvaluator,
    GroundingEvaluator,
    CitationEvaluator
)


def main():
    print("=" * 60)
    print("   RAG EVALUATION RUNNER -- PHASES 1, 2, 3, 4 & 5")
    print("=" * 60)


    dataset_path = os.path.join("evaluation", "datasets", "rag_test_cases.json")
    print(f"Loading golden dataset: '{dataset_path}'...")
    try:
        test_cases = EvaluationDatasetLoader.load(dataset_path)
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        sys.exit(1)
    print(f"Loaded {len(test_cases)} test cases.")

    source_doc = "test.txt"
    if not os.path.exists(source_doc):
        print(f"[Error] Source file '{source_doc}' not found.")
        sys.exit(1)

    # Initialize shared engines
    print("Initializing Embedding Engine...")
    embedder = Embedding_Engine.EmbeddingEngine()

    db_dir = "Vector_DB"
    collection_name = "rag_evaluation_collection"
    embedding_model = "BAAI/bge-small-en-v1.5"
    generation_model = "qwen2.5:3b"
    chunk_size = 300
    chunk_overlap = 30

    # ============================================================
    # PHASE 1: RETRIEVAL GATE SWEEP
    # ============================================================
    print(f"\n[PHASE 1] Initializing evaluation collection '{collection_name}'...")
    db = Vector_Store_Engine.ChromaVectorStore(db_dir=db_dir, collection_name=collection_name)

    if db.collection.count() == 0:
        print(f"  Empty collection. Ingesting '{source_doc}'...")
        text = Extraction_Engine.extract_text(source_doc)
        doc = {
            "document_id": "doc_test_setup",
            "text": text,
            "metadata": {
                "source_path": os.path.abspath(source_doc),
                "file_name": source_doc,
                "file_type": ".txt"
            }
        }
        chunks = Chunking_Engine.split_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embedded = embedder.embed_chunks(chunks)
        db.save_chunks(embedded)

    retriever = Retrieval_Engine.RetrievalConnector(db)
    phase_1_evaluator = RetrievalEvaluator(retriever, embedder)

    thresholds = [0.40, 0.50, 0.60, 0.70]
    k_values = [1, 3, 5]
    phase_1_experiments = []

    print("Running Phase 1 retrieval sweep...")
    for threshold in thresholds:
        for k in k_values:
            try:
                result = phase_1_evaluator.evaluate(test_cases, top_k=k, score_threshold=threshold)
                phase_1_experiments.append(result)
            except Exception as e:
                print(f"  [Error] K={k}, Gate={threshold}: {e}")

    # ============================================================
    # PHASE 2: CHUNKING CONFIG SWEEP
    # ============================================================
    print("\n[PHASE 2] Running chunking configuration sweep...")
    text = Extraction_Engine.extract_text(source_doc)
    documents = [{
        "document_id": "doc_test_setup",
        "text": text,
        "metadata": {"source_path": os.path.abspath(source_doc), "file_name": source_doc, "file_type": ".txt"}
    }]

    chunk_sizes = [200, 300, 500, 700]
    overlaps = [20, 30, 50]
    phase_2_experiments = []
    chunking_evaluator = ChunkingEvaluator(embedder)

    for size in chunk_sizes:
        for overlap in overlaps:
            if overlap >= size:
                continue
            print(f"  Evaluating chunk_size={size}, overlap={overlap}...")
            try:
                result = chunking_evaluator.evaluate_chunk_config(
                    documents=documents,
                    test_cases=test_cases,
                    chunk_size=size,
                    chunk_overlap=overlap,
                    score_threshold=0.40
                )
                phase_2_experiments.append(result)
            except Exception as e:
                print(f"  [Error] chunk_size={size}, overlap={overlap}: {e}")

    baseline_chunks = next(
        (e["total_chunks"] for e in phase_2_experiments if e["chunk_size"] == 300 and e["chunk_overlap"] == 30),
        25
    )

    # ============================================================
    # PHASE 3: ANSWER ACCURACY EVALUATION
    # ============================================================
    print("\n[PHASE 3] Running Answer Accuracy Evaluation...")

    # Check Ollama availability before starting generation runs
    provider = OllamaProvider()
    health = provider.health_check(generation_model)
    if not health["available"]:
        print(f"\n[PHASE 3] GENERATION_ENGINE_UNAVAILABLE: {health['error']}")
        print("  Skipping Phase 3. Please ensure Ollama is running with qwen2.5:3b pulled.")
        phase_3_result = None
    else:
        gen_engine = Generation_Engine.GenerationEngine(
            provider=provider,
            model=generation_model,
            temperature=0.0
        )
        context_builder = Context_Engine.ContextBuilder()

        # Use evaluation collection to ensure isolation
        prod_retriever = retriever

        accuracy_evaluator = AnswerAccuracyEvaluator(
            embedder=embedder,
            retriever=prod_retriever,
            context_builder=context_builder,
            generation_engine=gen_engine
        )
        try:
            phase_3_result = accuracy_evaluator.evaluate(
                test_cases=test_cases,
                top_k=3,
                score_threshold=0.40
            )
        except RuntimeError as e:
            print(f"\n[PHASE 3] {e}")
            phase_3_result = None

    # ============================================================
    # PHASE 4: GROUNDEDNESS & HALLUCINATION EVALUATION
    # ============================================================
    print("\n[PHASE 4] Running Groundedness & Hallucination Evaluation...")
    phase_4_result = None

    if not health["available"]:
        print(f"\n[PHASE 4] GENERATION_ENGINE_UNAVAILABLE: {health['error']}")
        print("  Skipping Phase 4. Please ensure Ollama is running with qwen2.5:3b pulled.")
    else:
        # Reuse same engine objects from Phase 3 (or rebuild if Phase 3 was skipped)
        if not health["available"]:
            print("  Skipping Phase 4 — Ollama unavailable.")
        else:
            try:
                # Rebuild engines if needed (they are already built above for Phase 3)
                _gen_engine = Generation_Engine.GenerationEngine(
                    provider=OllamaProvider(),
                    model=generation_model,
                    temperature=0.0
                )
                _context_builder = Context_Engine.ContextBuilder()
                # Use evaluation collection to ensure isolation
                _prod_retriever = retriever

                grounding_evaluator = GroundingEvaluator(
                    embedder=embedder,
                    retriever=_prod_retriever,
                    context_builder=_context_builder,
                    generation_engine=_gen_engine
                )
                phase_4_result = grounding_evaluator.evaluate(
                    test_cases=test_cases,
                    top_k=3,
                    score_threshold=0.40
                )
            except RuntimeError as e:
                print(f"\n[PHASE 4] {e}")
                phase_4_result = None

    # ============================================================
    # PHASE 5: SIMPLE CITATION EVALUATION
    # ============================================================
    print("\n[PHASE 5] Running Simple Citation Evaluation...")
    phase_5_result = None

    if not health["available"]:
        print(f"\n[PHASE 5] GENERATION_ENGINE_UNAVAILABLE: {health['error']}")
        print("  Skipping Phase 5. Please ensure Ollama is running.")
    else:
        try:
            # Rebuild engines if needed
            _gen_engine = Generation_Engine.GenerationEngine(
                provider=OllamaProvider(),
                model=generation_model,
                temperature=0.0
            )
            _context_builder = Context_Engine.ContextBuilder()
            # Use evaluation collection to ensure isolation
            _prod_retriever = retriever

            citation_evaluator = CitationEvaluator(
                embedder=embedder,
                retriever=_prod_retriever,
                context_builder=_context_builder,
                generation_engine=_gen_engine
            )
            phase_5_result = citation_evaluator.evaluate(
                test_cases=test_cases,
                top_k=3,
                score_threshold=0.40
            )
        except RuntimeError as e:
            print(f"\n[PHASE 5] {e}")
            phase_5_result = None


    # ============================================================
    # REPORT GENERATION
    # ============================================================
    report_p1 = EvaluationReportGenerator.generate_text_report(
        experiment_data=phase_1_experiments,
        dataset_path=dataset_path,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    report_p2 = EvaluationReportGenerator.generate_chunking_report(
        chunking_data=phase_2_experiments,
        baseline_chunks=baseline_chunks
    )

    full_report = report_p1 + "\n\n" + report_p2

    if phase_3_result:
        report_p3 = EvaluationReportGenerator.generate_answer_accuracy_report(phase_3_result)
        full_report += "\n\n" + report_p3

        timestamp_p3 = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(os.path.join("evaluation", "reports"), exist_ok=True)

        txt_p3 = os.path.join("evaluation", "reports", f"answer_accuracy_{timestamp_p3}.txt")
        with open(txt_p3, "w", encoding="utf-8") as f:
            f.write(report_p3)
        print(f"\nSaved Phase 3 TXT report: '{txt_p3}'")

        json_p3 = os.path.join("evaluation", "reports", f"answer_accuracy_{timestamp_p3}.json")
        with open(json_p3, "w", encoding="utf-8") as f:
            json.dump(phase_3_result, f, indent=2)
        print(f"Saved Phase 3 JSON results: '{json_p3}'")

    if phase_4_result:
        report_p4 = EvaluationReportGenerator.generate_grounding_report(phase_4_result)
        full_report += "\n\n" + report_p4

        timestamp_p4 = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(os.path.join("evaluation", "reports"), exist_ok=True)

        txt_p4 = os.path.join("evaluation", "reports", f"grounding_{timestamp_p4}.txt")
        with open(txt_p4, "w", encoding="utf-8") as f:
            f.write(report_p4)
        print(f"Saved Phase 4 TXT report: '{txt_p4}'")

        json_p4 = os.path.join("evaluation", "reports", f"grounding_{timestamp_p4}.json")
        with open(json_p4, "w", encoding="utf-8") as f:
            json.dump(phase_4_result, f, indent=2)
        print(f"Saved Phase 4 JSON results: '{json_p4}'")

    if phase_5_result:
        report_p5 = EvaluationReportGenerator.generate_citation_report(phase_5_result)
        full_report += "\n\n" + report_p5

        timestamp_p5 = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(os.path.join("evaluation", "reports"), exist_ok=True)

        txt_p5 = os.path.join("evaluation", "reports", f"citation_evaluation_{timestamp_p5}.txt")
        with open(txt_p5, "w", encoding="utf-8") as f:
            f.write(report_p5)
        print(f"Saved Phase 5 TXT report: '{txt_p5}'")

        json_p5 = os.path.join("evaluation", "reports", f"citation_evaluation_{timestamp_p5}.json")
        with open(json_p5, "w", encoding="utf-8") as f:
            json.dump(phase_5_result, f, indent=2)
        print(f"Saved Phase 5 JSON results: '{json_p5}'")

    print("\n" + full_report)


    # Save consolidated report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join("evaluation", "reports"), exist_ok=True)

    txt_path = os.path.join("evaluation", "reports", f"evaluation_report_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"Saved consolidated text report: '{txt_path}'")

    json_path = os.path.join("evaluation", "reports", f"evaluation_results_{timestamp}.json")
    raw_output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dataset_path": dataset_path,
            "embedding_model": embedding_model,
            "generation_model": generation_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "tested_k_values": k_values,
            "tested_thresholds": thresholds,
            "tested_chunk_sizes": chunk_sizes,
            "tested_overlaps": overlaps
        },
        "phase_1_experiments": phase_1_experiments,
        "phase_2_experiments": phase_2_experiments,
        "phase_3_result": phase_3_result,
        "phase_4_result": phase_4_result,
        "phase_5_result": phase_5_result
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(raw_output, f, indent=2)
    print(f"Saved consolidated JSON logs: '{json_path}'")

    print("\nAll evaluation phases complete.")


if __name__ == "__main__":
    main()
