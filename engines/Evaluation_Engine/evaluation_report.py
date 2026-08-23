class EvaluationReportGenerator:
    """
    Formatter class to generate human-readable evaluation reports for RAG Retrieval & Chunking.
    """
    @staticmethod
    def generate_text_report(
        experiment_data: list, 
        dataset_path: str = "rag_test_cases.json",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 300,
        chunk_overlap: int = 30
    ) -> str:
        """
        Generates a detailed text-based report for Phase 1 (Retrieval runs).
        """
        if not experiment_data:
            return "No experiment data to report."

        first_run = experiment_data[0]
        summary = first_run["summary"]
        total_queries = summary["total_queries"]
        answerable_count = summary["answerable_count"]
        unanswerable_count = summary["unanswerable_count"]

        k_values = sorted(list(set(run["top_k"] for run in experiment_data)))
        thresholds = sorted(list(set(run["score_threshold"] for run in experiment_data)))

        lines = []
        lines.append("="*60)
        lines.append("                RAG RETRIEVAL EVALUATION REPORT")
        lines.append("="*60)
        lines.append("")
        lines.append("------------------------------------------------------------")
        lines.append("Evaluation Configuration")
        lines.append("------------------------------------------------------------")
        lines.append(f"Dataset Path:             {dataset_path}")
        lines.append(f"Number of Test Cases:     {total_queries}")
        lines.append(f"  - Answerable Queries:   {answerable_count}")
        lines.append(f"  - Unanswerable Queries: {unanswerable_count}")
        lines.append(f"Embedding Model:          {embedding_model}")
        lines.append(f"Chunk Size:               {chunk_size}")
        lines.append(f"Chunk Overlap:            {chunk_overlap}")
        lines.append(f"Tested K Values:          {', '.join(map(str, k_values))}")
        lines.append(f"Tested Relevance Gates:   {', '.join(f'{t:.2f}' for t in thresholds)}")
        lines.append("")

        lines.append("------------------------------------------------------------")
        lines.append("Overall Results Grid")
        lines.append("------------------------------------------------------------")
        lines.append(f"{'K':<3} | {'Gate':<5} | {'Hit@K':<8} | {'Recall@K':<9} | {'MRR':<8} | {'Src Hit':<8} | {'Src Recall':<10} | {'False Retr':<10}")
        lines.append("-" * 88)

        for run in experiment_data:
            k = run["top_k"]
            thresh = run["score_threshold"]
            sum_data = run["summary"]
            
            lines.append(
                f"{k:<3} | {thresh:<5.2f} | "
                f"{sum_data['hit_rate']:<8.4f} | "
                f"{sum_data['recall']:<9.4f} | "
                f"{sum_data['mrr']:<8.4f} | "
                f"{sum_data['source_hit_rate']:<8.4f} | "
                f"{sum_data['source_recall']:<10.4f} | "
                f"{sum_data['false_retrieval_rate']:<10.4f}"
            )
        lines.append("")

        lines.append("------------------------------------------------------------")
        lines.append("Best Configuration Recommendations & Trade-offs")
        lines.append("------------------------------------------------------------")
        
        best_run_answerable = max(experiment_data, key=lambda x: (x["summary"]["recall"] + x["summary"]["mrr"]))
        best_run_abstain = min(experiment_data, key=lambda x: x["summary"]["false_retrieval_rate"])

        lines.append(f"Best Answerable Retrieval Config: K={best_run_answerable['top_k']} at Gate Threshold={best_run_answerable['score_threshold']:.2f}")
        lines.append(f"  - Hit@K: {best_run_answerable['summary']['hit_rate']:.4f} | Recall@K: {best_run_answerable['summary']['recall']:.4f} | MRR: {best_run_answerable['summary']['mrr']:.4f}")
        lines.append(f"  - False Retrieval Rate: {best_run_answerable['summary']['false_retrieval_rate']:.4f}")
        lines.append("")
        lines.append(f"Best Abstaining Config:            K={best_run_abstain['top_k']} at Gate Threshold={best_run_abstain['score_threshold']:.2f}")
        lines.append(f"  - False Retrieval Rate: {best_run_abstain['summary']['false_retrieval_rate']:.4f}")
        lines.append(f"  - Answerable Recall:    {best_run_abstain['summary']['recall']:.4f}")
        lines.append("")
        lines.append("Trade-off Analysis:")
        lines.append("  - Lower Gate Thresholds (0.40) provide high recall but increase False Retrieval Rates for unanswerable queries.")
        lines.append("  - Higher Gate Thresholds (0.60/0.70) drop False Retrieval Rates but risk gating out relevant context, reducing Recall.")
        lines.append("")

        lines.append("------------------------------------------------------------")
        lines.append("Per-Query Results Breakdown (Default Run: K=3, Gate=0.40)")
        lines.append("------------------------------------------------------------")
        
        default_run = next((run for run in experiment_data if run["top_k"] == 3 and run["score_threshold"] == 0.40), experiment_data[0])
        
        for item in default_run["results"]:
            lines.append(f"Query ID:        {item['test_id']}")
            lines.append(f"Question:        {item['question']}")
            lines.append(f"Answerable:      {item['answerable']}")
            lines.append(f"Expected Chunks: {', '.join(item['expected_chunk_ids']) if item['expected_chunk_ids'] else 'None'}")
            lines.append(f"Expected Source: {', '.join(item['expected_sources']) if item['expected_sources'] else 'None'}")
            
            ret_chunks_str = []
            for cid, score in zip(item["retrieved_chunk_ids"], item["retrieval_scores"]):
                ret_chunks_str.append(f"{cid} (score: {score:.4f})")
            lines.append(f"Retrieved Chunks: {', '.join(ret_chunks_str) if ret_chunks_str else 'None'}")
            lines.append(f"Retrieved Source: {', '.join(set(item['retrieved_sources'])) if item['retrieved_sources'] else 'None'}")
            lines.append(f"Accepted Chunks:  {item['accepted_chunks']}")

            if item["answerable"]:
                lines.append(
                    f"Metrics:         Hit: {item['hit_at_k']} | "
                    f"Recall: {item['recall_at_k'] if item['recall_at_k'] is not None else 'N/A'} | "
                    f"RR: {item['reciprocal_rank'] if item['reciprocal_rank'] is not None else 'N/A'} | "
                    f"Source Hit: {item['source_hit_at_k'] if item['source_hit_at_k'] is not None else 'N/A'} | "
                    f"Source Recall: {item['source_recall_at_k'] if item['source_recall_at_k'] is not None else 'N/A'}"
                )
            else:
                lines.append(f"Metrics:         False Retrieval: {item['false_retrieval']}")
            
            lines.append("-" * 40)

        return "\n".join(lines)

    @staticmethod
    def generate_chunking_report(chunking_data: list, baseline_chunks: int = 25) -> str:
        """
        Generates comparison tables and recommendations for the Phase 2 Chunking sweep.
        """
        if not chunking_data:
            return "No chunking sweep data available."

        lines = []
        lines.append("="*60)
        lines.append("=========================================")
        lines.append("PHASE 2 — CHUNKING EVALUATION")
        lines.append("=========================================")
        lines.append("================*========================")
        lines.append("")

        # 1. Print metrics for each run
        for run in chunking_data:
            size = run["chunk_size"]
            overlap = run["chunk_overlap"]
            stats = run["chunk_statistics"]
            metrics = run["retrieval_metrics"]

            lines.append(f"Configuration: chunk_size={size}, chunk_overlap={overlap}")
            lines.append(f"Documents:     {run['document_count']}")
            lines.append(f"Chunks:        {run['total_chunks']}")
            lines.append(f"Chunk Character Lengths: min={stats['min_length']}, max={stats['max_length']}, avg={stats['average_length']:.1f}, median={stats['median_length']}")
            lines.append(f"Quality Metrics: very_small={stats['very_small_chunks']}, close_to_max={stats['close_to_max_chunks']}, incomplete_sentences={stats['incomplete_sentences']}, orphaned_headings={stats['orphaned_headings']}")
            lines.append(
                f"Retrieval Quality (K=3, Gate=0.40):\n"
                f"  Hit@3:          {metrics['hit_at_3']:.4f}\n"
                f"  Recall@3:       {metrics['recall_at_3']:.4f}\n"
                f"  MRR:            {metrics['mrr']:.4f}\n"
                f"  Source Recall:  {metrics['source_recall_at_3']:.4f}\n"
                f"  False Retrieval: {metrics['false_retrieval_rate']:.4f}"
            )
            lines.append("-" * 40)

        lines.append("\n=========================================")
        lines.append("CHUNKING COMPARISON")
        lines.append("=========================================")
        lines.append(f"{'Size':<5} | {'Over':<4} | {'Chunks':<6} | {'Hit@3':<7} | {'Rec@3':<7} | {'MRR':<7} | {'Src Rec':<7} | {'False Ret':<9} | {'Workload':<8}")
        lines.append("-" * 88)

        # Build comparison grid
        for run in chunking_data:
            size = run["chunk_size"]
            overlap = run["chunk_overlap"]
            metrics = run["retrieval_metrics"]
            workload = run["total_chunks"] / baseline_chunks

            lines.append(
                f"{size:<5} | {overlap:<4} | {run['total_chunks']:<6} | "
                f"{metrics['hit_at_3']:<7.4f} | {metrics['recall_at_3']:<7.4f} | "
                f"{metrics['mrr']:<7.4f} | {metrics['source_recall_at_3']:<7.4f} | "
                f"{metrics['false_retrieval_rate']:<9.4f} | {workload:<8.2f}"
            )

        # 2. Recommendations scoring
        scored_configs = []
        for run in chunking_data:
            metrics = run["retrieval_metrics"]
            workload = run["total_chunks"] / baseline_chunks

            # Formula calculations
            retr_score = metrics["recall_at_3"] * 0.4 + metrics["mrr"] * 0.4 + metrics["source_recall_at_3"] * 0.2
            eff_score = 1.0 / (1.0 + workload)
            balanced_score = retr_score * 0.7 + eff_score * 0.3

            scored_configs.append({
                "run": run,
                "retr_score": retr_score,
                "balanced_score": balanced_score
            })

        best_retr = max(scored_configs, key=lambda x: x["retr_score"])
        best_bal = max(scored_configs, key=lambda x: x["balanced_score"])

        lines.append("\n=========================================")
        lines.append("RECOMMENDATION")
        lines.append("=========================================")
        
        lines.append(f"Best Retrieval Configuration: chunk_size={best_retr['run']['chunk_size']}, chunk_overlap={best_retr['run']['chunk_overlap']}")
        lines.append(f"  - Retrieval Score: {best_retr['retr_score']:.4f}")
        lines.append(f"  - Chunks Generated: {best_retr['run']['total_chunks']}")
        lines.append("")
        lines.append(f"Best Balanced Configuration:  chunk_size={best_bal['run']['chunk_size']}, chunk_overlap={best_bal['run']['chunk_overlap']}")
        lines.append(f"  - Balanced Score:  {best_bal['balanced_score']:.4f}")
        lines.append(f"  - Chunks Generated: {best_bal['run']['total_chunks']}")
        lines.append("")
        
        lines.append("Reason:")
        lines.append("  - Retrieval scoring favors configurations yielding the highest Hit, Recall, and MRR bounds.")
        lines.append("  - Balanced scoring incorporates an efficiency penalty to discourage chunk explosion (excessive workload / storage splits).")
        lines.append("  - Check chunk quality stats to avoid configurations producing too many orphaned headings or incomplete sentence splits.")

        return "\n".join(lines)

    @staticmethod
    def generate_answer_accuracy_report(accuracy_data: dict) -> str:
        """
        Generates a human-readable Phase 3 Answer Accuracy report.
        """
        if not accuracy_data:
            return "No answer accuracy data available."

        summary = accuracy_data.get("summary", {})
        per_query = accuracy_data.get("per_query", [])
        config = accuracy_data.get("config", {})

        lines = []
        lines.append("=" * 60)
        lines.append("PHASE 3 — ANSWER ACCURACY EVALUATION")
        lines.append("=" * 60)
        lines.append("")
        lines.append("------------------------------------------------------------")
        lines.append("Configuration")
        lines.append("------------------------------------------------------------")
        lines.append(f"Model:                qwen2.5:3b (local Ollama)")
        lines.append(f"Temperature:          0.0 (deterministic)")
        lines.append(f"Retrieval K:          {config.get('top_k', 3)}")
        lines.append(f"Relevance Gate:       {config.get('score_threshold', 0.40)}")
        lines.append("")

        lines.append("------------------------------------------------------------")
        lines.append("Overall Metrics")
        lines.append("------------------------------------------------------------")
        lines.append(f"Total Queries:        {summary.get('total_queries', 0)}")
        lines.append(f"  Answerable:         {summary.get('answerable_count', 0)}")
        lines.append(f"  Unanswerable:       {summary.get('unanswerable_count', 0)}")
        lines.append("")
        lines.append(f"  Correct:            {summary.get('correct', 0)}")
        lines.append(f"  Partially Correct:  {summary.get('partially_correct', 0)}")
        lines.append(f"  Incorrect:          {summary.get('incorrect', 0)}")
        lines.append("")
        lines.append(f"Answer Accuracy:      {summary.get('answer_accuracy', 0.0):.4f}")
        lines.append(f"Partial Accuracy:     {summary.get('partial_accuracy_rate', 0.0):.4f}")
        lines.append(f"Incorrect Rate:       {summary.get('incorrect_rate', 0.0):.4f}")
        lines.append("")
        lines.append(f"  Correct Abstention: {summary.get('correct_abstention', 0)}")
        lines.append(f"  Hallucinated:       {summary.get('hallucinated_answers', 0)}")
        lines.append(f"  Unsafe Abstention:  {summary.get('unsafe_abstentions', 0)}")
        lines.append("")
        lines.append(f"Abstention Rate:      {summary.get('correct_abstention_rate', 0.0):.4f}")
        lines.append(f"Hallucination Rate:   {summary.get('hallucination_rate', 0.0):.4f}")
        lines.append("")

        lines.append("------------------------------------------------------------")
        lines.append("Failure Breakdown")
        lines.append("------------------------------------------------------------")
        lines.append(f"End-to-End Success:   {summary.get('end_to_end_success', 0)}")
        lines.append(f"Generation Failures:  {summary.get('generation_failures', 0)}")
        lines.append(f"Retrieval Failures:   {summary.get('retrieval_failures', 0)}")
        lines.append("")

        lines.append("------------------------------------------------------------")
        lines.append("Per-Query Results")
        lines.append("------------------------------------------------------------")

        for r in per_query:
            lines.append(f"Query ID:         {r['test_id']}")
            lines.append(f"Question:         {r['question']}")
            lines.append(f"Answerable:       {r['answerable']}")
            lines.append(f"Retrieved Sources: {', '.join(r['retrieved_sources']) if r['retrieved_sources'] else 'None'}")

            answer_preview = r.get("generated_answer", "")[:200].replace("\n", " ")
            lines.append(f"Generated Answer: {answer_preview}{'...' if len(r.get('generated_answer','')) > 200 else ''}")

            if r["answerable"]:
                lines.append(f"Expected Facts:   {len(r.get('expected_facts', []))}")
                for fr in r.get("fact_results", []):
                    icon = "[Y]" if fr.get("supported") else "[N]"
                    lines.append(f"  {icon} {fr['fact']}")
                lines.append(f"Supported:        {r.get('facts_supported', 0)}/{r.get('facts_expected', 0)}")
                lines.append(f"Accuracy Class:   {r.get('accuracy_class', 'N/A')}")
                lines.append(f"Failure Mode:     {r.get('failure_mode', 'N/A')}")
                lines.append(f"Eval Method:      {r.get('evaluation_method', 'N/A')}")
            else:
                lines.append(f"Abstention Class: {r.get('accuracy_class', 'N/A')}")

            lines.append("-" * 40)

        return "\n".join(lines)

    @staticmethod
    def generate_grounding_report(grounding_data: dict) -> str:
        """
        Generates a human-readable Phase 4 Groundedness & Hallucination report.
        """
        if not grounding_data:
            return "No grounding evaluation data available."

        summary = grounding_data.get("summary", {})
        per_query = grounding_data.get("per_query", [])
        config = grounding_data.get("config", {})

        lines = []
        lines.append("=" * 60)
        lines.append("PHASE 4 -- GROUNDEDNESS & HALLUCINATION EVALUATION")
        lines.append("=" * 60)
        lines.append("")
        lines.append("-" * 60)
        lines.append("Configuration")
        lines.append("-" * 60)
        lines.append(f"Model:                  qwen2.5:3b (local Ollama)")
        lines.append(f"Temperature:            0.0 (deterministic)")
        lines.append(f"Retrieval K:            {config.get('top_k', 3)}")
        lines.append(f"Relevance Gate:         {config.get('score_threshold', 0.40)}")
        lines.append(f"Grounding basis:        Retrieved context only")
        lines.append("")

        lines.append("-" * 60)
        lines.append("Overall Metrics")
        lines.append("-" * 60)
        lines.append(f"Total Queries:          {summary.get('total_queries', 0)}")
        lines.append(f"  Answerable:           {summary.get('answerable_count', 0)}")
        lines.append(f"  Unanswerable:         {summary.get('unanswerable_count', 0)}")
        lines.append(f"  Evaluable (w/ facts): {summary.get('evaluable_count', 0)}")
        lines.append("")
        lines.append(f"  Grounded:             {summary.get('grounded', 0)}")
        lines.append(f"  Partially Grounded:   {summary.get('partially_grounded', 0)}")
        lines.append(f"  Ungrounded:           {summary.get('ungrounded', 0)}")
        lines.append(f"  No Factual Claims:    {summary.get('no_factual_claims', 0)}")
        lines.append("")
        lines.append(f"Groundedness Rate:      {summary.get('groundedness_rate', 0.0):.4f}")
        lines.append(f"Partial Grounding Rate: {summary.get('partial_grounding_rate', 0.0):.4f}")
        lines.append(f"Hallucination Rate:     {summary.get('hallucination_rate', 0.0):.4f}")
        lines.append("")
        lines.append(f"Total Claims Evaluated: {summary.get('total_claims_evaluated', 0)}")
        lines.append(f"  Supported:            {summary.get('total_supported_claims', 0)}")
        lines.append(f"  Partial:              {summary.get('total_partial_claims', 0)}")
        lines.append(f"  Unsupported:          {summary.get('total_unsupported_claims', 0)}")
        lines.append("")
        lines.append(f"Claim Support Rate:     {summary.get('claim_support_rate', 0.0):.4f}")
        lines.append(f"Unsupported Claim Rate: {summary.get('unsupported_claim_rate', 0.0):.4f}")
        lines.append("")
        lines.append(f"  Correct Abstention:   {summary.get('correct_abstention', 0)}")
        lines.append(f"  False Answer:         {summary.get('false_answer', 0)}")
        lines.append(f"  Hallucination (Unans):{summary.get('hallucination_unanswerable', 0)}")
        lines.append("")
        lines.append(f"Abstention Rate:        {summary.get('correct_abstention_rate', 0.0):.4f}")
        lines.append(f"False Answer Rate:      {summary.get('false_answer_rate', 0.0):.4f}")
        lines.append("")

        # Highlight failed cases
        failed = [
            r for r in per_query
            if r.get("classification") in {
                "UNGROUNDED", "PARTIALLY_GROUNDED", "FALSE_ANSWER", "HALLUCINATION"
            }
        ]

        if failed:
            lines.append("-" * 60)
            lines.append("FAILED / PARTIALLY FAILED CASES")
            lines.append("-" * 60)
            for r in failed:
                lines.append(f"[FAIL] {r['test_id']}")
                lines.append(f"Question:       {r['question']}")
                lines.append(f"Classification: {r['classification']}")
                lines.append(f"Answer preview: {r.get('answer', '')[:200]}")
                unsupported_claims = [
                    c for c in r.get("claims", [])
                    if c.get("status") == "UNSUPPORTED"
                ]
                for uc in unsupported_claims:
                    lines.append(f"  [UNSUPPORTED] {uc['claim']}")
                    lines.append(f"  Reason:       {uc.get('reason', 'N/A')}")
                lines.append("-" * 40)
            lines.append("")

        lines.append("-" * 60)
        lines.append("Per-Query Results")
        lines.append("-" * 60)

        icon_map = {
            "SUPPORTED": "[Y]",
            "PARTIALLY_SUPPORTED": "[~]",
            "UNSUPPORTED": "[N]",
        }

        for r in per_query:
            cls = r.get("classification", "N/A")
            lines.append(f"Query ID:         {r['test_id']}")
            lines.append(f"Question:         {r['question']}")
            lines.append(f"Answerable:       {r['answerable']}")
            lines.append(f"Classification:   {cls}")
            lines.append(f"Retrieved:        {', '.join(r.get('retrieved_sources', [])) or 'None'}")

            answer_preview = r.get("answer", "")[:200].replace("\n", " ")
            lines.append(f"Answer:           {answer_preview}{'...' if len(r.get('answer','')) > 200 else ''}")

            if r.get("claims"):
                lines.append(f"Claims ({r['total_claims']}):")
                for c in r["claims"]:
                    icon = icon_map.get(c.get("status"), "[?]")
                    lines.append(f"  {icon} {c['claim']}")
                    if c.get("evidence"):
                        lines.append(f"      Evidence: {c['evidence'][:120]}")
                    lines.append(f"      Reason:   {c.get('reason', '')[:120]}")
                    lines.append(f"      Method:   {c.get('evaluation_method', 'N/A')}")
                lines.append(
                    f"Counts: {r['supported_claims']} supported / "
                    f"{r['partial_claims']} partial / "
                    f"{r['unsupported_claims']} unsupported"
                )
            else:
                lines.append("Claims:           (none extracted)")

            lines.append("-" * 40)

        return "\n".join(lines)

    @staticmethod
    def generate_citation_report(citation_data: dict) -> str:
        """
        Generates a human-readable Phase 5 Simple Citation Evaluation report.
        """
        if not citation_data:
            return "No citation evaluation data available."

        summary = citation_data.get("summary", {})
        per_query = citation_data.get("per_query", [])

        lines = []
        lines.append("=" * 60)
        lines.append("CITATION EVALUATION")
        lines.append("-" * 60)
        lines.append("")
        lines.append(f"Total Evaluated Queries: {summary.get('total_queries', 0)}")
        lines.append("")
        lines.append(f"Correct Citations: {summary.get('correct_citations', 0)}")
        lines.append(f"Missing Citations: {summary.get('missing_citations', 0)}")
        lines.append(f"Wrong Citations: {summary.get('wrong_citations', 0)}")
        lines.append(f"Unsupported Citations: {summary.get('unsupported_citations', 0)}")
        lines.append(f"Correct Abstentions: {summary.get('correct_abstentions', 0)}")
        lines.append("")
        lines.append(f"Citation Accuracy: {summary.get('citation_accuracy', 0.0) * 100:.2f}%")
        lines.append(f"Citation Coverage: {summary.get('citation_coverage', 0.0) * 100:.2f}%")
        lines.append(f"Missing Citation Rate: {summary.get('missing_citation_rate', 0.0) * 100:.2f}%")
        lines.append(f"Unsupported Citation Rate: {summary.get('unsupported_citation_rate', 0.0) * 100:.2f}%")
        lines.append("")

        # Failed cases highlight
        failed_cases = [
            q for q in per_query
            if q["classification"] in {"MISSING_CITATION", "WRONG_CITATION", "UNSUPPORTED_CITATION"}
        ]

        if failed_cases:
            lines.append("FAILED CASES")
            lines.append("------------")
            for q in failed_cases:
                lines.append("[FAIL]")
                lines.append(f"Test ID:    {q['test_id']}")
                lines.append(f"Question:   {q['question']}")
                lines.append(f"Answer:     {q['answer']}")
                lines.append(f"Citation:   {', '.join(q.get('cited_sources', [])) or 'None'}")
                lines.append(f"Reason:     {q.get('reason', 'N/A')}")
                lines.append("-" * 40)
            lines.append("")

        return "\n".join(lines)

