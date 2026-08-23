def calculate_hit_at_k(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str], answerable: bool = True) -> int:
    """
    Calculates Hit@K. Returns 1 if at least one expected chunk is in retrieved list; else 0.
    For unanswerable queries, returns None.
    """
    if not answerable:
        return None
    if not expected_chunk_ids:
        return 0
    # Ensure expected_chunk_ids is a set for faster lookup
    expected_set = set(expected_chunk_ids)
    return 1 if any(cid in expected_set for cid in retrieved_chunk_ids) else 0


def calculate_recall_at_k(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str], answerable: bool = True) -> float:
    """
    Calculates Recall@K. Returns fraction of expected chunks retrieved.
    For unanswerable queries, returns None.
    """
    if not answerable:
        return None
    if not expected_chunk_ids:
        return 0.0
    expected_set = set(expected_chunk_ids)
    intersection = [cid for cid in retrieved_chunk_ids if cid in expected_set]
    # Unique expected chunks retrieved / total expected chunks
    return len(set(intersection)) / len(expected_set)


def calculate_reciprocal_rank(retrieved_chunk_ids: list[str], expected_chunk_ids: list[str], answerable: bool = True) -> float:
    """
    Calculates Reciprocal Rank (RR). Returns 1 / rank of first relevant chunk; else 0.0.
    For unanswerable queries, returns None.
    """
    if not answerable:
        return None
    if not expected_chunk_ids:
        return 0.0
    expected_set = set(expected_chunk_ids)
    for idx, cid in enumerate(retrieved_chunk_ids):
        if cid in expected_set:
            return 1.0 / (idx + 1)
    return 0.0


def calculate_source_hit_at_k(retrieved_sources: list[str], expected_sources: list[str], answerable: bool = True) -> int:
    """
    Calculates Source Hit@K. Returns 1 if at least one expected source is in retrieved sources; else 0.
    For unanswerable queries, returns None.
    """
    if not answerable:
        return None
    if not expected_sources:
        return 0
    expected_set = set(expected_sources)
    return 1 if any(src in expected_set for src in retrieved_sources) else 0


def calculate_source_recall_at_k(retrieved_sources: list[str], expected_sources: list[str], answerable: bool = True) -> float:
    """
    Calculates Source Recall@K. Returns fraction of expected sources retrieved.
    For unanswerable queries, returns None.
    """
    if not answerable:
        return None
    if not expected_sources:
        return 0.0
    expected_set = set(expected_sources)
    intersection = [src for src in retrieved_sources if src in expected_set]
    return len(set(intersection)) / len(expected_set)


def calculate_false_retrieval(answerable: bool, accepted_chunks: list) -> bool:
    """
    Calculates false retrieval. Returns True if unanswerable and there are accepted chunks; else False.
    """
    if answerable:
        return False
    return len(accepted_chunks) > 0


def calculate_mrr(reciprocal_ranks: list[float]) -> float:
    """
    Calculates Mean Reciprocal Rank (MRR) across a list of reciprocal ranks.
    Filters out None values (unanswerable queries).
    """
    valid_rrs = [rr for rr in reciprocal_ranks if rr is not None]
    if not valid_rrs:
        return 0.0
    return sum(valid_rrs) / len(valid_rrs)


def calculate_false_retrieval_rate(false_retrieval_list: list[bool]) -> float:
    """
    Calculates False Retrieval Rate across a list of false retrieval booleans.
    Filters out None values.
    """
    valid_frs = [fr for fr in false_retrieval_list if fr is not None]
    if not valid_frs:
        return 0.0
    fr_numeric = [1.0 if fr else 0.0 for fr in valid_frs]
    return sum(fr_numeric) / len(fr_numeric)


class MetricCalculator:
    """
    Calculator class for compatibility, exposing pure staticmethods.
    """
    @staticmethod
    def calculate_hit_at_k(*args, **kwargs):
        return calculate_hit_at_k(*args, **kwargs)

    @staticmethod
    def calculate_recall_at_k(*args, **kwargs):
        return calculate_recall_at_k(*args, **kwargs)

    @staticmethod
    def calculate_reciprocal_rank(*args, **kwargs):
        return calculate_reciprocal_rank(*args, **kwargs)

    @staticmethod
    def calculate_source_hit_at_k(*args, **kwargs):
        return calculate_source_hit_at_k(*args, **kwargs)

    @staticmethod
    def calculate_source_recall_at_k(*args, **kwargs):
        return calculate_source_recall_at_k(*args, **kwargs)

    @staticmethod
    def calculate_false_retrieval(*args, **kwargs):
        return calculate_false_retrieval(*args, **kwargs)

    @staticmethod
    def calculate_mrr(*args, **kwargs):
        return calculate_mrr(*args, **kwargs)

    @staticmethod
    def calculate_false_retrieval_rate(*args, **kwargs):
        return calculate_false_retrieval_rate(*args, **kwargs)
