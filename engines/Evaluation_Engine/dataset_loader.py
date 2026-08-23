import json
import os

class EvaluationDatasetLoader:
    """
    Loader class for parsing and validating RAG evaluation golden datasets.
    """
    @staticmethod
    def load(path: str) -> list[dict]:
        """
        Loads a JSON evaluation dataset from path and performs validation checks.
        
        Args:
            path (str): File path to the JSON golden dataset.
            
        Returns:
            list[dict]: Validated list of test cases.
            
        Raises:
            ValueError: If dataset is invalid, empty, or has duplicate IDs.
            TypeError: If fields have incorrect types.
            FileNotFoundError: If the file path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Evaluation dataset path does not exist: {path}")

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse golden dataset JSON: {e}")

        if not isinstance(data, list):
            raise TypeError("Dataset top-level structure must be a JSON Array (list).")

        if len(data) == 0:
            raise ValueError("The evaluation dataset is empty.")

        seen_test_ids = set()

        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise TypeError(f"Test case at index {idx} must be a JSON object (dict).")

            # Check required fields
            required_keys = ["test_id", "question", "answerable", "expected_chunk_ids", "expected_content", "expected_sources", "expected_facts"]
            for key in required_keys:
                if key not in item:
                    raise ValueError(f"Test case at index {idx} is missing required field '{key}'")

            test_id = item["test_id"]
            question = item["question"]
            answerable = item["answerable"]
            expected_chunk_ids = item["expected_chunk_ids"]
            expected_content = item["expected_content"]
            expected_sources = item["expected_sources"]
            expected_facts = item["expected_facts"]

            # Validate types and contents
            if not isinstance(test_id, str) or not test_id.strip():
                raise ValueError(f"Test case at index {idx} has invalid 'test_id'. Must be a non-empty string.")
            
            if test_id in seen_test_ids:
                raise ValueError(f"Duplicate test_id found: '{test_id}' at index {idx}.")
            seen_test_ids.add(test_id)

            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Test case '{test_id}' has invalid 'question'. Must be a non-empty string.")

            if not isinstance(answerable, bool):
                raise TypeError(f"Test case '{test_id}' has invalid 'answerable'. Must be a boolean.")

            if not isinstance(expected_chunk_ids, list):
                raise TypeError(f"Test case '{test_id}' has invalid 'expected_chunk_ids'. Must be a list.")
            for cid in expected_chunk_ids:
                if not isinstance(cid, str):
                    raise TypeError(f"Test case '{test_id}' expected_chunk_ids contains non-string elements.")

            if not isinstance(expected_content, list):
                raise TypeError(f"Test case '{test_id}' has invalid 'expected_content'. Must be a list.")
            for c in expected_content:
                if not isinstance(c, str):
                    raise TypeError(f"Test case '{test_id}' expected_content contains non-string elements.")

            if not isinstance(expected_sources, list):
                raise TypeError(f"Test case '{test_id}' has invalid 'expected_sources'. Must be a list.")
            for src in expected_sources:
                if not isinstance(src, str):
                    raise TypeError(f"Test case '{test_id}' expected_sources contains non-string elements.")

            if not isinstance(expected_facts, list):
                raise TypeError(f"Test case '{test_id}' has invalid 'expected_facts'. Must be a list.")
            for fact in expected_facts:
                if not isinstance(fact, str):
                    raise TypeError(f"Test case '{test_id}' expected_facts contains non-string elements.")

            # Validate unanswerable query constraints
            if not answerable:
                if len(expected_chunk_ids) > 0:
                    raise ValueError(f"Unanswerable test case '{test_id}' must have empty expected_chunk_ids.")
                if len(expected_content) > 0:
                    raise ValueError(f"Unanswerable test case '{test_id}' must have empty expected_content.")
                if len(expected_sources) > 0:
                    raise ValueError(f"Unanswerable test case '{test_id}' must have empty expected_sources.")
                if len(expected_facts) > 0:
                    raise ValueError(f"Unanswerable test case '{test_id}' must have empty expected_facts.")

        return data
