import re

class QueryRouter:
    """
    Lightweight deterministic query router to classify user queries
    as CONVERSATIONAL or KNOWLEDGE without invoking an LLM.
    """

    CONVERSATIONAL_PATTERNS = [
        # Greetings
        r'^\s*(hi|hello|hey|hola|greetings|good\s+morning|good\s+afternoon|good\s+evening)\b',
        # Social check-ins / Identity
        r'^\s*how\s+are\s+you\b',
        r'^\s*who\s+are\s+you\b',
        r'^\s*what\s+is\s+your\s+name\b',
        r'^\s*how\'s\s+it\s+going\b',
        # Gratitude
        r'^\s*(thanks|thank\s+you|thx|cheers)\b',
        # Farewells / System
        r'^\s*(bye|goodbye|see\s+you|farewell)\b',
        r'^\s*(exit|quit)\b'
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.CONVERSATIONAL_PATTERNS]

    def route(self, query: str) -> dict:
        """
        Classifies the incoming query as CONVERSATIONAL or KNOWLEDGE.
        
        Args:
            query (str): The user query string.
            
        Returns:
            dict: A dictionary containing "query_type": "CONVERSATIONAL" | "KNOWLEDGE".
        """
        if not query or not query.strip():
            return {"query_type": "CONVERSATIONAL"}

        clean_query = query.strip()
        
        # Check against conversational patterns
        is_conversational_match = False
        for pattern in self.patterns:
            if pattern.search(clean_query):
                is_conversational_match = True
                break

        if is_conversational_match:
            # Check length: if the query contains a greeting but is longer than 4 words,
            # it is likely a hybrid query (e.g. "Hi, what are the listed KPIs?"),
            # which should be routed to KNOWLEDGE.
            words = clean_query.split()
            if len(words) <= 4:
                return {"query_type": "CONVERSATIONAL"}

        return {"query_type": "KNOWLEDGE"}
