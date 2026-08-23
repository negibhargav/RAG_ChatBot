from abc import ABC, abstractmethod

class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class OllamaConnectionError(LLMProviderError):
    """Raised when unable to connect to the local Ollama server."""
    pass


class OllamaModelNotFoundError(LLMProviderError):
    """Raised when the requested model is not found locally in Ollama."""
    pass


class LLMTimeoutError(LLMProviderError):
    """Raised when the generation request times out."""
    pass


class BaseLLMProvider(ABC):
    """
    Abstract Base Class for LLM/SLM inference engines.
    Allows swapping local runtimes (e.g. Ollama, llama.cpp, Transformers).
    """

    @abstractmethod
    def generate(
        self,
        prompt,
        model,
        temperature=0.0,
        max_tokens=None,
        timeout=60
    ):
        """
        Send prompt completion request to the inference provider.
        
        Returns:
            dict: Standard normalized response schema:
                  {
                      "answer": str,
                      "model": str,
                      "provider": str,
                      "usage": dict or None,
                      "finish_reason": str
                  }
        """
        pass

    @abstractmethod
    def health_check(self, model=None):
        """
        Verify that the provider runtime is running and reachable.
        
        Returns:
            dict: Status details:
                  {
                      "available": bool,
                      "model": str or None,
                      "endpoint": str,
                      "error": str or None
                  }
        """
        pass
