from .providers.base import BaseLLMProvider

class GenerationEngine:
    """
    SLM Generation Coordinator.
    Decoupled from specific runtimes, routing prompts to BaseLLMProvider.
    """

    def __init__(
        self,
        provider,
        model=None,
        temperature=0.0,
        max_tokens=None,
        timeout=60
    ):
        if not isinstance(provider, BaseLLMProvider):
            raise TypeError("provider must be an instance of BaseLLMProvider")

        self.provider = provider
        # Default model to config or allow custom overrides
        self.model = model if model is not None else "qwen2.5:3b"
        self.temperature = float(temperature)
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, prompt_input):
        """
        Generates completion response from prompt string or Context Builder dict.
        
        Args:
            prompt_input (str | dict): Prompt string or output dictionary from ContextBuilder.
            
        Returns:
            dict: Normalized output schema containing response answers and token usage.
        """
        # Handle Context Builder output package
        if isinstance(prompt_input, dict):
            # No-context handling short-circuit
            if not prompt_input.get("has_context", True):
                # Retrieve provider name if present, fallback to ollama
                provider_name = getattr(self.provider, "provider_name", "ollama")
                return {
                    "answer": "I could not find this information in the uploaded documents.",
                    "model": self.model,
                    "provider": provider_name,
                    "usage": None,
                    "finish_reason": "no_context"
                }
            prompt = prompt_input.get("prompt")
        else:
            prompt = prompt_input

        # Validate prompt is not empty
        if not prompt or not str(prompt).strip():
            raise ValueError("Prompt cannot be empty.")

        # Delegate execution to abstract provider
        return self.provider.generate(
            prompt=prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout
        )
