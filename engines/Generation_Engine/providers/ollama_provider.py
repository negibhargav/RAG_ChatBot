import requests
from .base import (
    BaseLLMProvider,
    LLMProviderError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    LLMTimeoutError
)

class OllamaProvider(BaseLLMProvider):
    """
    Ollama Local SLM Provider.
    Communicates via REST calls to the local Ollama server.
    """

    def __init__(self, base_url="http://127.0.0.1:11434"):
        self.base_url = base_url.rstrip("/")

    def health_check(self, model=None):
        """
        Check if Ollama server is up and the model is downloaded locally.
        """
        try:
            # Query local tags to verify connection and model presence
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return {
                    "available": False,
                    "model": model,
                    "endpoint": self.base_url,
                    "error": f"Ollama returned HTTP status {response.status_code}"
                }
            
            data = response.json()
            local_models = [m["name"] for m in data.get("models", [])]
            
            if model:
                # Check for model name match (handle implicit tag matching like model:latest)
                matched = False
                for m in local_models:
                    if m == model or m.startswith(model + ":") or model.startswith(m + ":"):
                        matched = True
                        break
                    if ":" not in model and m == f"{model}:latest":
                        matched = True
                        break
                
                if not matched:
                    return {
                        "available": False,
                        "model": model,
                        "endpoint": self.base_url,
                        "error": f"Model '{model}' is not pulled. Available: {local_models}"
                    }
            
            return {
                "available": True,
                "model": model,
                "endpoint": self.base_url,
                "error": None
            }

        except requests.exceptions.ConnectionError:
            return {
                "available": False,
                "model": model,
                "endpoint": self.base_url,
                "error": f"Connection refused. Ensure Ollama is running at {self.base_url}"
            }
        except Exception as e:
            return {
                "available": False,
                "model": model,
                "endpoint": self.base_url,
                "error": str(e)
            }

    def generate(
        self,
        prompt,
        model,
        temperature=0.0,
        max_tokens=None,
        timeout=60
    ):
        """
        Send prompt completion POST request to local Ollama.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        # Ensure model is downloaded/valid before generating
        status = self.health_check(model)
        if not status["available"]:
            # If server is down
            if "Connection refused" in str(status["error"]):
                raise OllamaConnectionError(
                    f"Unable to connect to local Ollama server at {self.base_url}. Ensure Ollama is running."
                )
            # If model is missing
            if "is not pulled" in str(status["error"]):
                raise OllamaModelNotFoundError(status["error"])
            raise LLMProviderError(f"Ollama health check failed: {status['error']}")

        # Build payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        # Build execution options
        options = {"temperature": float(temperature)}
        if max_tokens is not None and max_tokens > 0:
            options["num_predict"] = int(max_tokens)
        payload["options"] = options

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            
            if response.status_code != 200:
                raise LLMProviderError(
                    f"Ollama server returned error code {response.status_code}: {response.text}"
                )

            data = response.json()
            answer = data.get("response", "").strip()

            # Parse token usage metrics from Ollama response variables
            usage = None
            prompt_eval_count = data.get("prompt_eval_count")
            eval_count = data.get("eval_count")
            
            if prompt_eval_count is not None and eval_count is not None:
                usage = {
                    "input_tokens": prompt_eval_count,
                    "output_tokens": eval_count,
                    "total_tokens": prompt_eval_count + eval_count
                }

            # Map standard finish reasons
            finish_reason = "stop" if data.get("done") else None

            return {
                "answer": answer,
                "model": model,
                "provider": "ollama",
                "usage": usage,
                "finish_reason": finish_reason
            }

        except requests.exceptions.Timeout:
            raise LLMTimeoutError(f"Request to Ollama server timed out after {timeout} seconds.")
        except requests.exceptions.ConnectionError:
            raise OllamaConnectionError(f"Connection lost to Ollama server at {self.base_url}")
        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise e
            raise LLMProviderError(f"Unexpected error communicating with Ollama: {e}")
