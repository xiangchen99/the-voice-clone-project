"""Local LLM client with streaming sentence chunker for low-latency speech synthesis."""

import json
import re
import urllib.request
from http.client import HTTPException
from typing import Dict, Generator, List, Optional


class GenerationError(RuntimeError):
    """The configured language model could not produce a reply."""


class LocalLLMClient:
    """Client for local LLM engines (Ollama, llama.cpp, or explicit demo mode) with streaming support."""

    def __init__(
        self,
        backend: str = "ollama",
        endpoint_url: str = "http://127.0.0.1:11434",
        model_name: str = "llama3.1:8b",
        temperature: float = 0.7,
        max_tokens: int = 300,
    ) -> None:
        self.backend = backend.lower()
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._reply_status: Optional[str] = None

    def get_status(self) -> str:
        """Report current model availability and the outcome of the last reply."""
        if self.backend == "mock":
            return "demo"
        if self.backend != "ollama":
            return self._reply_status or "unchecked"

        try:
            with urllib.request.urlopen(f"{self.endpoint_url}/api/tags", timeout=1) as response:
                result = json.loads(response.read())
        except (OSError, ValueError, HTTPException):
            return "unavailable"

        if not isinstance(result, dict) or not isinstance(result.get("models"), list):
            return "unavailable"

        expected_names = {self.model_name}
        if ":" not in self.model_name.rsplit("/", 1)[-1]:
            expected_names.add(f"{self.model_name}:latest")

        installed_names = set()
        for model in result["models"]:
            if not isinstance(model, dict):
                return "unavailable"
            name = model.get("model", model.get("name"))
            if not isinstance(name, str) or not name.strip():
                return "unavailable"
            installed_names.add(name)

        if expected_names.isdisjoint(installed_names):
            return "missing_model"
        return self._reply_status or "installed"

    def generate_response(
        self,
        system_prompt: str,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generates a complete response from the local LLM."""
        history = chat_history or []
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_message})

        try:
            if self.backend == "ollama":
                reply = self._call_ollama(messages)
            elif self.backend in ("llama_cpp", "openai_compatible"):
                reply = self._call_openai_compatible(messages)
            elif self.backend == "mock":
                reply = self._mock_response(user_message)
            else:
                raise GenerationError(f"Unsupported language model backend: {self.backend}.")
            if not reply:
                raise GenerationError("The language model returned an empty reply. Try sending again.")
        except GenerationError:
            self._reply_status = "reply_failed"
            raise
        except (OSError, ValueError, LookupError, AttributeError, TypeError, HTTPException) as exc:
            self._reply_status = "reply_failed"
            backend = "Ollama" if self.backend == "ollama" else self.backend
            raise GenerationError(f"{backend} could not reply. Check that the local service and model are available, then retry.") from exc
        self._reply_status = "working"
        return reply

    def _call_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Calls local Ollama /api/chat endpoint."""
        url = f"{self.endpoint_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["message"]["content"].strip()

    def _call_openai_compatible(self, messages: List[Dict[str, str]]) -> str:
        """Calls OpenAI-compatible /v1/chat/completions endpoint (llama.cpp / vLLM)."""
        url = f"{self.endpoint_url}/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"].strip()

    def _mock_response(self, user_prompt: str) -> str:
        """Deterministic mock response for offline/testing runs."""
        return (
            f"To be honest, looking at that from my perspective, "
            f"the core issue with '{user_prompt}' comes down to how we structure the fundamentals. "
            f"Let's break this down systematically."
        )

    def stream_sentence_chunks(
        self, text: str
    ) -> Generator[str, None, None]:
        """Splits full text into natural spoken sentences for incremental TTS synthesis.

        Emits complete clauses terminated by '.', '!', or '?'.
        """
        # Split on sentence boundaries while keeping punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        for s in sentences:
            s_clean = s.strip()
            if s_clean:
                yield s_clean

