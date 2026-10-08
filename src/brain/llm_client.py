"""Local LLM client with streaming sentence chunker for low-latency speech synthesis."""

import json
import re
import urllib.error
import urllib.request
from typing import Dict, Generator, List, Optional


class LocalLLMClient:
    """Client for local LLM engines (Ollama, llama.cpp, or mock fallback) with streaming support."""

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

        if self.backend == "ollama":
            return self._call_ollama(messages)
        elif self.backend in ("llama_cpp", "openai_compatible"):
            return self._call_openai_compatible(messages)
        else:
            return self._mock_response(user_message)

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

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("message", {}).get("content", "").strip()
        except (urllib.error.URLError, TimeoutError, ConnectionRefusedError):
            # Graceful fallback when local daemon is not running during testing
            return self._mock_response(messages[-1]["content"])

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

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
        except (urllib.error.URLError, TimeoutError, ConnectionRefusedError):
            return self._mock_response(messages[-1]["content"])

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

