"""End-to-end voice-to-voice conversation orchestrator."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from src.asr.transcriber import SpeechTranscriber
from src.brain.llm_client import LocalLLMClient
from src.brain.prompt_builder import PromptBuilder
from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore
from src.voice.tts_engine import BaseTTSEngine, get_tts_engine


class DigitalSelfOrchestrator:
    """Coordinates ASR, autobiographical memory retrieval, LLM persona reasoning,

    and neural voice synthesis for real-time interaction.
    """

    def __init__(self, settings_path: str = "configs/settings.yaml") -> None:
        self.settings = self._load_settings(settings_path)
        self.memory_store = MemoryStore(
            db_dir=self.settings.get("memory", {}).get("db_dir", "data/memory_db"),
            collection_name=self.settings.get("memory", {}).get("collection_name", "autobiography"),
        )
        self.memory_retriever = MemoryRetriever(
            memory_store=self.memory_store,
            top_k=self.settings.get("memory", {}).get("top_k_retrieval", 3),
        )
        self.prompt_builder = PromptBuilder()

        llm_cfg = self.settings.get("llm", {})
        self.llm_client = LocalLLMClient(
            backend=llm_cfg.get("backend", "ollama"),
            endpoint_url=llm_cfg.get("endpoint_url", "http://127.0.0.1:11434"),
            model_name=llm_cfg.get("model_name", "llama3.1:8b"),
            temperature=llm_cfg.get("temperature", 0.75),
            max_tokens=llm_cfg.get("max_tokens", 350),
        )

        asr_cfg = self.settings.get("asr", {})
        self.transcriber = SpeechTranscriber(
            model_size=asr_cfg.get("model_size", "base.en"),
            device=asr_cfg.get("device", "cuda"),
            compute_type=asr_cfg.get("compute_type", "float16"),
        )

        tts_cfg = self.settings.get("tts", {})
        self.tts_engine: BaseTTSEngine = get_tts_engine(
            engine_name=tts_cfg.get("engine_name", tts_cfg.get("engine", "xtts_v2")),
            device=self.settings.get("hardware", {}).get("device", "cuda"),
        )
        self.default_ref_audio = tts_cfg.get("reference_audio_path", "data/audio/reference/master_ref.wav")

        self.chat_history: List[Dict[str, str]] = []

    def _load_settings(self, path: str) -> Dict[str, Any]:
        """Loads system settings from YAML file."""
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def process_text_turn(
        self,
        user_text: str,
        reference_audio_path: Optional[str] = None,
        output_audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Processes a text input turn: recalls memories, queries LLM, and synthesizes clone voice."""
        t_start = time.time()
        ref_audio = reference_audio_path or self.default_ref_audio
        out_audio = output_audio_path or f"data/audio/response_{int(time.time())}.wav"

        # 1. Retrieve autobiographical memories
        t_mem_start = time.time()
        memory_context = self.memory_retriever.retrieve_context(user_text)
        recalled_raw = self.memory_retriever.retrieve_memories_raw(user_text)
        t_mem = time.time() - t_mem_start

        # 2. Build system prompt
        system_prompt = self.prompt_builder.build_system_prompt(memory_context)

        # 3. Query LLM Persona
        t_llm_start = time.time()
        raw_response = self.llm_client.generate_response(
            system_prompt=system_prompt,
            user_message=user_text,
            chat_history=self.chat_history,
        )
        t_llm = time.time() - t_llm_start

        # 4. Clean text for natural speech
        spoken_text = self.prompt_builder.sanitize_for_tts(raw_response)

        # 5. Synthesize cloned voice
        t_tts_start = time.time()
        audio_file = self.tts_engine.synthesize(
            text=spoken_text,
            reference_audio_path=ref_audio,
            output_path=out_audio,
        )
        t_tts = time.time() - t_tts_start

        # 6. Update conversation history
        self.chat_history.append({"role": "user", "content": user_text})
        self.chat_history.append({"role": "assistant", "content": spoken_text})

        total_time = time.time() - t_start

        return {
            "user_text": user_text,
            "response_text": spoken_text,
            "audio_path": audio_file,
            "retrieved_memories": recalled_raw,
            "metrics": {
                "memory_retrieval_sec": round(t_mem, 3),
                "llm_generation_sec": round(t_llm, 3),
                "tts_synthesis_sec": round(t_tts, 3),
                "total_sec": round(total_time, 3),
            },
        }

    def process_audio_turn(
        self,
        input_audio_path: Union[str, Path],
        reference_audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Processes an audio input turn: transcribes speech, queries LLM, and responds with cloned voice."""
        t_asr_start = time.time()
        user_text = self.transcriber.transcribe(input_audio_path)
        t_asr = time.time() - t_asr_start

        result = self.process_text_turn(user_text, reference_audio_path=reference_audio_path)
        result["metrics"]["asr_transcription_sec"] = round(t_asr, 3)
        result["metrics"]["total_sec"] = round(result["metrics"]["total_sec"] + t_asr, 3)
        return result

    def clear_history(self) -> None:
        """Resets the active conversation history."""
        self.chat_history = []

