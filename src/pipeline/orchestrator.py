"""End-to-end voice-to-voice conversation orchestrator."""

import re
import time
from threading import Lock
from uuid import uuid4
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from src.asr.transcriber import InvalidRecording, SpeechTranscriber
from src.brain.llm_client import GenerationError, LocalLLMClient
from src.brain.prompt_builder import PromptBuilder
from src.memory.retriever import MemoryRetriever
from src.memory.store import MemoryStore
from src.voice.tts_engine import BaseTTSEngine, ReferenceAudio, SynthesisError, get_tts_engine


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
        self.default_ref_audio = tts_cfg.get("reference_audio_path") or []
        self.tts_options = {
            key: tts_cfg[key] for key in ("language", "temperature", "speed") if key in tts_cfg
        }
        self.turn_lock = Lock()

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
        reference_audio_path: Optional[ReferenceAudio] = None,
        output_audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a real or explicitly configured demo reply and attempt audio."""
        with self.turn_lock:
            started = time.perf_counter()
            memories = self.memory_retriever.retrieve_memories_raw(user_text, approved_only=True)
            system_prompt = self.prompt_builder.build_system_prompt(self.memory_retriever.format_context(memories))
            recalled = time.perf_counter()
            reply = self.llm_client.generate_response(system_prompt, user_text, chat_history=self.chat_history)
            generated = time.perf_counter()

            if self.llm_client.backend == "mock":
                memories = []
            labels = {f"M{number}" for number in range(1, len(memories) + 1)}
            cited_labels = labels.intersection(re.findall(r"\[(M[0-9]+)\]", reply))
            cited = [
                {**memory, "citation": f"M{number}"}
                for number, memory in enumerate(memories, 1) if f"M{number}" in cited_labels
            ]
            reply = re.sub(
                r"\[(M[0-9]+)\]",
                lambda match: match[0] if match[1] in labels else "",
                reply,
            ).strip()
            reply_without_citations = re.sub(r"\s*\[M[0-9]+\]", "", reply).strip()
            if not reply_without_citations:
                raise GenerationError("The language model returned an empty reply. Try sending again.")
            spoken_text = self.prompt_builder.sanitize_for_tts(reply_without_citations)
            self.chat_history.extend((
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": reply_without_citations},
            ))

            audio_file = None
            audio_error = ""
            tts_status = self.tts_engine.get_status()
            try:
                if not spoken_text:
                    raise SynthesisError("This reply has no text to speak.")
                audio_file = self.tts_engine.synthesize(
                    text=spoken_text,
                    reference_audio_path=(
                        self.default_ref_audio if reference_audio_path is None else reference_audio_path
                    ),
                    output_path=output_audio_path or f"data/audio/response_{uuid4().hex}.wav",
                    **self.tts_options,
                )
            except SynthesisError as exc:
                audio_error = str(exc)
            except OSError:
                audio_error = "Could not save the generated audio."
            finished = time.perf_counter()

            return {
                "user_text": user_text,
                "response_text": reply,
                "llm_backend": self.llm_client.backend,
                "audio_path": audio_file,
                "retrieved_memories": memories,
                "cited_memories": cited,
                "tts_engine": tts_status["name"] if audio_file else None,
                "tts_is_mock": bool(audio_file and tts_status["is_mock"]),
                "tts_message": audio_error,
                "metrics": {
                    "memory_retrieval_sec": round(recalled - started, 3),
                    "llm_generation_sec": round(generated - recalled, 3),
                    "tts_synthesis_sec": round(finished - generated, 3),
                    "total_sec": round(finished - started, 3),
                },
            }

    def process_audio_turn(
        self,
        input_audio_path: Union[str, Path],
        reference_audio_path: Optional[ReferenceAudio] = None,
    ) -> Dict[str, Any]:
        """Transcribe a WAV, then process the recognized text."""
        started = time.perf_counter()
        user_text = self.transcriber.transcribe(input_audio_path)
        if not user_text:
            raise InvalidRecording("No speech detected. Record again or type a message.")
        asr_seconds = time.perf_counter() - started
        result = self.process_text_turn(user_text, reference_audio_path=reference_audio_path)
        result["metrics"]["asr_transcription_sec"] = round(asr_seconds, 3)
        result["metrics"]["total_sec"] = round(result["metrics"]["total_sec"] + asr_seconds, 3)
        return result

    def clear_history(self) -> None:
        """Resets the active conversation history."""
        with self.turn_lock:
            self.chat_history.clear()

