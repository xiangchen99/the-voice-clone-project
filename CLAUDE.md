# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

"Digital Self": a locally-run digital twin. Speech in → ASR → memory retrieval → persona-conditioned local LLM → cloned-voice TTS out. Targets an RTX 4080 (16 GB) on WSL2, but every heavy component has a CPU/mock fallback so the code and tests run without GPU, model weights, or a running Ollama.

## Commands

All commands run from the repo root; modules import as `src.*`, `web.*`, `scripts.*`, so the root must be on `sys.path`.

- Setup (uv, Python 3.11 `.venv`, CUDA 12.4 torch, then `requirements.txt`): `./scripts/setup_environment.sh && source .venv/bin/activate`
- Web UI: `python3 run_ui.py [--port 8080] [--host 127.0.0.1]` — auto re-execs into `.venv/bin/python` if it exists.
- LLM backend: `ollama run llama3.1:8b` (endpoint/model in `configs/settings.yaml`).
- Interview CLI: `python scripts/guided_interview.py` (writes cards to `data/biography/`).
- Recorder CLI: `python -m src.audio.recorder` / `python -m src.audio.recorder ingest <file.wav> <prompt_id>`
- All tests: `python3 -m unittest discover tests -v` (tests are `unittest`-style)
- Single test: `python3 -m unittest discover -s tests -p test_server.py -k TestDigitalSelfServer.test_get_status -v` — `tests/` has no `__init__.py`, so dotted `tests.test_x...` names don't import.

No linter/formatter config is checked in.

## Architecture

**Pipeline** — `src/pipeline/orchestrator.py` `DigitalSelfOrchestrator` is the hub. Its constructor reads `configs/settings.yaml` and wires every component; `process_text_turn` runs retrieve memories → `PromptBuilder.build_system_prompt` → `LocalLLMClient.generate_response` → `PromptBuilder.sanitize_for_tts` → `tts_engine.synthesize`, appends to in-memory `chat_history`, and returns a dict with text, audio path, recalled memories, TTS status, and per-stage timing `metrics`. `process_audio_turn` prepends ASR. The web server and tests depend on this return-dict shape.

**Graceful degradation is a design rule** — keep it when editing components:
- `src/voice/tts_engine.py`: `get_tts_engine()` returns `XTTSv2Engine` (lazy-loads Coqui `TTS`) or falls back to `MockTTSEngine` (synthetic tone WAV). Engines expose `get_status()` (`name`, `is_mock`, `ready`, `message`) that the UI surfaces.
- `src/brain/llm_client.py`: Ollama `/api/chat` or OpenAI-compatible endpoint via stdlib `urllib`; on connection errors (or unknown backend) returns a deterministic mock response.
- `src/asr/transcriber.py`: lazy faster-whisper load, retries on CPU/int8 if CUDA fails.
- `src/audio/preprocessor.py`: numpy optional; stdlib `wave`/`array` paths exist.
Heavy imports (torch, TTS, faster_whisper, numpy) are deferred/guarded so the UI and tests run on a bare interpreter.

**Memory** — `src/memory/store.py` `MemoryStore` is a JSON file (`data/memory_db/<collection>.json`) with a hand-written BM25 ranker; ChromaDB/sentence-transformers are in requirements but not used yet. `load_biography_directory` indexes the Markdown cards in `data/biography/`. `retriever.py` formats results into prompt context. `chat_parser.py` parses WhatsApp/Discord/Telegram exports into chunks + linguistic stats.

**Persona** — `PromptBuilder` loads `configs/persona.yaml`, falling back to `configs/persona.example.yaml`. The TTS sanitizer strips markdown/emoji/hashtags so speech output stays clean.

**Web** — `web/server.py` is the live UI: stdlib `http.server` (threaded) with `DigitalSelfServer` (state/API logic, owns one orchestrator) and `RequestHandler` (routing in `_handle_get`/`_handle_post`, `/api/*` JSON endpoints, hand-rolled multipart parsing, serves `web/static/` and generated audio at `/api/audio/<name>`). It reuses `scripts/guided_interview.QUESTIONS`/`save_memory_card`. `web/static/` is a vanilla JS/CSS single page. `web/app.py` is an older Gradio dashboard (README still describes it) and is not launched by `run_ui.py`.

**Config & data** — `configs/settings.yaml` holds hardware/audio/memory/llm/tts/asr settings; components read it with `.get()` defaults, so missing keys are tolerated. Paths are relative to the repo root (`data/...`). `data/` holds personal content (biography cards, chat logs, reference voice `data/audio/reference/master_ref.wav`, memory DB); tests and the server touch these real directories, so be careful not to clobber user data.
