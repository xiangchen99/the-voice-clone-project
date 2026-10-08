# Digital Self: Local Voice & Personality Preservation

A self-contained, 100% locally-run digital twin architecture designed to preserve your personality, vocal timbre, and autobiographical memories on consumer hardware (tested on **NVIDIA RTX 4080 16GB VRAM**).

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       Guided Voice Recorder           |
                      | (Phonetic prompts + LUFS -20 + VAD)   |
                      +-------------------+-------------------+
                                          |
                                          v
+-----------------------+     +-----------------------+     +-------------------------+
|   User Speech (Mic)   | --> |   faster-whisper ASR  | --> |     LLM Persona Brain   |
+-----------------------+     +-----------------------+     |   (Ollama / Llama 3.1)  |
                                                                +-----------+-----------+
+-----------------------+     +-----------------------+                     |
|  Autobiographical RAG | --> |  Episodic Memories    | --------------------+
|  (Interviews & Chats) |     |  (BM25 / ChromaDB)    |                     |
+-----------------------+     +-----------------------+                     v
                                                                +-------------------------+
+-----------------------+                                       |     Neural Voice TTS    |
| Cloned Voice Output   | <------------------------------------ |   (XTTS-v2 / F5-TTS)    |
+-----------------------+                                       +-------------------------+
```

---

## Key Capabilities

1. **Acoustic Voice Cloning**:
   - Guided recording catalog with phonetically balanced reading scripts (pangrams, Harvard sentences, casual dialogue).
   - Audio preprocessor: EBU R128 loudness normalization (-20 dBFS), silence trimming, and slice chunking.
   - Zero-shot neural voice cloning interface supporting Coqui XTTS-v2 and F5-TTS.

2. **Persona Matrix & Cognitive Stance**:
   - Structured persona matrix (`configs/persona.yaml`) defining Big Five traits, characteristic phrases, forbidden words, and speech rules.
   - Built-in speech text sanitizer: automatically strips markdown symbols, hashtags, and emojis to prevent TTS artifacts.

3. **Autobiographical Memory Engine (RAG)**:
   - Guided CLI interview (`scripts/guided_interview.py`) covering 12 life dimensions: origins, turning points, craft, philosophy, and quirks.
   - Chat log parser (`src/memory/chat_parser.py`): ingests WhatsApp, Discord, or Telegram exports, extracts linguistic statistics, and chunks conversation histories.
   - Persistent episodic memory index with semantic retrieval.

4. **Speech-to-Speech Web Interface**:
   - Real-time conversation tab with microphone audio streaming and voice output playback.
   - Memory Studio tab to view, search, and add personal stories.
   - Voice Studio tab with reading scripts and calibration tools.

---

## Quickstart Guide

### 1. Environment Setup
Run the automated installer script:
```bash
./scripts/setup_environment.sh
source .venv/bin/activate
```

### 2. Capture Your Life Story (Interview)
Launch the interactive terminal questionnaire to record your core memories:
```bash
python scripts/guided_interview.py
```
*Your responses are saved as structured Markdown files in `data/biography/` and automatically indexed.*

### 3. Record Your Voice Samples
Review the prompt catalog and record reference audio:
```bash
python -m src.audio.recorder
```
Ingest any recorded clean WAV file:
```bash
python -m src.audio.recorder ingest <path_to_audio.wav> phonetic_01
```

### 4. Launch Local LLM (Ollama)
Ensure your local Ollama daemon is running with an 8B model:
```bash
ollama run llama3.1:8b
```

### 5. Launch the Web UI
Start the interactive Gradio dashboard:
```bash
python web/app.py
```
Open your browser at `http://127.0.0.1:7860`.

---

## Hardware Budget (16GB VRAM RTX 4080)

| Component | Model | VRAM Usage | Latency (TTFT) |
|---|---|---|---|
| **LLM Brain** | Llama-3.1-8B-Instruct (Q4_K_M) | ~5.2 GB | ~300ms |
| **Voice Cloning** | Coqui XTTS-v2 | ~3.4 GB | ~250ms (Streaming) |
| **Speech ASR** | faster-whisper (base.en / float16) | ~1.1 GB | ~150ms |
| **Total Pipeline** | Concurrent VRAM | **~9.7 GB / 16.0 GB** | **< 1.0s turnaround** |

---

## Repository Structure

```
the-voice-clone-project/
├── configs/
│   ├── persona.yaml          # Personality matrix, tone, vocabulary rules
│   └── settings.yaml         # Audio, model, memory, and hardware parameters
├── data/
│   ├── audio/                # Raw and normalized reference audio clips
│   ├── biography/            # Structured autobiographical memory cards
│   ├── chat_logs/            # Exported chat histories
│   └── memory_db/            # Persistent vector memory index
├── src/
│   ├── audio/
│   │   ├── preprocessor.py   # Audio normalization, resampling, slicing
│   │   ├── prompts.py        # Reading scripts catalog
│   │   └── recorder.py       # Recording manager and master ref generator
│   ├── memory/
│   │   ├── chat_parser.py    # Chat export parser and linguistic profiler
│   │   ├── store.py          # Vector memory storage and BM25 index
│   │   └── retriever.py      # Autobiographical context retrieval
│   ├── brain/
│   │   ├── prompt_builder.py # Persona prompt constructor and TTS sanitizer
│   │   └── llm_client.py     # Local LLM client with streaming chunker
│   ├── voice/
│   │   └── tts_engine.py     # Neural voice cloning interface (XTTS / Mock)
│   ├── asr/
│   │   └── transcriber.py    # faster-whisper integration
│   └── pipeline/
│       └── orchestrator.py   # Full voice-in -> think -> voice-out loop
├── web/
│   └── app.py                # Gradio multi-tab dashboard
├── scripts/
│   ├── setup_environment.sh  # Automated setup script
│   └── guided_interview.py   # Terminal autobiographical questionnaire
├── tests/                    # Comprehensive unit and integration test suite
└── requirements.txt
```

---

## Running Automated Tests

```bash
python3 -m unittest discover tests -v
```
All 13 test suites verify audio math, memory retrieval, chat parsing, prompt generation, and end-to-end pipeline coordination.

