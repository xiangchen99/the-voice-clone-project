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
   - Zero-shot neural voice cloning with Coqui XTTS-v2.

2. **Persona Matrix & Cognitive Stance**:
   - Structured persona matrix (`configs/persona.local.yaml` for dashboard edits) defining Big Five traits, characteristic phrases, forbidden words, and speech rules.
   - Built-in speech text sanitizer: automatically strips markdown symbols, hashtags, and emojis to prevent TTS artifacts.

3. **Autobiographical Memory Engine (RAG)**:
   - Guided CLI interview (`scripts/guided_interview.py`) covering 12 life dimensions: origins, turning points, craft, philosophy, and quirks.
   - Chat log parser (`src/memory/chat_parser.py`): ingests WhatsApp, Discord, or Telegram exports, extracts linguistic statistics, and chunks conversation histories.
   - Persistent episodic memory index with semantic retrieval.

4. **Speech-to-Speech Web Interface**:
   - Conversation tab with microphone recording, editable transcripts, and cloned-voice playback.
   - Memory Studio tab to view, search, and add personal stories.
   - Voice Studio tab with reading scripts and calibration tools.

---

## Quickstart Guide

Run `./run.sh` from the repository root to set up the Python 3.11 environment with CUDA PyTorch and start the dashboard. Open `http://127.0.0.1:8080` in your browser. Later runs reuse the environment. Dashboard options can be passed to the script, for example `./run.sh --port 8081`.

### 1. Environment Setup
For command-line tools, set up the environment and activate it:
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

### 5. Launch the Web UI Dashboard
To start the dashboard directly when the environment is already set up:
```bash
python3 run_ui.py
```
Open `http://127.0.0.1:8080` in your browser.

1. In **Persona Matrix**, add replies you wrote or approved, one per line. Save examples of how you give advice, disagree, joke, and express uncertainty.
2. In **Life Interview**, select **Use this answer as a source in Conversation** for each card you want the clone to cite. Save each card. Existing cards start with this permission off.
3. In **Voice Studio**, record and review your clips. Select the WAVs to use for Conversation and save the selection. The local selection takes precedence over `tts.reference_audio_path` in `configs/settings.yaml`.
4. In **Conversation**, record, stop, and review the transcript. Edit it in the message field and select **Send**. Use **Replay** on a reply to hear its original audio. A reply can cite approved cards under **Memory Sources**.

When transcription fails, record again or type. If text generation fails, the draft stays ready to retry. If voice synthesis fails, the app keeps the reply text and displays the audio error. For a UI demo, explicitly set `llm.backend` and `tts.engine` to `mock` in the settings. Demo replies and chimes are labeled.

### Compare your voice references

Use at least two clean recordings of your own voice. The first clip is the single-reference candidate. All supplied clips make up the other candidate.

```bash
.venv/bin/python -m scripts.compare_voice \
  data/audio/reference/phonetic_01_normalized.wav \
  data/audio/reference/phonetic_02_normalized.wav
```

The command generates three A/B pairs in a local, Git-ignored directory. Listen to them before opening `reveal.txt`. Judge voice likeness and naturalness separately, then save your preferred references in **Voice Studio**.

### Check a real conversation

Use a local Ollama model, faster-whisper weights, and XTTS-v2 with your selected recordings. Try an everyday opinion, a known personal fact, a follow-up, a personal question with no supporting memory, and a typed continuation. Correct one transcript and replay a reply.

Compare the same questions with and without your approved style examples. Keep the other settings and source cards the same, and use a fresh app session for each comparison. Judge whether you would say each reply separately from whether its audio sounds like you. Check that the known memory is accurate and the clone acknowledges the information it lacks. The **Send to playback** metric measures time until the browser starts playing each new reply. Record the first turn after startup and subsequent turns separately.

---

## Local runtime

The dashboard waits for the full local model reply and then synthesizes the audio. Conversation shows server processing and browser playback timing for each turn. The first turn may also load model weights.

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

