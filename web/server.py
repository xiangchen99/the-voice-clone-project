"""Zero-dependency multi-threaded HTTP server and REST API for the Digital Self UI."""

import email
import http.server
import json
import mimetypes
import os
import re
import socketserver
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from scripts.guided_interview import QUESTIONS, save_memory_card
from src.audio.preprocessor import AudioPreprocessor
from src.audio.prompts import RECORDING_PROMPTS
from src.audio.recorder import VoiceRecorder
from src.memory.chat_parser import ChatLogParser
from src.pipeline.orchestrator import DigitalSelfOrchestrator


class DigitalSelfServer:
    """Manages system state and handles API calls for the web dashboard."""

    def __init__(self, settings_path: str = "configs/settings.yaml") -> None:
        self.settings_path = settings_path
        self.orchestrator = DigitalSelfOrchestrator(settings_path=settings_path)
        self.recorder = VoiceRecorder()
        self.preprocessor = AudioPreprocessor()
        # Ensure memory store loads existing biography cards on boot
        self.orchestrator.memory_store.load_biography_directory("data/biography")

    def get_status(self) -> Dict[str, Any]:
        """Returns overall system state."""
        memories = len(self.orchestrator.memory_store.documents)
        ref_files = list(Path("data/audio/reference").glob("*_normalized.wav"))
        has_master = (Path("data/audio/reference") / "master_ref.wav").exists()
        voice_count = len(ref_files) + (1 if has_master else 0)

        llm_cfg = self.orchestrator.settings.get("llm", {})
        tts_cfg = self.orchestrator.settings.get("tts", {})
        tts_status = getattr(
            self.orchestrator.tts_engine,
            "get_status",
            lambda: {"name": "mock", "is_mock": True, "ready": True, "message": "TTS engine status unavailable"},
        )()

        return {
            "status": "ready",
            "memories_count": memories,
            "voice_samples_count": voice_count,
            "llm_model": llm_cfg.get("model_name", "llama3.1:8b"),
            "tts_engine": tts_status.get("name", tts_cfg.get("engine", "xtts_v2")),
            "tts_is_mock": tts_status.get("is_mock", True),
            "tts_ready": tts_status.get("ready", False),
            "tts_message": tts_status.get("message", ""),
        }

    def get_interview_questions(self) -> List[Dict[str, Any]]:
        """Returns the 12 interview questions populated with any existing answers."""
        out = []
        bio_dir = Path("data/biography")
        for q in QUESTIONS:
            slug = q["slug"]
            card_path = bio_dir / f"{slug}.md"
            answer = ""
            answered = False

            if card_path.exists():
                try:
                    with open(card_path, "r", encoding="utf-8") as f:
                        raw = f.read()
                    if "## My Personal Memory & Reflections" in raw:
                        answer = raw.split("## My Personal Memory & Reflections", 1)[1].strip()
                        answered = bool(answer)
                    elif "\n\n" in raw:
                        answer = raw.split("\n\n", 1)[1].strip()
                        answered = bool(answer)
                except Exception:
                    pass

            out.append({
                "slug": slug,
                "category": q["category"],
                "question": q["question"],
                "answer": answer,
                "answered": answered,
            })
        return out

    def save_interview_card(self, slug: str, category: str, question: str, answer: str) -> Dict[str, Any]:
        """Saves memory card and updates memory store."""
        bio_dir = Path("data/biography")
        save_memory_card(category, slug, question, answer, bio_dir)
        # Update memory store
        self.orchestrator.memory_store.add_document(
            doc_id=f"bio_{slug}",
            content=answer,
            category=category,
            metadata={"slug": slug, "source": "interview"},
        )
        return {"success": True, "slug": slug}

    def get_voice_prompts(self) -> List[Dict[str, Any]]:
        """Returns reading prompts with status."""
        status = self.recorder.get_session_status()
        out = []
        for p in RECORDING_PROMPTS:
            pid = p["id"]
            is_recorded = status.get(pid, {}).get("processed", False)
            out.append({
                "id": pid,
                "category": p["category"],
                "prompt": p["prompt"],
                "intent": p["intent"],
                "recorded": is_recorded,
            })
        return out

    def handle_voice_upload(self, prompt_id: str, audio_bytes: bytes) -> Dict[str, Any]:
        """Processes uploaded voice sample and updates master reference."""
        if not audio_bytes:
            return {"success": False, "error": "No audio data received"}

        raw_p = Path("data/audio/raw") / f"{prompt_id}.wav"
        raw_p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(raw_p, "wb") as f:
                f.write(audio_bytes)

            # Preprocess and slice
            res = self.preprocessor.process_file(raw_p, "data/audio/reference", base_name=prompt_id)
            # Update master reference
            self.recorder.create_master_reference()
            return {"success": True, "result": res}
        except Exception as e:
            if raw_p.exists():
                try:
                    raw_p.unlink()
                except OSError:
                    pass
            return {"success": False, "error": str(e)}

    def chat_message(self, message: str) -> Dict[str, Any]:
        """Runs a text conversational turn."""
        turn = self.orchestrator.process_text_turn(message)
        # Convert audio_path to web-accessible URL
        audio_name = Path(turn["audio_path"]).name
        turn["audio_url"] = f"/api/audio/{audio_name}"
        return turn

    def chat_voice(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Runs a voice conversational turn from uploaded microphone audio."""
        if not audio_bytes:
            return {
                "user_text": "",
                "response_text": "No audio was captured. Please try recording again.",
                "audio_path": "",
                "retrieved_memories": [],
                "metrics": {},
            }

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            turn = self.orchestrator.process_audio_turn(tmp_path)
            audio_name = Path(turn["audio_path"]).name
            turn["audio_url"] = f"/api/audio/{audio_name}"
            return turn
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def import_chat(self, text: str, author: str) -> Dict[str, Any]:
        """Imports and analyzes chat logs."""
        parser = ChatLogParser(target_author=author if author else None)
        messages = parser.parse_whatsapp(text)
        if not messages:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            messages = [{"author": author or "User", "text": l} for l in lines]

        chunks = parser.chunk_messages_for_memory(messages)
        ts = int(time.time())
        for idx, chunk in enumerate(chunks):
            self.orchestrator.memory_store.add_document(
                doc_id=f"chat_import_{idx}_{ts}",
                content=chunk["content"],
                category="Chat Conversations",
                metadata={"source": "chat_import"},
            )

        profile = parser.extract_linguistic_profile(messages)
        return {"success": True, "chunk_count": len(chunks), "profile": profile}

    def get_persona(self) -> Dict[str, Any]:
        """Returns persona configuration dictionary."""
        return self.orchestrator.prompt_builder.persona

    def save_persona(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Updates persona configuration in configs/persona.yaml."""
        p_path = Path("configs/persona.yaml")
        existing = self.get_persona()

        existing.setdefault("identity", {})["name"] = data.get("name", existing.get("identity", {}).get("name", "User"))
        existing["identity"]["full_name"] = data.get("full_name", existing["identity"]["name"])

        traits = existing.setdefault("core_traits", {})
        if "openness" in data:
            traits["openness"] = data["openness"]
        if "conscientiousness" in data:
            traits["conscientiousness"] = data["conscientiousness"]
        if "agreeableness" in data:
            traits["agreeableness"] = data["agreeableness"]

        lexicon = existing.setdefault("lexicon", {})
        if "favorite_phrases" in data:
            lexicon["favorite_phrases"] = data["favorite_phrases"]
        if "avoid_words" in data:
            lexicon["avoid_words"] = data["avoid_words"]

        if "spoken_constraints" in data:
            existing.setdefault("communication_style", {})["spoken_constraints"] = data["spoken_constraints"]

        with open(p_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(existing, f, sort_keys=False)

        # Reload prompt builder
        self.orchestrator.prompt_builder = type(self.orchestrator.prompt_builder)(config_path=str(p_path))
        return {"success": True}


class RequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Request Handler routing static files and REST API requests."""

    server_instance: Optional[DigitalSelfServer] = None

    def log_message(self, format: str, *args: Any) -> None:
        """Quiet log handler."""
        # Pass or print concise access line
        return

    def _send_json(self, data: Any, status: int = 200) -> None:
        """Sends JSON response."""
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_file(self, file_path: Path, content_type: Optional[str] = None) -> None:
        """Serves a static file."""
        if not file_path.exists() or file_path.is_dir():
            self.send_error(404, "File Not Found")
            return

        ctype = content_type or mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        file_size = file_path.stat().st_size

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(file_size))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        if self.command != "HEAD":
            with open(file_path, "rb") as f:
                while chunk := f.read(65536):
                    self.wfile.write(chunk)

    def _read_body_json(self) -> Dict[str, Any]:
        """Parses JSON request body."""
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _parse_multipart_form(self) -> Tuple[Dict[str, str], Dict[str, bytes]]:
        """Parses multipart/form-data with pure standard library without deprecated cgi."""
        ctype = self.headers.get("Content-Type", "")
        if "boundary=" not in ctype:
            return {}, {}

        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}, {}
        body = self.rfile.read(length)

        fields: Dict[str, str] = {}
        files: Dict[str, bytes] = {}

        try:
            from email.policy import default
            msg = email.message_from_bytes(
                b"Content-Type: " + ctype.encode("latin1", errors="ignore") + b"\r\n\r\n" + body,
                policy=default,
            )
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if not name:
                    continue
                filename = part.get_filename()
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                if filename is not None:
                    files[name] = payload
                else:
                    fields[name] = payload.decode("utf-8", errors="replace")
            return fields, files
        except Exception:
            m = re.search(r'boundary=(?:"([^"]+)"|([^;\s]+))', ctype)
            if not m:
                return {}, {}
            boundary = (m.group(1) or m.group(2)).strip().encode("ascii", errors="ignore")
            delimiter = b"--" + boundary
            parts = body.split(delimiter)
            for part in parts:
                if not part or part == b"--\r\n" or part == b"--":
                    continue
                if b"\r\n\r\n" not in part:
                    continue
                header_raw, content_raw = part.split(b"\r\n\r\n", 1)
                if content_raw.endswith(b"\r\n"):
                    content_raw = content_raw[:-2]
                header_text = header_raw.decode("latin1", errors="ignore")
                m_name = re.search(r'name="([^"]+)"', header_text)
                m_filename = re.search(r'filename="([^"]+)"', header_text)
                if m_name:
                    field_name = m_name.group(1)
                    if m_filename:
                        files[field_name] = content_raw
                    else:
                        fields[field_name] = content_raw.decode("utf-8", errors="ignore")
            return fields, files

    def do_OPTIONS(self) -> None:
        """Handles CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_HEAD(self) -> None:
        """Handles HEAD requests using GET routing without sending the response body."""
        try:
            self._handle_get()
        except Exception as exc:
            self._send_json({"success": False, "error": str(exc)}, status=500)

    def do_GET(self) -> None:
        """Handles GET requests for static files and REST endpoints with error safety."""
        try:
            self._handle_get()
        except Exception as exc:
            self._send_json({"success": False, "error": str(exc)}, status=500)

    def _handle_get(self) -> None:
        """Internal router for GET and HEAD requests."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        server = self.server_instance
        if server is None:
            self.send_error(500, "Server backend not initialized")
            return

        # 1. Root & Static Files
        if path == "/" or path == "/index.html":
            self._send_file(Path("web/static/index.html"), "text/html")
            return

        if path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            file_path = Path("web/static") / rel_path
            self._send_file(file_path)
            return

        # 2. Audio Streaming Endpoint
        if path.startswith("/api/audio/"):
            filename = Path(path[len("/api/audio/"):]).name
            candidates = [
                Path("data/audio") / filename,
                Path("data/audio/reference") / filename,
                Path("data/audio/raw") / filename,
            ]
            for c in candidates:
                if c.exists():
                    self._send_file(c, "audio/wav")
                    return
            self.send_error(404, "Audio file not found")
            return

        # 3. REST API Endpoints
        if path == "/api/status":
            self._send_json(server.get_status())
            return

        if path == "/api/questions":
            self._send_json(server.get_interview_questions())
            return

        if path == "/api/prompts":
            self._send_json(server.get_voice_prompts())
            return

        if path == "/api/memories":
            self._send_json(server.orchestrator.memory_store.documents)
            return

        if path == "/api/persona":
            self._send_json(server.get_persona())
            return

        self.send_error(404, "Endpoint Not Found")

    def do_POST(self) -> None:
        """Handles POST REST API requests with top-level error protection."""
        try:
            self._handle_post()
        except Exception as exc:
            self._send_json({"success": False, "error": str(exc)}, status=500)

    def _handle_post(self) -> None:
        """Internal router for POST requests."""
        path = urllib.parse.urlparse(self.path).path
        server = self.server_instance
        if server is None:
            self.send_error(500, "Server backend not initialized")
            return

        if path == "/api/chat/message":
            body = self._read_body_json()
            msg = body.get("message", "")
            turn = server.chat_message(msg)
            self._send_json(turn)
            return

        if path == "/api/chat/voice":
            fields, files = self._parse_multipart_form()
            audio_data = files.get("audio") or (list(files.values())[0] if files else b"")
            if not audio_data:
                # Raw audio body fallback
                length = int(self.headers.get("Content-Length", 0))
                audio_data = self.rfile.read(length)

            turn = server.chat_voice(audio_data)
            self._send_json(turn)
            return

        if path == "/api/interview/save":
            body = self._read_body_json()
            res = server.save_interview_card(
                slug=body.get("slug", ""),
                category=body.get("category", "General"),
                question=body.get("question", ""),
                answer=body.get("answer", ""),
            )
            self._send_json(res)
            return

        if path == "/api/voice/upload":
            fields, files = self._parse_multipart_form()
            prompt_id = fields.get("prompt_id", "phonetic_01")
            audio_data = files.get("audio", b"")
            if not audio_data and files:
                audio_data = list(files.values())[0]
            res = server.handle_voice_upload(prompt_id, audio_data)
            status_code = 200 if res.get("success", False) else 400
            self._send_json(res, status=status_code)
            return

        if path == "/api/memories/search":
            body = self._read_body_json()
            q = body.get("query", "")
            results = server.orchestrator.memory_store.query(q, top_k=8)
            self._send_json(results)
            return

        if path == "/api/chat/import":
            body = self._read_body_json()
            res = server.import_chat(body.get("text", ""), body.get("author", ""))
            self._send_json(res)
            return

        if path == "/api/persona":
            body = self._read_body_json()
            res = server.save_persona(body)
            self._send_json(res)
            return

        self.send_error(404, "Endpoint Not Found")


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Multi-threaded HTTP server so audio streaming and chat turns do not block."""
    daemon_threads = True
    allow_reuse_address = True


def run_server(port: int = 8080, host: str = "127.0.0.1") -> None:
    """Starts the digital self web server."""
    backend_service = DigitalSelfServer()
    RequestHandler.server_instance = backend_service

    server_address = (host, port)
    httpd = ThreadedHTTPServer(server_address, RequestHandler)
    print("==========================================================")
    print(" Digital Self Studio Web Dashboard")
    print(f" URL: http://{host}:{port}")
    print(" Press Ctrl+C in terminal to stop.")
    print("==========================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run_server()

