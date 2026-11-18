"""Unit and integration tests for the web server and REST API."""

import json
import os
import shutil
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from web.server import DigitalSelfServer, RequestHandler, ThreadedHTTPServer


ROOT = Path(__file__).resolve().parents[1]


def enter_workspace(add_cleanup, *, static=False):
    scratch = tempfile.TemporaryDirectory()
    add_cleanup(scratch.cleanup)
    add_cleanup(os.chdir, Path.cwd())
    os.chdir(scratch.name)
    Path("configs").mkdir()
    shutil.copy(ROOT / "configs/persona.example.yaml", "configs")
    Path("configs/settings.yaml").write_text("llm:\n  backend: mock\ntts:\n  engine: mock\n")
    if static:
        shutil.copytree(ROOT / "web/static", "web/static")


class TestDigitalSelfServer(unittest.TestCase):

    def setUp(self):
        enter_workspace(self.addCleanup)
        self.server = DigitalSelfServer()

    def test_get_status(self):
        status = self.server.get_status()
        self.assertEqual(status["status"], "ready")
        self.assertIn("memories_count", status)
        self.assertIn("voice_samples_count", status)
        self.assertIn("llm_model", status)
        self.assertIn("tts_engine", status)
        self.assertIn("tts_is_mock", status)
        self.assertIn("tts_ready", status)

    def test_get_interview_questions_and_save(self):
        questions = self.server.get_interview_questions()
        self.assertEqual(len(questions), 12)
        self.assertIn("slug", questions[0])
        self.assertIn("category", questions[0])

        # Test saving
        res = self.server.save_interview_card(
            slug="test_slug_card",
            category="Work & Craft",
            question="What do you build?",
            answer="I build autonomous agents and neural pipelines.",
        )
        self.assertTrue(res["success"])
        # Clean up test card
        test_file = Path("data/biography/test_slug_card.md")
        if test_file.exists():
            test_file.unlink()

    def test_approval_is_preserved_after_reload_and_can_be_revoked(self):
        save = self.server.save_interview_card
        save("origins_early_life", "Origins", "Where?", "I built a cardboard lighthouse.", True)
        restarted = DigitalSelfServer()
        self.assertTrue(restarted.get_interview_questions()[0]["approved_for_chat"])
        store = restarted.orchestrator.memory_store
        self.assertEqual(store.query("lighthouse", approved_only=True)[0]["content"], "I built a cardboard lighthouse.")
        restarted.save_interview_card("origins_early_life", "Origins", "Where?", "I built a cardboard lighthouse.")
        self.assertEqual(DigitalSelfServer().orchestrator.memory_store.query("lighthouse", approved_only=True), [])

    def test_get_voice_prompts(self):
        prompts = self.server.get_voice_prompts()
        self.assertGreaterEqual(len(prompts), 10)
        self.assertIn("prompt", prompts[0])

    def test_chat_message(self):
        res = self.server.chat_message("What is your view on coding?")
        self.assertIn("response_text", res)
        self.assertIn("audio_url", res)
        self.assertTrue(res["audio_url"].startswith("/api/audio/"))

    def test_voice_reference_selection_reloads_empty_and_nonempty(self):
        directory = Path("data/audio/reference")
        directory.mkdir(parents=True, exist_ok=True)
        sample = directory / "phonetic_01_normalized.wav"
        sample.touch()
        self.assertEqual(self.server.save_voice_references([sample.name])["selected"], [sample.name])
        self.assertEqual(DigitalSelfServer().selected_voice_references(), [sample])
        with self.assertRaisesRegex(ValueError, "Voice Studio"):
            self.server.save_voice_references(["../response.wav"])
        self.assertEqual(self.server.get_voice_references()["selected"], [sample.name])
        self.server.save_voice_references([])
        self.assertEqual(DigitalSelfServer().selected_voice_references(), [])

    def test_import_chat(self):
        sample_chat = """[10/12/23, 1:00 PM] User: First message about systems engineering.
[10/12/23, 1:05 PM] User: Second message about autonomous robotics.
"""
        res = self.server.import_chat(sample_chat, "User")
        self.assertTrue(res["success"])
        self.assertGreaterEqual(res["chunk_count"], 1)
        self.assertIn("profile", res)


class TestWebServerHttpIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        enter_workspace(cls.addClassCleanup, static=True)
        backend = DigitalSelfServer()
        RequestHandler.server_instance = backend
        cls.addClassCleanup(setattr, RequestHandler, "server_instance", None)
        cls.httpd = ThreadedHTTPServer(("127.0.0.1", 0), RequestHandler)
        cls.port = cls.httpd.server_port
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def test_get_root_html(self):
        url = f"http://127.0.0.1:{self.port}/"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            content = resp.read().decode("utf-8")
            self.assertIn("Digital Self Studio", content)
            self.assertIn("<!DOCTYPE html>", content)

    def test_get_static_css(self):
        url = f"http://127.0.0.1:{self.port}/static/styles.css"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            content = resp.read().decode("utf-8")
            self.assertIn("--bg-main", content)

    def test_get_api_status(self):
        url = f"http://127.0.0.1:{self.port}/api/status"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "ready")

    def test_post_chat_message(self):
        url = f"http://127.0.0.1:{self.port}/api/chat/message"
        payload = json.dumps({"message": "Hello digital twin!"}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("response_text", data)
            self.assertIn("audio_url", data)

    def test_head_request_supported(self):
        # HEAD request should succeed without 501 Unsupported Method error
        url = f"http://127.0.0.1:{self.port}/api/status"
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("Content-Type"), "application/json")
            self.assertEqual(len(resp.read()), 0)

    def test_post_voice_upload_valid_wav_and_master(self):
        # Generate valid test WAV bytes
        import io, math, struct, wave
        buf = io.BytesIO()
        sr = 24000
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            samples = [int(0.2 * 32767.0 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(sr)]
            wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        wav_bytes = buf.getvalue()

        # Preserve any existing master_ref.wav to avoid clobbering user recordings
        master_file = Path("data/audio/reference/master_ref.wav")
        original_master_bytes = master_file.read_bytes() if master_file.exists() else None

        raw_file = Path("data/audio/raw/unit_test_prompt.wav")
        norm_file = Path("data/audio/reference/unit_test_prompt_normalized.wav")

        try:
            boundary = "----WebKitFormBoundaryUnitTest123"
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="prompt_id"\r\n\r\n'
                f"unit_test_prompt\r\n"
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="audio"; filename="unit_test_prompt.wav"\r\n'
                f"Content-Type: audio/wav\r\n\r\n"
            ).encode("utf-8") + wav_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

            url = f"http://127.0.0.1:{self.port}/api/voice/upload"
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(data.get("success"))
                self.assertIn("result", data)

            # Verify created files exist
            self.assertTrue(raw_file.exists())
            self.assertTrue(norm_file.exists())
            self.assertTrue(master_file.exists())

            # Test HEAD on master_ref.wav
            head_url = f"http://127.0.0.1:{self.port}/api/audio/master_ref.wav"
            head_req = urllib.request.Request(head_url, method="HEAD")
            with urllib.request.urlopen(head_req) as head_resp:
                self.assertEqual(head_resp.status, 200)
                self.assertEqual(head_resp.headers.get("Content-Type"), "audio/wav")
        finally:
            # Cleanup test files and restore original master reference
            if raw_file.exists():
                raw_file.unlink()
            if norm_file.exists():
                norm_file.unlink()
            for chunk in Path("data/audio/reference").glob("unit_test_prompt_chunk_*.wav"):
                chunk.unlink()
            if original_master_bytes is not None:
                master_file.write_bytes(original_master_bytes)

    def test_post_voice_upload_invalid_non_riff_returns_clean_error(self):
        # Non-RIFF payload (e.g. raw WebM header or corrupt text)
        invalid_bytes = b"\x1a\x45\xdf\xa3webm_fake_audio_stream_data_without_riff"
        boundary = "----WebKitFormBoundaryInvalid123"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="prompt_id"\r\n\r\n'
            f"invalid_test_prompt\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="audio"; filename="invalid.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + invalid_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        url = f"http://127.0.0.1:{self.port}/api/voice/upload"
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.assertFalse(data.get("success"))
                self.assertIn("error", data)
        except urllib.error.HTTPError as err:
            with err:
                self.assertEqual(err.code, 400)
                data = json.loads(err.read().decode("utf-8"))
                self.assertFalse(data.get("success"))
                self.assertIn("error", data)

    def test_transcript_draft_does_not_generate_before_send(self):
        backend = RequestHandler.server_instance
        before = list(backend.orchestrator.chat_history)
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/chat/transcribe",
            data=b"microphone fixture",
            headers={"Content-Type": "audio/wav"},
        )
        with patch.object(backend.orchestrator.transcriber, "transcribe", return_value="Uncorrected words"):
            with urllib.request.urlopen(request, timeout=5) as response:
                self.assertEqual(json.load(response)["user_text"], "Uncorrected words")
        self.assertEqual(backend.orchestrator.chat_history, before)

        with patch.object(backend.orchestrator.transcriber, "transcribe", return_value=""):
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=5)
            with error.exception as response:
                self.assertEqual(response.code, 422)
                self.assertIn("No speech detected", json.load(response)["error"])
        self.assertEqual(backend.orchestrator.chat_history, before)

        with patch.object(backend.orchestrator.transcriber, "transcribe", return_value="Legacy spoken turn"):
            request.full_url = f"http://127.0.0.1:{self.port}/api/chat/voice"
            with urllib.request.urlopen(request, timeout=5) as response:
                turn = json.load(response)
                self.assertEqual(turn["user_text"], "Legacy spoken turn")
                self.assertTrue(turn["audio_url"].startswith("/api/audio/"))

    def test_empty_multipart_returns_without_reading_body_twice(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/chat/transcribe",
            data=b"--empty--\r\n",
            headers={"Content-Type": "multipart/form-data; boundary=empty"},
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=5)
        with error.exception as response:
            self.assertEqual(response.code, 422)
            self.assertIn("No audio was captured", json.load(response)["error"])

    def test_post_memories_search_and_persona(self):
        # Test memories search
        url_search = f"http://127.0.0.1:{self.port}/api/memories/search"
        payload = json.dumps({"query": "robotics"}).encode("utf-8")
        req_search = urllib.request.Request(url_search, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req_search) as resp:
            self.assertEqual(resp.status, 200)
            results = json.loads(resp.read().decode("utf-8"))
            self.assertIsInstance(results, list)

        url_persona = f"http://127.0.0.1:{self.port}/api/persona"
        original = Path("configs/persona.yaml")
        original.write_text("identity:\n  name: Existing Persona\n")
        reply = "I'd try the simplest thing first."
        payload = json.dumps({"name": "Autonomous Tester", "approved_replies": [reply]}).encode("utf-8")
        request = urllib.request.Request(url_persona, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertTrue(json.load(response)["success"])
        with urllib.request.urlopen(url_persona, timeout=5) as response:
            persona = json.load(response)
        self.assertEqual(persona["identity"]["name"], "Autonomous Tester")
        self.assertEqual(persona["communication_style"]["approved_replies"], [reply])
        self.assertIn("Existing Persona", original.read_text())
        self.assertEqual(DigitalSelfServer().get_persona()["communication_style"]["approved_replies"], [reply])


if __name__ == "__main__":
    unittest.main()


