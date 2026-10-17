"""Unit and integration tests for the web server and REST API."""

import json
import socket
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from web.server import DigitalSelfServer, RequestHandler, ThreadedHTTPServer


class TestDigitalSelfServer(unittest.TestCase):

    def setUp(self):
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

    def test_get_voice_prompts(self):
        prompts = self.server.get_voice_prompts()
        self.assertGreaterEqual(len(prompts), 10)
        self.assertIn("prompt", prompts[0])

    def test_chat_message(self):
        res = self.server.chat_message("What is your view on coding?")
        self.assertIn("response_text", res)
        self.assertIn("audio_url", res)
        self.assertTrue(res["audio_url"].startswith("/api/audio/"))

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
        # Pick a free ephemeral port
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        cls.port = s.getsockname()[1]
        s.close()

        backend = DigitalSelfServer()
        RequestHandler.server_instance = backend
        cls.httpd = ThreadedHTTPServer(("127.0.0.1", cls.port), RequestHandler)
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

    def test_post_chat_voice(self):
        import io, math, struct, wave
        buf = io.BytesIO()
        sr = 24000
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            samples = [int(0.1 * 32767.0 * math.sin(2 * math.pi * 300 * i / sr)) for i in range(sr)]
            wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        wav_bytes = buf.getvalue()

        url = f"http://127.0.0.1:{self.port}/api/chat/voice"
        req = urllib.request.Request(url, data=wav_bytes, headers={"Content-Type": "audio/wav"}, method="POST")
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("response_text", data)
            self.assertIn("audio_url", data)

    def test_post_memories_search_and_persona(self):
        # Test memories search
        url_search = f"http://127.0.0.1:{self.port}/api/memories/search"
        payload = json.dumps({"query": "robotics"}).encode("utf-8")
        req_search = urllib.request.Request(url_search, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req_search) as resp:
            self.assertEqual(resp.status, 200)
            results = json.loads(resp.read().decode("utf-8"))
            self.assertIsInstance(results, list)

        # Test persona update while preserving original config file
        p_path = Path("configs/persona.yaml")
        orig_persona = p_path.read_text(encoding="utf-8") if p_path.exists() else None
        try:
            url_persona = f"http://127.0.0.1:{self.port}/api/persona"
            payload_p = json.dumps({"name": "Autonomous Tester"}).encode("utf-8")
            req_p = urllib.request.Request(url_persona, data=payload_p, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req_p) as resp:
                self.assertEqual(resp.status, 200)
                res_p = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(res_p.get("success"))
        finally:
            if orig_persona is not None:
                p_path.write_text(orig_persona, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()


