#!/usr/bin/env python3
"""Start an isolated Digital Self dashboard and verify a browser interview save."""

import argparse
import base64
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory, mkdtemp
import time
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[3]
ANSWER = "I built a cardboard lighthouse at summer camp."
HTTP = build_opener(ProxyHandler({}))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def wait_for(probe, description):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        result = probe()
        if result:
            return result
        time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for {description}")


@contextmanager
def process(command, cwd, log_path):
    with log_path.open("wb") as log:
        child = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        try:
            yield child
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def prepare(app):
    for directory in ("src", "web"):
        shutil.copytree(ROOT / directory, app / directory, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("run_ui.py", "scripts/guided_interview.py", "configs/persona.example.yaml"):
        target = app / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (app / "data").mkdir()
    marker = app.name
    (app / "configs/settings.yaml").write_text(
        f"llm:\n  backend: mock\n  model_name: {marker}\ntts:\n  engine: mock\n",
        encoding="utf-8",
    )
    return marker


def read(url, path):
    with HTTP.open(url + path, timeout=2) as response:
        return response.read()


def doctor(url, app, marker, child, evidence):
    def ready():
        require(child.poll() is None, f"Dashboard exited with {child.returncode}")
        try:
            status = json.loads(read(url, "/api/status"))
        except OSError:
            return None
        require(status["llm_model"] == marker, "Loopback port belongs to a different instance")
        return status if status["status"] == "ready" and status["tts_ready"] else None

    status = wait_for(ready, "dashboard initialization")
    require(status["tts_is_mock"] and status["memories_count"] == 0
            and status["voice_samples_count"] == 0, "Expected an empty dashboard in mock mode")
    require(b"Digital Self:" in read(url, "/"), "Dashboard HTML is missing")
    script = read(url, "/static/app.js")
    require(script == (app / "web/static/app.js").read_bytes(), "Dashboard JavaScript differs from this run's copy")
    questions = json.loads(read(url, "/api/questions"))
    require(len(questions) == 12 and not any(q["answered"] for q in questions),
            "Expected twelve unanswered interview questions")
    record = {
        "url": url,
        "pid": child.pid,
        "checkout": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "served_js_sha256": hashlib.sha256(script).hexdigest(),
        "status": status,
        "question_count": len(questions),
    }
    (evidence / "doctor.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Dashboard ready at {url} with scratch data in {app}", flush=True)


class Browser:
    def __init__(self, connection, evidence):
        self.connection = connection
        self.evidence = evidence
        self.request_id = 0
        self.command("session.new", capabilities={})
        self.context = self.command("browsingContext.getTree")["contexts"][0]["context"]

    def command(self, method, **params):
        self.request_id += 1
        self.connection.send(json.dumps({"id": self.request_id, "method": method, "params": params}))
        while True:
            response = json.loads(self.connection.recv(timeout=25))
            if response.get("id") == self.request_id:
                require(response["type"] == "success", f"{method}: {response}")
                return response["result"]

    def evaluate(self, expression):
        result = self.command("script.evaluate", expression=expression,
                              target={"context": self.context}, awaitPromise=False)
        require(result["type"] == "success", f"Browser evaluation failed: {result}")
        return result["result"].get("value")

    def log(self, **step):
        with (self.evidence / "interview.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(step) + "\n")

    def navigate(self, url):
        self.log(action="open dashboard", url=url)
        self.command("browsingContext.navigate", context=self.context, url=url, wait="interactive")

    def click(self, selector):
        self.log(action="click", selector=selector)
        # Resolve the visible target for native pointer input.
        position = json.loads(self.evaluate(f"""(() => {{
            const element = document.querySelector({json.dumps(selector)});
            if (!element || !element.checkVisibility()) throw Error("Hidden or absent target");
            element.scrollIntoView({{block: "center", inline: "center"}});
            const rect = element.getBoundingClientRect();
            return JSON.stringify({{x: Math.round(rect.x + rect.width / 2),
                                   y: Math.round(rect.y + rect.height / 2)}});
        }})()"""))
        self.command("input.performActions", context=self.context, actions=[{
            "type": "pointer", "id": "mouse", "actions": [
                {"type": "pointerMove", **position},
                {"type": "pointerDown", "button": 0},
                {"type": "pointerUp", "button": 0},
            ],
        }])

    def type_text(self, text):
        self.log(action="type", text=text)
        self.command("input.performActions", context=self.context, actions=[{
            "type": "key", "id": "keyboard", "actions": [
                {"type": action, "value": char}
                for char in text for action in ("keyDown", "keyUp")
            ],
        }])

    def screenshot(self, name):
        result = self.command("browsingContext.captureScreenshot", context=self.context, origin="document")
        (self.evidence / name).write_bytes(base64.b64decode(result["data"]))


def interview(url, app, evidence, browser):
    tab = 'button[data-tab="tab-interview"]'
    answer = "document.querySelector('#interview-answer-input').value"
    progress = "document.querySelector('#interview-progress-text').innerText"
    browser.navigate(url)
    browser.click(tab)
    wait_for(lambda: browser.evaluate(f"{progress}.startsWith('0 / 12 completed (0%)')"),
             "the empty interview")
    require(browser.evaluate("document.querySelector('#active-q-title').innerText")
            == "1. ORIGINS EARLY LIFE", "The first interview question differs from the fixture")
    require(browser.evaluate(answer) == "", "Interview editor already contains an answer")
    browser.click("#interview-answer-input")
    browser.type_text(ANSWER)
    require(browser.evaluate(answer) == ANSWER, "Answer did not reach the interview editor")
    browser.screenshot("interview-before-save.png")
    browser.click("#btn-save-question")
    wait_for(lambda: browser.evaluate("document.querySelector('#save-status-indicator').innerText === 'Saved!'"),
             "save confirmation")
    browser.log(observation="Saved!", progress=browser.evaluate(progress))
    browser.screenshot("interview-saved.png")

    browser.navigate(url)
    browser.click(tab)
    wait_for(lambda: browser.evaluate(answer) == ANSWER, "saved answer after reload")
    completed = browser.evaluate(progress)
    require(completed == "1 / 12 completed (8%)", f"Unexpected progress after reload: {completed}")
    browser.log(observation="reopened answer", answer=browser.evaluate(answer), progress=completed)
    browser.screenshot("interview-reopened.png")

    card = app / "data/biography/origins_early_life.md"
    require(card.read_text(encoding="utf-8").split("## My Personal Memory & Reflections", 1)[1].strip() == ANSWER,
            "Saved Markdown card differs from the answer")
    store = app / "data/memory_db/autobiography.json"
    documents = json.loads(store.read_text(encoding="utf-8"))
    require(any(d["id"] == "bio_origins_early_life" and d["content"] == ANSWER for d in documents),
            "Saved answer is missing from the memory index")
    shutil.copy2(card, evidence / card.name)
    shutil.copy2(store, evidence / store.name)


def drive(url, app, evidence):
    from websockets.sync.client import connect

    profile = app / "firefox-profile"
    profile.mkdir()
    log = evidence / "firefox.log"
    command = ["firefox", "--headless", "--no-remote", "--profile", str(profile),
               "--remote-debugging-port", "0", "about:blank"]
    with process(command, app, log) as child:
        def endpoint():
            require(child.poll() is None, f"Firefox exited with {child.returncode}")
            match = re.search(r"WebDriver BiDi listening on (ws://127\.0\.0\.1:\d+)", log.read_text(errors="replace"))
            return match.group(1) if match else None

        address = wait_for(endpoint, "Firefox WebDriver BiDi")
        with connect(address + "/session", proxy=None, max_size=16 * 1024 * 1024) as connection:
            interview(url, app, evidence, Browser(connection, evidence))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="leave the isolated dashboard open until Ctrl+C")
    args = parser.parse_args()
    evidence = Path(mkdtemp(prefix="digital-self-proof-", dir="/tmp"))
    print(f"Evidence directory: {evidence}", flush=True)
    try:
        with TemporaryDirectory(prefix="digital-self-app-", dir="/tmp") as directory:
            app = Path(directory)
            marker = prepare(app)
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            command = [sys.executable, "-u", "run_ui.py", "--host", "127.0.0.1", "--port", str(port)]
            url = f"http://127.0.0.1:{port}"
            with process(command, app, evidence / "dashboard.log") as child:
                doctor(url, app, marker, child, evidence)
                if args.serve:
                    print("Press Ctrl+C to stop the isolated dashboard.", flush=True)
                    while child.poll() is None:
                        time.sleep(0.5)
                    raise RuntimeError(f"Dashboard exited with {child.returncode}")
                drive(url, app, evidence)
    except KeyboardInterrupt:
        return 0 if args.serve else 130
    print(f"Interview save verified. Evidence retained at {evidence}", flush=True)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))
    sys.exit(main())
