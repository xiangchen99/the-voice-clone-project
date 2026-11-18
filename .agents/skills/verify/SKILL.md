---
name: verify
description: Verify Digital Self Studio's local web dashboard in a browser. Use for changes to conversation, life interviews, voice recording, memory search and chat import, or the persona editor.
---

# Verify Digital Self Studio

Read the [feature map](features/README.md) and open the journey that matches the change. The browser dashboard is the primary interface. The guided interview and audio recorder also have terminal entry points.

## Launch

Run from the repository root. Use the project's Python environment with PyYAML. The automated browser journey also requires Firefox on `PATH` and Python `websockets` 15 or later.

```bash
.venv/bin/python .agents/skills/verify/verify.py
```

This runs the Life Interview journey. To drive a different journey, start a fresh dashboard and leave the helper running in its own terminal.

```bash
.venv/bin/python .agents/skills/verify/verify.py --serve
```

The helper runs `python -u run_ui.py --host 127.0.0.1 --port <allocated-port>` from a temporary copy of the current checkout's app files. Both commands print the dashboard URL, the scratch directory, and the evidence directory. Wait for `Dashboard ready` before opening the URL. The dashboard has no sign-in.

Every run starts with empty data. The app creates its own storage, supplies the interview questions and reading prompts, and loads the example persona. The scratch settings select the app's existing mock LLM and TTS backends. Each run gets its own data directory, loopback port, and automated browser profile.

## Doctor

The helper runs a read-only check before it permits any browser actions. It checks the child process, matches the per-run engine label from `/api/status`, compares served JavaScript to the copied build, and verifies the fresh interview catalog. Read `doctor.json` for the PID, URL, expected engine label, checkout revision, and served JavaScript hash. The source copy includes local edits.

If an interactive run looks off, first check the PID from `doctor.json` with `kill -0`. Run `curl --noproxy '*' --fail --max-time 3 "$url/api/status"` with `url` set to the printed dashboard URL. Match `llm_model` to `status.llm_model` in `doctor.json` and check for `"status": "ready"`. Counts change as the journey adds data.

## Drive

Use the selectors in the feature map. With `--serve`, open the printed URL in a dedicated browser session. For an interactive Firefox profile, set `app` to the printed scratch directory and `url` to the dashboard URL, then run the following in another terminal.

```bash
mkdir "$app/manual-firefox-profile"
firefox --no-remote --profile "$app/manual-firefox-profile" "$url"
```

For automation, use your browser driver with the printed URL and a fresh profile. The interview helper uses Firefox WebDriver BiDi for pointer and keyboard actions, DOM observations, and screenshots. [Firefox Remote Agent](https://firefox-source-docs.mozilla.org/remote/Security.html)

## Evidence

Save evidence in the printed `/tmp/digital-self-proof-*` directory. Record the feature and entry point, the controls you used, and the observed result. Capture the draft or input action and the resulting visible state. For a write, reopen the result and copy the corresponding scratch file into evidence before teardown. Use the dashboard's controls for actions and the network log or persisted files to confirm results.

The automated interview journey writes `interview.jsonl`, screenshots before save, after save, and after reload, the saved Markdown card, and the memory index. Both modes retain the doctor result and server log. Keep mock-backend claims to the dashboard behavior they exercise. A test that needs transcription, a local model response, or cloned speech needs the matching ASR, LLM, or neural TTS prerequisites.

## Cleanup

The automated command stops its Firefox and server processes and removes the scratch app on exit, including on failure. For an interactive run, close its browser first, then interrupt the helper with Ctrl+C. If running in a managed shell, send SIGTERM to the helper process you started. The proof directory survives either cleanup. Confirm the expected evidence files still exist.

## Helpers

Execute `./.agents/skills/verify/verify.py` directly when the project virtual environment is active. The commands under Launch select that interpreter explicitly.
