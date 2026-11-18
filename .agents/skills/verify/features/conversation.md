# Conversation

Send typed or spoken messages, read the assistant's reply, and play its audio.

## Sub-features

- `conversation-text` sends from the button or the input's Enter key.
- `conversation-mic` records a draft for review before sending.
- `conversation-audio` loads the generated audio and exposes replay of the original reply audio.

## How to get to it (user POV)

Open the dashboard. **Conversation** is the default tab. Use **Send**, Enter in the message field, or the **Record** button.

## Driving it with a browser

Start a fresh dashboard. A voice turn also requires an enabled microphone and a usable faster-whisper model.

- Fill `#chat-text-input` with `Hello from the verification run.` and click `#btn-send-chat`. Check the user and assistant bubbles. Confirm `POST /api/chat/message` returned the rendered reply. In explicit mock mode the reply shows `Demo reply`.
- Fill the same field with `Can we try a second message?` and press Enter. Check for a second user bubble and new assistant reply. The input clears after a successful send.
- Click `#btn-record-chat`, allow the microphone, and say `I am testing the microphone.` Wait for `#recording-banner`, then click the same button to stop. Check that `POST /api/chat/transcribe` returns the editable transcript in `#chat-text-input`. Check that the recording added no chat messages.
- Edit the transcript, then click `#btn-send-chat`. Verify the corrected text is the new user bubble. Record again in silence. After you stop, the status should prompt you to retry or type and the chat should stay unchanged.
- Record the `src` on `#audio-playback` after each reply. Confirm each URL serves a WAV with nonzero frames. Click **Replay** on the earlier reply and verify the player loads its original URL. Expect `Demo chime` in explicit mock mode.
- With real engines, approve a synthetic interview card, ask about it, and confirm any source label in the reply maps to that card under **Memory Sources**. Try an unrelated personal question and inspect the reply for unsupported claims.

## Gotchas

- Mock TTS emits a chime. Verifying cloned speech requires neural TTS and a reference WAV in the isolated run.
- Microphone permissions and ASR model availability can block the spoken journey. Record the blocker against the voice entry point.
- If autoplay is blocked, use the audio player controls. Send-to-playback timing reflects when playback starts.
