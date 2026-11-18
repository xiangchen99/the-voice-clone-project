# Voice Studio

Record a reading prompt, play back the normalized take, and listen to the combined master recording.

## Sub-features

- `studio-record` records and normalizes a prompt.
- `studio-preview` plays the normalized take and the master.
- `studio-retake` replaces an existing take.
- `studio-select` saves recordings for Conversation.
- `studio-cli` lists prompts and ingests a WAV from the terminal.

## How to get to it (user POV)

Open **Voice Studio** in the dashboard. Use **Record Sample** on a reading prompt and stop when finished. The recorder also accepts a WAV through its terminal command.

## Driving it with a browser and terminal

Start with no recorded prompts. Allow microphone access on the loopback URL.

- Click `button[data-tab="tab-voice"]`. Wait for `#prompt-card-phonetic_01` and check `#master-ref-badge` for `Master Audio: Pending`.
- Click `#btn-rec-phonetic_01`, read the prompt aloud, and click it again to stop. Wait for `#p-status-phonetic_01` to show `Recorded & Normalized`.
- Check that `#audio-prev-phonetic_01` loads the normalized audio and that `#master-audio-player` loads the master. Use their play controls. Confirm nonempty WAVs at `data/audio/raw/phonetic_01.wav`, `data/audio/reference/phonetic_01_normalized.wav`, and `data/audio/reference/master_ref.wav`. Reopen the tab and check that both playback controls still load their audio.
- Under `#voice-reference-form`, select the normalized clip and save. Reload and verify the checkbox remains selected and the scratch `data/audio/reference/selection.json` lists that filename.
- Record a second take of the same prompt. Capture both takes' hashes and verify that the raw and normalized WAVs change and the master still plays.
- For the terminal path, use the project interpreter from a fresh scratch copy. Run `python -m src.audio.recorder` to see the prompt ID. Obtain a disposable spoken WAV at `data/take.wav`, then run `python -m src.audio.recorder ingest data/take.wav phonetic_01`. Open **Voice Studio** and check the same WAV files and playback controls.

## Gotchas

- The browser path needs an actual microphone input with recorded sound. Report a missing device or denied permission against this journey.
- Starting a recording on a second prompt discards the first prompt's active recording. Stop the active take before switching prompts.
