# Life Interview

Answer guided questions and come back to saved reflections later. Saving an answer also adds it to the dashboard's memory index.

## Sub-features

- `interview-select` opens a question from the list or advances with **Next Question**.
- `interview-save` saves the active answer and updates completion. The source checkbox controls whether Conversation can use it.
- `interview-reopen` retrieves the saved answer after a reload.
- `interview-cli` saves a card from the terminal questionnaire.

## How to get to it (user POV)

Open **Life Interview** in the dashboard. Select a prompt in the left-hand list or choose **Next Question** in the editor. For the terminal questionnaire, run `python scripts/guided_interview.py` from the scratch copy.

## Driving it with a browser and terminal

Start with the twelve unanswered questions and the memory index empty.

- Open `button[data-tab="tab-interview"]`. Check `#active-q-title` for `ORIGINS EARLY LIFE` and `#interview-progress-text` for `0 / 12 completed (0%)`.
- Click the list item containing `family and roots` under `#questions-list-container`. Check the title for `FAMILY AND ROOTS`. Click `#btn-next-question` and check for `TURNING POINTS`. Select the list item containing `origins early life` again.
- Type `I built a cardboard lighthouse at summer camp.` into `#interview-answer-input`. Click `#btn-save-question` and wait for `Saved!` in `#save-status-indicator`.
- Reload the page and open **Life Interview** again. Check the exact answer and `1 / 12 completed (8%)`. Read `data/biography/origins_early_life.md` and confirm the answer in `data/memory_db/autobiography.json` under `bio_origins_early_life`. Select `#interview-chat-approval`, save, and reload. Verify the checkbox stays selected and the card contains `approved_for_chat: true`. Clear and save to revoke it.
- To automate the dashboard save and reopen, run `.venv/bin/python .agents/skills/verify/verify.py` from the checkout root. The helper opens the default first question.
- For the terminal entry, use a scratch copy without an origins card. Run `python scripts/guided_interview.py` from that directory with the project interpreter. Enter the same sentence for the first prompt, type `DONE` on a new line, and end input. Capture the displayed save path and read the card. Reload the dashboard's interview tab to verify the answer.

## Gotchas

- Tab data loads asynchronously. Wait for the populated list after opening the tab or reloading.
- The CLI writes the card. The running server picks up the CLI-written answer when you reopen the interview. Its memory index loads CLI-written cards at startup.
