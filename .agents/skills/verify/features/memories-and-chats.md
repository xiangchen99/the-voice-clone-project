# Memories & Chats

Browse stored memories, find a card by text, or paste a chat export to add searchable conversation blocks.

## Sub-features

- `memories-browse` shows the stored cards or the empty state.
- `memories-import` ingests a chat export and shows its vocabulary and chunk count.
- `memories-search` searches with the button or Enter and handles matches and misses.
- `memories-clear` restores the full list after clearing the query and submitting again.

## How to get to it (user POV)

Open **Memories & Chats**. Use the search box on the left or paste a chat export into the importer on the right. The optional author field selects the author.

## Driving it with a browser

Start with no memories. Use a disposable chat export.

- Click `button[data-tab="tab-memories"]` and wait for `#memory-search-results` to show `No matching memories found.`
- Fill `#chat-author-filter` with `Casey` and paste the two lines below into `#chat-import-text`. Click `#btn-import-chat`. Wait for `#analytics-content` to show a positive indexed chunk count and for the list to show the lighthouse text.
- Fill `#memory-search-input` with `lighthouse`. Click `#btn-search-memories`. Confirm a result contains `cardboard lighthouse`. Replace the query with `zebrafish` and press Enter. Confirm the empty state.
- Clear the search field and click **Search** again. Verify the imported card returns. Reopen the tab and read `data/memory_db/autobiography.json` in scratch. Match the stored `Chat Conversations` card against the import text.

```text
[12/31/23, 10:45 PM] Casey: I built a cardboard lighthouse with a paper lantern and an old cereal box.
[12/31/23, 10:46 PM] Casey: The lighthouse stood beside a model harbor with blue paper waves.
```

## Gotchas

- The browser importer accepts pasted text. For author filtering use a parseable WhatsApp export and match the author spelling.
- Submit after clearing the input to restore the unfiltered list.
