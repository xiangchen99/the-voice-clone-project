# Persona Matrix

Edit how the twin is named and described, including its vocabulary and speech constraints.

## Sub-features

- `persona-load` shows the current persona.
- `persona-save` saves identity, traits, vocabulary, speech constraints, and approved replies.
- `persona-reopen` reloads saved values and shows the identity in Conversation.

## How to get to it (user POV)

Open **Persona Matrix**, edit the fields, and select **Save Changes**.

## Driving it with a browser

Start from the example persona in a fresh dashboard.

- Click `button[data-tab="tab-persona"]`. Wait for `#persona-name` to contain the example name.
- Set `#persona-name` to `Harbor Twin` and `#persona-fullname` to `Harbor Studio Twin`. Set `#persona-openness`, `#persona-conscientiousness`, and `#persona-agreeableness` to `Curious`, `Careful`, and `Warm`. Enter `steady, shipshape` in `#persona-phrases` and `synergy, paradigm` in `#persona-avoid`. Enter `Use short sentences.` and `Avoid acronyms.` on separate lines in `#persona-constraints`. Add `I usually start with an example.` to `#persona-approved-replies`.
- Click `#btn-save-persona`. Capture the response to `POST /api/persona` and require `{"success": true}`. Read scratch `configs/persona.local.yaml` and check the values, including list entries for phrases, avoided words, constraints, and approved replies.
- Reload, reopen **Persona Matrix**, and compare every edited field. Return to `button[data-tab="tab-chat"]` and check that `#chat-twin-name` reads `Harbor Twin`.

## Gotchas

- The save label reverts quickly. Use the persisted YAML and reloaded fields as the proof of save.
- Phrases and avoided words split on commas. Speech constraints split on newlines.
