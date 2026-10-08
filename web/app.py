"""Interactive Web Application for Digital Self Preservation.

Features real-time voice conversation, autobiographical memory studio,
and voice calibration.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from src.audio.prompts import RECORDING_PROMPTS
from src.audio.recorder import VoiceRecorder
from src.memory.chat_parser import ChatLogParser
from src.pipeline.orchestrator import DigitalSelfOrchestrator


def create_app() -> gr.Blocks:
    orchestrator = DigitalSelfOrchestrator()
    recorder = VoiceRecorder()

    def handle_voice_chat(
        mic_audio: Optional[str],
        text_input: str,
        chat_history: List[List[Optional[str]]],
    ) -> Tuple[List[List[Optional[str]]], Optional[str], str]:
        """Handles a conversational turn from either voice or text input."""
        if not mic_audio and not text_input.strip():
            return chat_history, None, "Please provide audio input or type a message."

        if mic_audio:
            result = orchestrator.process_audio_turn(mic_audio)
            user_msg = f"[Spoken] {result['user_text']}"
        else:
            result = orchestrator.process_text_turn(text_input)
            user_msg = result["user_text"]

        bot_reply = result["response_text"]
        audio_out = result["audio_path"]
        chat_history.append([user_msg, bot_reply])

        # Format retrieved memories for debug view
        mem_info = []
        for m in result.get("retrieved_memories", []):
            cat = m.get("category", "General")
            snip = m.get("content", "")[:140].replace("\n", " ")
            mem_info.append(f"• [{cat}] {snip}...")

        mem_display = "\n".join(mem_info) if mem_info else "No specific episodic memories triggered."
        metrics = result.get("metrics", {})
        metric_str = f"Response Time: {metrics.get('total_sec', 0)}s (LLM: {metrics.get('llm_generation_sec', 0)}s, TTS: {metrics.get('tts_synthesis_sec', 0)}s)\nRecalled Memories:\n{mem_display}"

        return chat_history, audio_out, metric_str

    def add_memory(category: str, title: str, content: str) -> str:
        """Adds a new memory to the store."""
        if not title.strip() or not content.strip():
            return "Title and content cannot be empty."
        slug = title.strip().lower().replace(" ", "_")
        orchestrator.memory_store.add_document(
            doc_id=f"manual_{slug}",
            content=content.strip(),
            category=category.strip(),
            metadata={"title": title.strip()},
        )
        return f"Memory '{title}' successfully indexed into {category}!"

    def search_memories(query: str) -> str:
        """Searches indexed memories."""
        if not query.strip():
            return "Please enter a search query."
        results = orchestrator.memory_store.query(query, top_k=5)
        if not results:
            return "No matching memories found."
        lines = []
        for r in results:
            score = r.get("relevance_score", "N/A")
            cat = r.get("category", "General")
            lines.append(f"=== [{cat}] (Score: {score}) ===")
            lines.append(r.get("content", ""))
            lines.append("")
        return "\n".join(lines)

    def import_chat(chat_text: str, author_filter: str) -> str:
        """Parses and ingests chat log text."""
        if not chat_text.strip():
            return "Chat text cannot be empty."
        parser = ChatLogParser(target_author=author_filter if author_filter.strip() else None)
        messages = parser.parse_whatsapp(chat_text)
        if not messages:
            # Fallback to general lines
            lines = [l for l in chat_text.splitlines() if l.strip()]
            messages = [{"author": author_filter or "User", "text": l} for l in lines]

        chunks = parser.chunk_messages_for_memory(messages)
        for idx, chunk in enumerate(chunks):
            orchestrator.memory_store.add_document(
                doc_id=f"chat_chunk_{idx}",
                content=chunk["content"],
                category="Chat Conversations",
                metadata={"source": "chat_import"},
            )

        profile = parser.extract_linguistic_profile(messages)
        summary = (
            f"Successfully indexed {len(chunks)} chat memory chunks from {len(messages)} messages!\n"
            f"Linguistic summary: {profile['total_words']} total words, "
            f"Avg {profile['avg_words_per_message']} words/msg.\n"
            f"Top words: {', '.join([w for w, c in profile['top_words'][:8]])}"
        )
        return summary

    with gr.Blocks(title="Digital Self: Voice & Personality Preservation") as demo:
        gr.Markdown("# Digital Self Preservation Studio")
        gr.Markdown(
            "An autonomous, 100% locally hosted digital twin emulating your voice, personality, and autobiographical memories."
        )

        with gr.Tab("Real-Time Conversation"):
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Conversation", height=450)
                    with gr.Row():
                        text_input = gr.Textbox(
                            placeholder="Type a message or question...",
                            label="Text Input",
                            scale=4,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    mic_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Voice Input (Microphone)")
                with gr.Column(scale=1):
                    audio_output = gr.Audio(label="Digital Twin Voice Reply", autoplay=True)
                    inspect_box = gr.Textbox(label="Cognitive Trace & Memory Recall", lines=8, interactive=False)
                    clear_btn = gr.Button("Clear History")

            send_btn.click(
                handle_voice_chat,
                inputs=[mic_input, text_input, chatbot],
                outputs=[chatbot, audio_output, inspect_box],
            )
            clear_btn.click(lambda: ([], orchestrator.clear_history(), ""), outputs=[chatbot, None, inspect_box])

        with gr.Tab("Autobiographical Memory Studio"):
            gr.Markdown("### Manage and Curate Your Life Memories")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Add New Memory Card")
                    cat_input = gr.Dropdown(
                        choices=["Origins & Family", "Milestones", "Career & Projects", "Beliefs & Values", "Humor & Quirks"],
                        value="Origins & Family",
                        label="Memory Category",
                    )
                    title_input = gr.Textbox(placeholder="E.g., Learning to code in my bedroom", label="Title")
                    story_input = gr.Textbox(placeholder="Describe the story, thoughts, and emotions in your own words...", lines=6, label="Reflections & Narrative")
                    save_mem_btn = gr.Button("Save Memory Card", variant="primary")
                    mem_status = gr.Textbox(label="Status", interactive=False)
                    save_mem_btn.click(add_memory, inputs=[cat_input, title_input, story_input], outputs=[mem_status])

                with gr.Column():
                    gr.Markdown("#### Semantic Memory Search")
                    query_input = gr.Textbox(placeholder="E.g., What was your childhood like?", label="Search Query")
                    search_btn = gr.Button("Search Memories")
                    results_box = gr.Textbox(label="Retrieved Results", lines=10, interactive=False)
                    search_btn.click(search_memories, inputs=[query_input], outputs=[results_box])

            gr.Markdown("---")
            gr.Markdown("#### Ingest Exported Chat Logs (WhatsApp / Discord / Text)")
            with gr.Row():
                chat_text_box = gr.Textbox(placeholder="Paste exported chat lines here...", lines=6, label="Chat Log Content")
                author_box = gr.Textbox(placeholder="Your name or username in the chat", label="Author Filter (Optional)")
            import_chat_btn = gr.Button("Parse and Ingest Chat Logs")
            import_chat_status = gr.Textbox(label="Chat Ingestion Result", interactive=False)
            import_chat_btn.click(import_chat, inputs=[chat_text_box, author_box], outputs=[import_chat_status])

        with gr.Tab("Voice Studio & Recording Guide"):
            gr.Markdown("### Phonetically Balanced Reading Prompts")
            prompt_select = gr.Dropdown(
                choices=[f"{p['id']} - {p['category']}" for p in RECORDING_PROMPTS],
                value=f"{RECORDING_PROMPTS[0]['id']} - {RECORDING_PROMPTS[0]['category']}",
                label="Select Prompt",
            )
            prompt_text_box = gr.Textbox(value=RECORDING_PROMPTS[0]["prompt"], label="Script to Read Aloud", lines=3)
            prompt_intent_box = gr.Textbox(value=RECORDING_PROMPTS[0]["intent"], label="Vocal Intent & Cadence")

            def on_prompt_select(choice: str) -> Tuple[str, str]:
                pid = choice.split(" - ")[0]
                for p in RECORDING_PROMPTS:
                    if p["id"] == pid:
                        return p["prompt"], p["intent"]
                return "", ""

            prompt_select.change(on_prompt_select, inputs=[prompt_select], outputs=[prompt_text_box, prompt_intent_box])

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)

