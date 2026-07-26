"""Gradio web client; all orchestration happens in the remote API service."""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

from connoisseur.orchestrators.settings import AppSettings

from .api_client import (
    ApiClientError,
    ConnoisseurApiClient,
    normalize_ui_history,
)


def create_demo(
    api_client: ConnoisseurApiClient | None = None,
) -> Any:
    """Build the UI without importing Gradio in non-web processes."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Install the Gradio client dependencies first") from exc

    settings = AppSettings.from_env()
    client = api_client or ConnoisseurApiClient(
        settings.api_base_url,
        timeout_seconds=settings.client_timeout_seconds,
        max_history_messages=settings.max_history_messages,
    )

    async def handle_chat(
        message: str,
        history: list[dict[str, str]] | None,
        backend: str,
        session_id: str | None,
    ) -> tuple[str, list[dict[str, str]], str]:
        normalized_history = normalize_ui_history(history)
        cleaned = (message or "").strip()
        active_session = session_id or str(uuid4())
        if not cleaned:
            return "", normalized_history, active_session
        try:
            result = await client.chat(
                cleaned,
                backend=backend,
                session_id=active_session,
                history=normalized_history,
            )
            answer = result.answer
            active_session = result.session_id
        except ApiClientError as exc:
            answer = (
                "I couldn't complete that request because the API reported: "
                f"{exc}"
            )
        normalized_history.extend(
            [
                {"role": "user", "content": cleaned},
                {"role": "assistant", "content": answer},
            ]
        )
        return "", normalized_history, active_session

    with gr.Blocks(title="Connoisseur Companion") as demo:
        gr.Markdown(
            "# Connoisseur Companion\n"
            "Explore restaurants and recipes with real retrieval and your choice "
            "of multi-agent orchestrator."
        )
        session = gr.State(value=str(uuid4()))
        with gr.Row():
            backend = gr.Dropdown(
                choices=["langgraph", "agno"],
                value="langgraph",
                label="Orchestration backend",
                interactive=True,
            )
        # Gradio 6 uses message dictionaries exclusively; its former ``type``
        # argument was removed with the legacy tuple format.
        chatbot = gr.Chatbot(height=500, label="Conversation")
        message = gr.Textbox(
            label="Ask about restaurants or recipes",
            placeholder=(
                'Try: "Find a romantic Japanese restaurant and explain why it fits."'
            ),
            lines=2,
        )
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.ClearButton([message, chatbot])

        submit_inputs = [message, chatbot, backend, session]
        submit_outputs = [message, chatbot, session]
        message.submit(
            handle_chat,
            submit_inputs,
            submit_outputs,
        )
        send.click(
            handle_chat,
            submit_inputs,
            submit_outputs,
        )
        clear.click(
            lambda: str(uuid4()),
            outputs=session,
        )

        gr.Examples(
            examples=[
                ["Find me a moody restaurant in DTLA."],
                ["Suggest a Japanese dining experience for a quiet date."],
                ["Show me a recipe with Mediterranean flavors."],
            ],
            inputs=message,
        )
    return demo


def main() -> None:
    demo = create_demo()
    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
    )


if __name__ == "__main__":
    main()
