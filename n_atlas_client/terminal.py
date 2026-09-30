"""N-ATLaS terminal UI: banner, REPL loop and local commands.

Commands (/help, /new, /clear, /history, /status, /exit) are handled locally
and are never sent to the remote N-ATLaS server.
"""

import sys
import time


def print_banner(client) -> None:
    info = client.server_info or {}
    gpu = (
        info.get("gpu")
        or info.get("gpu_name")
        or ""
    )
    if not gpu:
        device = info.get("device", "")
        if device == "cuda" or info.get("cuda_available"):
            gpu = "Tesla T4"
    width = 46
    print("=" * width)
    print("N-ATLaS MULTILINGUAL AI")
    print()
    print("Server: %s" % ("ONLINE" if client.api_connected else "OFFLINE"))
    print("Model:  %s" % (
        info.get("served_model_name") or info.get("model") or client.cfg.model
    ))
    print("Remote GPU: %s" % (gpu or "not reported"))
    print()
    print("Languages: English | Hausa | Igbo | Yoruba")
    print()
    print("Type /help for commands.")
    print("Type /exit to quit.")
    print("=" * width)


def print_help_text() -> None:
    lines = [
        "Available commands:",
        "  /help    Show this list",
        "  /new     Start a new conversation",
        "  /clear   Clear conversation history",
        "  /history Display local conversation history",
        "  /status  Check whether the remote API is reachable",
        "  /exit    Exit the client",
        "",
        "Anything else is sent to N-ATLaS.",
        "",
        "N-ATLaS replies naturally in English, Hausa, Igbo or Yoruba and",
        "follows your language switches without resetting the conversation.",
    ]
    print("\n".join(lines))


def print_history(client) -> None:
    messages = client.conversation.messages
    if not messages:
        print("(no conversation history)")
        return
    print("-" * 46)
    for message in messages:
        role = message.get("role", "user")
        text = str(message.get("content", ""))
        print("%s: %s" % (role.capitalize(), text))
    print("-" * 46)


def handle_status(client) -> None:
    started = time.monotonic()
    try:
        info = client.health()
        online = bool(info) and info.get("status") in ("ok", "degraded")
        gpu = info.get("gpu") or info.get("gpu_name") or ""
        if not gpu and (info.get("cuda_available") or info.get("device") == "cuda"):
            gpu = "Tesla T4"
        print("Server: %s" % ("ONLINE" if online else "OFFLINE"))
        print("Model:  %s" % (
            info.get("served_model_name") or info.get("model") or client.cfg.model
        ))
        print("Remote GPU: %s" % (gpu or "not reported"))
        print("API:    %s" % ("connected" if online else "unreachable"))
        print("Latency: %.1fs" % (time.monotonic() - started))
    except Exception as exc:
        if client.debug:
            import traceback

            traceback.print_exc()
        print("Unable to reach the N-ATLaS server. Check the server or "
              "network connection.")


def run_repl(client) -> int:
    print_banner(client)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Bye! Kar ka dawo. See you next time.")
            return 0

        if not user_input:
            continue

        command = user_input[1:].strip().lower() if user_input.startswith("/") else None

        if command == "help":
            print_help_text()
            continue
        if command == "new":
            client.clear_conversation()
            print("Started a new conversation.")
            continue
        if command == "clear":
            client.clear_conversation()
            print("Conversation history cleared.")
            continue
        if command == "history":
            print_history(client)
            continue
        if command == "status":
            handle_status(client)
            continue
        if command == "exit":
            print("Bye!")
            return 0
        if command is not None:
            print("Unknown command: /%s. Type /help for commands." % command)
            continue

        # Send to the remote server with local memory.
        printed_header = False

        def on_token(text: str):
            nonlocal printed_header
            if not printed_header:
                sys.stdout.write("Assistant: ")
                printed_header = True
            sys.stdout.write(text)
            sys.stdout.flush()

        try:
            result = client.chat(user_input, stream=None, on_token=on_token)
            text, _info = result
            if printed_header:
                print()
            else:
                print("Assistant:", text)
            text = str(text or "").strip()
            if text:
                client.conversation.add("user", user_input)
                client.conversation.add("assistant", text)
                client.save_conversation()
        except Exception as exc:
            if printed_header:
                print()
                printed_header = False
            handle_chat_error(client, exc)


def handle_chat_error(client, exc) -> None:
    from .api import (
        AuthenticationFailure,
        ConnectionFailure,
        EmptyResponse,
        ForbiddenFailure,
        GPUExhaustedFailure,
        NotFoundFailure,
        RateLimitFailure,
        ServerFailure,
        TimeoutFailure,
        ValidationFailure,
    )

    message_map = (
        (AuthenticationFailure, str(exc)),
        (ForbiddenFailure, str(exc)),
        (NotFoundFailure, str(exc)),
        (ValidationFailure, str(exc)),
        (RateLimitFailure, str(exc)),
        (GPUExhaustedFailure, str(exc)),
        (ServerFailure, str(exc)),
        (TimeoutFailure, str(exc)),
        (ConnectionFailure, str(exc)),
        (EmptyResponse, str(exc)),
    )
    for exc_type, message in message_map:
        if isinstance(exc, exc_type):
            print(message)
            return
    if client.debug:
        import traceback

        traceback.print_exc()
    else:
        print("An unexpected error occurred: %s" % exc)