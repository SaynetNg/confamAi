"""N-ATLaS terminal UI: banner, REPL loop and local commands.

Commands (/help, /new, /clear, /history, /status, /hausa, /exit) are handled
locally and are never sent to the remote N-ATLaS server.

The language is chosen here too. Each language is its own command, so the mode
is one keystroke away rather than an argument to remember:

    /hausa  /yoruba  /igbo  /english  /auto

A pinned mode holds: in Yoruba mode a Hausa message is still answered in
Yoruba, because the mode is a choice about the reply. `/langueage` shows the
current one without changing it.
"""

import sys
import time

from .language import AUTO

# Command word -> mode code. An empty code means "show the current mode"
# without setting one. Adding a spelling is a one-line change, so the whole
# command surface lives in one readable block.
LANGUAGE_COMMANDS = {
    "hausa": "ha",
    "yoruba": "yo",
    "igbo": "ig",
    "english": "en",
    "auto": AUTO,
    "langueage": "",
    "language": "",
    "lang": "",
}


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
    print("CONFAM AI")
    print()
    print("Server: %s" % ("ONLINE" if client.api_connected else "OFFLINE"))
    print("Model:  %s MULTILINGUAL MODEL" % (
        info.get("served_model_name") or info.get("model") or client.cfg.model
    ))
    print("Remote GPU: %s" % (gpu or "not reported"))
    print()
    mode = client._language()
    print("Language: %s%s" % (
        mode.label(),
        "   (/hausa /yoruba /igbo /english /auto to change)" if mode.pinned else "",
    ))
    print()
    if getattr(getattr(client, "cfg", None), "nafdac_enabled", False):
        print("NAFDAC verification: ON")
    print()
    print("Type /help for commands.")
    print("Type /exit to quit.")
    print("=" * width)


def print_help_text(client=None) -> None:
    lines = [
        "Available commands:",
        "  /help    Show this list",
        "  /new     Start a new conversation",
        "  /clear   Clear conversation history",
        "  /history Display local conversation history",
        "  /status  Check whether the remote API is reachable",
        "  /nafdac  Show which NAFDAC data sources are available",
        "  /verify  Check one NAFDAC number   (/verify A4-100160)",
        "  /search  Search products           (/search malaria)",
        "  /hausa   Reply in Hausa from now on",
        "  /yoruba  Reply in Yoruba from now on",
        "  /igbo    Reply in Igbo from now on",
        "  /english Reply in English from now on",
        "  /auto    Follow the language of each message",
        "  /langueage  Show the current language mode",
        "  /exit    Exit the client",
        "",
        "Anything else is sent to ConfamAI.",
        "",
        "ConfamAI replies in English, Hausa, Igbo or Yoruba and follows",
        "your language switches without resetting the conversation.",
        "",
        "A language command pins the reply, so it holds even if the next",
        "message is in another language. /auto hands the choice back.",
    ]
    if client is not None and getattr(client, "cfg", None) is not None:
        if getattr(client.cfg, "nafdac_enabled", False):
            lines.append("")
            lines.append("NAFDAC verification is on. Ask about a product in your")
            lines.append("own words — \"is A4-100160 genuine?\" — and the answer")
            lines.append("comes from the greenbook data, not from the model's memory.")
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
        mode = client._language()
        # The pinned mode is shown in the prompt, because a hard pin is
        # otherwise invisible until a reply comes back in the wrong language.
        prompt = "\nYou [%s]: " % mode.label() if mode.pinned else "\nYou: "
        try:
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Bye! Kar ka dawo. See you next time.")
            return 0

        if not user_input:
            continue

        # The command word, and whatever followed it, are separated before the
        # bare-command checks above so `/verify A4-100160` reaches the handlers
        # as ("verify", "A4-100160").
        command = None
        argument = ""
        if user_input.startswith("/"):
            parts = user_input[1:].split(None, 1)
            command = parts[0].strip().lower() if parts else ""
            argument = parts[1].strip() if len(parts) > 1 else ""

        if command == "help":
            print_help_text(client)
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
        if command == "nafdac":
            print(client.nafdac_status())
            continue
        if command == "verify":
            print(client._nafdac().verify_command(argument))
            continue
        if command == "search":
            print(client._nafdac().search_command(argument))
            continue
        if command == "exit":
            print("Bye!")
            return 0
        if command in LANGUAGE_COMMANDS:
            # `/hausa` and friends pin the mode; `/langueage` only shows it.
            print(mode.command(LANGUAGE_COMMANDS[command]))
            continue
        if command is not None:
            print("Unknown command: /%s. Type /help for commands." % command)
            continue

        # Send to the remote server with local memory.
        printed_header = False

        def on_token(text: str):
            nonlocal printed_header
            if not printed_header:
                sys.stdout.write("ConfamAI: ")
                printed_header = True
            sys.stdout.write(text)
            sys.stdout.flush()

        try:
            result = client.chat(user_input, stream=None, on_token=on_token)
            text, _info = result
            if printed_header:
                print()
            else:
                print("ConfamAI:", text)
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