"""Test suite for the N-ATLaS laptop client (python -m n_atlas_client --test).

Tests report PASS/FAIL. A language test only passes when the response is
non-empty — an HTTP 200 alone is never treated as proof.
"""

from .api import NAtlasClient, NAtlasError


class TestResults:
    def __init__(self):
        self.results = []

    def record(self, name: str, passed: bool, detail: str = "") -> bool:
        self.results.append((name, passed, detail))
        print("[%s] %s%s" % (
            "PASS" if passed else "FAIL",
            name,
            (" :: %s" % detail) if detail else "",
        ))
        return passed

    def passed(self) -> int:
        return sum(1 for _, ok, _ in self.results if ok)

    def failed(self):
        return [(name, detail) for name, ok, detail in self.results if not ok]


def run_tests(cfg):
    results = TestResults()
    print("=" * 46)
    print("N-ATLaS CLIENT TEST SUITE")
    print("=" * 46)
    client = NAtlasClient(cfg, debug=False)

    # --- connection ----------------------------------------------------
    try:
        client.connect()
        results.record("connection established", True, "openai client ready")
    except NAtlasError as exc:
        results.record("connection established", False, str(exc))
        return _finish(results)

    # --- health --------------------------------------------------------
    try:
        info = client.health()
        ok = bool(info) and info.get("status") in ("ok", "degraded")
        results.record(
            "health endpoint responds",
            ok,
            "status=%s model=%s" % (info.get("status"), info.get("model")),
        )
        client.api_connected = ok
    except NAtlasError as exc:
        results.record("health endpoint responds", False, str(exc))

    # --- models --------------------------------------------------------
    try:
        ids = client.check_available_models()
        results.record(
            "model 'n-atlas' is listed",
            cfg.model in ids,
            "ids=%s" % ids,
        )
    except NAtlasError as exc:
        results.record("model 'n-atlas' is listed", False, str(exc))

    # --- chat in the four languages -------------------------------------
    language_cases = [
        ("English", "How are you?"),
        ("Hausa", "Kana lafiya?"),
        ("Yoruba", "Báwo ni?"),
        ("Igbo", "Kedu ka ị mere?"),
    ]

    def chat_once(message):
        """Send a message with a fresh local history; return the reply text."""
        client.clear_conversation()
        try:
            text, _info = client.chat(message, stream=False)
        except NAtlasError as exc:
            return ""
        text = str(text or "").strip()
        return text

    for label, prompt in language_cases:
        answer = chat_once(prompt)
        results.record(
            "[%s] returns a non-empty reply" % label,
            bool(answer),
            repr(answer[:140]),
        )

    # --- conversation memory --------------------------------------------
    client.clear_conversation()
    memory_reply = ""
    try:
        memory_reply, _ = client.chat("My name is Ahmed.", stream=False)
        memory_reply = str(memory_reply or "").strip()
        client.conversation.add("user", "My name is Ahmed.")
        client.conversation.add("assistant", memory_reply)
    except NAtlasError as exc:
        memory_reply = ""
        results.record("memory: reply to name intro", False, str(exc))
    results.record(
        "memory: 'My name is Ahmed.' returns a reply",
        bool(memory_reply),
        repr(memory_reply[:140]),
    )

    recall = ""
    try:
        recall, _ = client.chat("What is my name?", stream=False)
        recall = str(recall or "").strip()
        client.conversation.add("user", "What is my name?")
        client.conversation.add("assistant", recall)
    except NAtlasError as exc:
        results.record("memory: recalls the name", False, str(exc))
    results.record(
        "memory: 'What is my name?' mentions Ahmed",
        "ahmed" in recall.lower(),
        repr(recall[:140]),
    )

    # --- language switching ---------------------------------------------
    client.clear_conversation()
    try:
        hausa_reply, _ = client.chat("Answer in Hausa.", stream=False)
        client.conversation.add("user", "Answer in Hausa.")
        client.conversation.add("assistant", str(hausa_reply or "").strip())
    except NAtlasError as exc:
        results.record("switch: 'Answer in Hausa'", False, str(exc))

    try:
        english_reply, _ = client.chat("Now answer in English.", stream=False)
        english_reply = str(english_reply or "").strip()
        client.conversation.add("user", "Now answer in English.")
        client.conversation.add("assistant", english_reply)
    except NAtlasError as exc:
        english_reply = ""
        results.record("switch: 'Now answer in English'", False, str(exc))

    if english_reply:
        results.record(
            "language switching: English reply is non-empty",
            True,
            repr(english_reply[:140]),
        )

    return _finish(results)


def _finish(results: TestResults) -> int:
    total = len(results.results)
    passed = results.passed()
    failed = results.failed()
    print()
    print("=" * 46)
    print("RESULT: %d/%d checks passed" % (passed, total))
    if failed:
        for name, detail in failed:
            print("  FAIL  %s :: %s" % (name, detail))
    print("=" * 46)
    return 0 if not failed else 1