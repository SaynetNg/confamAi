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


def run_nafdac_tests(cfg, results) -> None:
    """NAFDAC verification, checked without touching the network.

    These run before the connection tests and do not need the server, because
    the lookup happens on this machine. That is the point of verifying locally:
    the answer to "is A4-100160 genuine?" survives the Kaggle session ending.
    """
    print("-" * 46)
    print("NAFDAC verification (local, no server needed)")
    print("-" * 46)

    try:
        from .nafdac import NafdacInterceptor
    except Exception as exc:  # nafdac_source missing from the tree
        results.record("nafdac package importable", False, str(exc)[:160])
        return

    interceptor = NafdacInterceptor(cfg, debug=False)
    results.record("nafdac verification is enabled", interceptor.enabled)

    status = interceptor.status()
    results.record(
        "nafdac sources are reported",
        "NAFDAC data sources" in status,
        status.splitlines()[2].strip() if len(status.splitlines()) > 2 else "",
    )

    # A real number from the greenbook export.
    result = interceptor.verify("A4-100160")
    results.record(
        "A4-100160 verifies from the greenbook",
        result.found,
        "nrn=%s status=%s" % (result.nrn, result.status),
    )
    if result.found:
        record = result.primary
        results.record(
            "the product name comes from the data, not the model",
            "Spironolactone" in record.product_name,
            record.product_name,
        )
        results.record(
            "the approval date comes from the data",
            record.approval_date.startswith("2021-12"),
            record.approval_date,
        )
        results.record(
            "a hit names the source that confirmed it",
            bool(record.sources),
            ",".join(record.sources),
        )

    # A number the greenbook lists as inactive must not read as authorised.
    inactive = interceptor.verify("C4-1142")
    results.record(
        "an inactive product is reported as not authorised",
        inactive.found and inactive.authorised is False,
        "authorised=%s" % inactive.authorised,
    )

    unknown = interceptor.verify("A4-0000000")
    results.record("an unknown number is not found", not unknown.found)
    results.record(
        "an unknown number claims nothing about authorisation",
        unknown.authorised is None,
    )

    # The interception decision: which messages get verified at all.
    for message, expected, label in [
        ("Is A4-100160 genuine?", True, "a verification question"),
        ("verify 04-3877 please", True, "an explicit verify request"),
        ("A4-100160", True, "a bare known number"),
        ("Báwo ni? How are you?", False, "ordinary Yoruba conversation"),
        ("what is the weather in Lagos", False, "an unrelated question"),
        ("I take 500mg of ibuprofen", False, "a quantity, not a number"),
    ]:
        results.record(
            "interception: %s" % label,
            interceptor.looks_like_lookup(message) is expected,
            repr(message[:48]),
        )

    # The context block handed to the model must forbid invention, which is the
    # whole reason the verified data is passed rather than a yes/no answer.
    context = interceptor.context_for("Is A4-100160 genuine?", result)
    results.record("context block is produced", bool(context))
    results.record(
        "context block tells the model not to invent values",
        "do not" in context.lower() and "nafdac" in context.lower(),
    )

    found_text = interceptor.verify_command("A4-100160")
    results.record(
        "/verify renders a product table",
        "Spironolactone" in found_text and "Applicant" in found_text,
    )
    missing_text = interceptor.verify_command("A4-0000000")
    results.record(
        "/verify on a miss does not call the product fake",
        "not found" in missing_text.lower() and "not proof" in missing_text.lower(),
    )
    results.record(
        "/verify with no argument explains the format",
        "A4-100160" in interceptor.verify_command(""),
    )

    try:
        hits = interceptor.search_command("paracetamol")
        results.record("/search returns products", "matched" in hits, hits.splitlines()[0])
    except Exception as exc:
        results.record("/search returns products", False, str(exc)[:160])


def run_tests(cfg):
    results = TestResults()
    print("=" * 46)
    print("N-ATLaS CLIENT TEST SUITE")
    print("=" * 46)
    client = NAtlasClient(cfg, debug=False)

    # NAFDAC first, and unconditionally: it is the one part of the client that
    # works with the server offline, so a failed connection should not hide a
    # working or broken lookup.
    run_nafdac_tests(cfg, results)

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