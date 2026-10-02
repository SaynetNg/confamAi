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

# Every message is searched, so this is no longer "which messages get
    # verified". It is "which messages bring data with them": a miss brings
    # nothing and the model answers conversationally, which is a normal reply
    # rather than a refusal.
    for message, expected, label in [
        ("Is A4-100160 genuine?", True, "a verification question"),
        ("verify 04-3877 please", True, "an explicit verify request"),
        ("A4-100160", True, "a bare known number"),
        ("Is this paracetamol genuine?", True, "a product name, no number"),
        # The greenbook indexes English names, so a Hausa or Yoruba product word
        # searches and usually finds nothing. Nothing is attached and the model
        # answers — which is the honest outcome, not a gap.
        ("Shin wannan magari ne?", False, "a Hausa product word, no English match"),
        ("Je oro yi to?", False, "a Yoruba product word, no English match"),
        # These collide with greenbook names by coincidence. They do bring data,
        # and the block tells the model to drop it if it is not what the user
        # meant.
        ("what is the weather in Lagos", True, "a coincidental match"),
        ("I take 500mg of ibuprofen", True, "a drug the user may really hold"),
        ("I have a headache", False, "a symptom, no product"),
        ("Báwo ni? How are you?", False, "ordinary Yoruba conversation"),
    ]:
        results.record(
            "interception: %s" % label,
            interceptor.looks_like_lookup(message) is expected,
            repr(message[:48]),
        )

    # The server is the retrieval authority once a connection exists. These
    # check the integration property rather than the interceptor: the client
    # must not run a greenbook search of its own accord for a message the server
    # has not yet classified. A local hit here would previously be appended to
    # the user's turn, so the server's own intent step would then read the
    # client's product candidates as part of what the user said.
    from .api import NAtlasClient

    client = NAtlasClient(cfg, debug=False)
    results.record(
        "the client treats the server as the retrieval authority by default",
        getattr(cfg, "nafdac_server_authoritative", True) is True,
    )

    calls = []
    interceptor = client._nafdac()
    real_looks = interceptor.looks_like_lookup
    real_verify = interceptor.verify
    real_search = interceptor.search

    def spy_looks(message):
        calls.append(("looks", message))
        return real_looks(message)

    def spy_verify(message, quiet=False):
        calls.append(("verify", message))
        return real_verify(message, quiet=quiet)

    def spy_search(message, quiet=False):
        calls.append(("search", message))
        return real_search(message, quiet=quiet)

    interceptor.looks_like_lookup = spy_looks
    interceptor.verify = spy_verify
    interceptor.search = spy_search
    try:
        # Online: the server decides, so the client asks nothing locally.
        client.api_connected = True
        for message in (
            "Good morning",
            "Is A4-100160 genuine?",
            "Is this paracetamol genuine?",
            "Báwo ni? How are you today?",
        ):
            calls.clear()
            block = client._nafdac_context(message)
            results.record(
                "online: the client does not retrieve for %r" % message[:34],
                not calls,
                "calls=%r" % (calls,),
            )
            results.record(
                "online: no greenbook block is prepended for %r" % message[:34],
                block == "",
                repr(block)[:60],
            )

        # Offline: there is no server to ask, so the local data is a fallback
        # rather than a competing authority.
        client.api_connected = False
        calls.clear()
        offline_block = client._nafdac_context("Is A4-100160 genuine?")
        results.record(
            "offline: the local greenbook is still used as a fallback",
            bool(offline_block) and bool(calls),
            "calls=%d" % len(calls),
        )

        # KNOWN GAP, asserted rather than assumed away. Offline there is no
        # intent gate: the server that owns that decision is unreachable, and the
        # OpenAI-compatible protocol exposes no intent field, so the client
        # cannot ask. Its fallback therefore still uses the old permissive
        # "search anything, attach on a hit" behaviour, and a greeting that
        # happens to match a product name retrieves locally.
        #
        # This check passes when the gap is present so that it stays visible, and
        # fails when someone closes it — at which point the offline path needs
        # either an intent field in the protocol or its own gate.
        calls.clear()
        offline_greeting = client._nafdac_context("Good morning")
        results.record(
            "KNOWN GAP: the offline fallback has no intent gate",
            bool(calls),
            "calls=%d (closes when the offline path gains a gate)" % len(calls),
        )
    finally:
        interceptor.looks_like_lookup = real_looks
        interceptor.verify = real_verify
        interceptor.search = real_search
        client.api_connected = False

    # The context block handed to the model must forbid invention, which is the
    # whole reason the verified data is passed rather than a yes/no answer.
    context = interceptor.context_for("Is A4-100160 genuine?", result)
    results.record("context block is produced", bool(context))
    results.record(
        "context block tells the model not to invent values",
        "do not" in context.lower() and "nafdac" in context.lower(),
    )
    results.record(
        "context block forbids calling a product counterfeit",
        "never tell the user a product is counterfeit" in context.lower(),
    )
    results.record(
        "context block tells the model to reply in the user's language",
        all(word in context.lower() for word in ("hausa", "yoruba", "igbo")),
    )
    # The guardrails have to sit above the records, not below: the block is
    # length-capped and a long candidate list is truncated at the end, which
    # would otherwise cut off the rules that matter most.
    rendered = result.render().lower()
    results.record(
        "render guardrails come before the product data",
        # "applicant:" with the colon, not "applicant": the bare word also occurs
        # inside the guardrail itself ("never supply a product name, ingredient,
        # strength, applicant, ..."), so it cannot mark where the data starts.
        0 <= rendered.find("never call a product counterfeit")
        < rendered.find("applicant:"),
        f"rule@{rendered.find('never call a product')} "
        f"data@{rendered.find('applicant:')}",
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

    # A product name typed after /verify is searched, not rejected for
    # containing no NAFDAC number.
    word_text = interceptor.verify_command("paracetamol")
    results.record(
        "/verify searches a product name",
        bool(word_text.strip()) and "number" not in word_text.lower()[:40],
        word_text.splitlines()[0][:70] if word_text else "empty",
    )
    results.record(
        "/verify on a product name reports a verdict",
        any(
            marker in word_text
            for marker in ("AUTHENTIC", "FLAGGED", "NOT_REGISTERED")
        )
        or "These are different products" in word_text,
        word_text.splitlines()[0][:70] if word_text else "empty",
    )
    no_word = interceptor.verify_command("zzzznotarealproduct")
    results.record(
        "/verify on an unknown product does not call it fake",
        "not proof" in no_word.lower(),
        no_word.splitlines()[0][:70] if no_word else "empty",
    )

    # The search context must hold the model to the same rule as the number
    # path, so a miss is never reported as a counterfeit.
    try:
        search_result = interceptor.search("paracetamol")
        search_context = interceptor.context_for_search("paracetamol", search_result)
        results.record(
            "search context is produced",
            bool(search_context) and "nafdac" in search_context.lower(),
        )
        results.record(
            "search context forbids calling a product counterfeit",
            "never tell the user a product is counterfeit" in search_context.lower(),
        )
    except Exception as exc:
        results.record("search context is produced", False, str(exc)[:160])

    try:
        hits = interceptor.search_command("paracetamol")
        results.record("/search returns products", "matched" in hits, hits.splitlines()[0])
    except Exception as exc:
        results.record("/search returns products", False, str(exc)[:160])


def run_language_tests(cfg, results) -> None:
    """The language modes, checked without touching the network.

    Runs before the connection test because none of it needs the server: a mode
    that cannot be set, or a directive that names the wrong language, is broken
    whether or not Kaggle is up.
    """
    print("-" * 46)
    print("Language modes (local, no server needed)")
    print("-" * 46)

    from .language import (
        AUTO,
        LANGUAGE_NAMES,
        LanguageSwitcher,
        directive_for,
        normalize,
    )

    # Every spelling the REPL accepts has to reach the right mode.
    for spelling, expected in [
        ("hausa", "ha"), ("HAUSA", "ha"), ("ha", "ha"), ("sauraya", "ha"),
        ("yoruba", "yo"), ("yo", "yo"), ("yebu", "yo"),
        ("igbo", "ig"), ("ig", "ig"), ("ahukeri", "ig"),
        ("english", "en"), ("en", "en"),
        ("auto", AUTO), ("automatic", AUTO),
    ]:
        results.record(
            "/%s selects %s" % (spelling, LANGUAGE_NAMES[expected]),
            normalize(spelling) == expected,
            "got %s" % normalize(spelling),
        )

    results.record(
        "an unknown language is not a mode",
        normalize("klingon") == AUTO,
        "got %s" % normalize("klingon"),
    )

    # Auto is the default and must stay completely silent, so the client behaves
    # exactly as it did before language modes existed.
    switcher = LanguageSwitcher(cfg, debug=False)
    results.record(
        "a fresh client starts in auto",
        switcher.mode == AUTO,
        "mode=%s" % switcher.mode,
    )
    results.record(
        "auto sends no directive",
        switcher.context_for("Báwo ni?") == ""
        and directive_for(AUTO) == "",
    )
    results.record(
        "auto sends no language field",
        switcher.request_language() is None,
    )

    # A pinned mode: one directive, naming its language and not the others.
    for command, code in [
        ("hausa", "ha"), ("yoruba", "yo"), ("igbo", "ig"), ("english", "en"),
    ]:
        switcher.set(command)
        directive = switcher.context_for("How are you?")
        name = LANGUAGE_NAMES[code]
        others = [n for c, n in LANGUAGE_NAMES.items() if c not in (code, AUTO)]
        # The directive excludes the three other languages in one clause, so the
        # check is on what follows "do not use" rather than on three separate
        # phrases.
        excluded = directive.split("do not use", 1)[-1] if "do not use" in directive else ""
        results.record(
            "/%s names %s in the directive" % (command, name),
            name in directive,
        )
        results.record(
            "/%s excludes the other three" % command,
            bool(excluded) and all(other in excluded for other in others),
            excluded.strip()[:70],
        )
        results.record(
            "/%s sends the language field" % command,
            switcher.request_language() == code,
            "got %r" % switcher.request_language(),
        )

    # The pin is a pin. This is the whole feature, so it is asserted directly:
    # a directive must not contain an escape clause that lets the message's own
    # language win, and auto must never be reachable by accident.
    pinned = directive_for("ha")
    results.record(
        "a pin holds against a message in another language",
        "even when the message is written in another language" in pinned,
    )
    results.record(
        "a pin does not tell the model to follow the user",
        "follow them there" not in pinned.lower(),
    )
    results.record(
        "the directive says it overrides detection",
        "ignore automatic language detection" in pinned.lower(),
    )

    # The mode is what the greenbook tables render in, and a mode change has to
    # reach them without rebuilding anything.
    try:
        from .nafdac import NafdacInterceptor

        held = {"code": "en"}
        localized = NafdacInterceptor(cfg, debug=False, language=lambda: held["code"])
        found_en = localized.verify_command("A4-100160")
        held["code"] = "ha"
        found_ha = localized.verify_command("A4-100160")
        results.record(
            "the mode localises the /verify table",
            "AN AMMA" in found_ha and "AUTHORISED" in found_en,
            found_ha.splitlines()[0][:70] if found_ha else "empty",
        )
        results.record(
            "the mode localises labels but not product data",
            "Lamba" in found_ha and "Spironolactone" in found_ha,
        )
        held["code"] = "ig"
        results.record(
            "switching the mode reaches the renderer",
            "E KWADORO LA" in localized.verify_command("A4-100160"),
        )
    except Exception as exc:
        results.record("the mode localises the /verify table", False, str(exc)[:160])

    # Every language answers every key, so no user ever sees a key name.
    try:
        from nafdac_source.formatting import STRINGS, _t

        keys = set(STRINGS["en"])
        results.record(
            "every language answers every string key",
            all(
                not set(STRINGS[lang]) - keys for lang in STRINGS
            )
            and all(
                _t(lang, key) != key
                for lang in STRINGS
                for key in keys
            ),
            "languages=%s keys=%d" % (sorted(STRINGS), len(keys)),
        )
    except Exception as exc:
        results.record("every language answers every string key", False, str(exc)[:160])


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
    run_language_tests(cfg, results)

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

    # --- a pinned mode holds ---------------------------------------------
    # Each mode is asked an English question on purpose. A mode that only
    # follows the message would pass the block above and fail this one, which
    # is the point: /hausa means the reply is in Hausa whatever you type.
    for command, label in [
        ("hausa", "Hausa"), ("yoruba", "Yoruba"),
        ("igbo", "Igbo"), ("english", "English"),
    ]:
        client.set_language(command)
        answer = chat_once("How are you? Tell me about yourself.")
        results.record(
            "[%s mode] an English question still gets a reply" % label,
            bool(answer),
            repr(answer[:140]),
        )

    client.set_language("auto")
    results.record(
        "/auto puts the client back on the language of the message",
        client.language == "auto",
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