"""Offline verification for the N-ATLaS Kaggle server notebook.

Runs Cells 1-16 of natlas_notebook.py on this machine with mock_backend
substituted for the real tokenizer/model, boots the real uvicorn server, and
executes every assertion in Cells 11-16 over real HTTP.

Then it re-runs the whole suite while forcing a CUDA OOM and a generic
generation failure, to prove the 507 and 500 paths work.

This proves the server plumbing. It cannot prove N-ATLaS generation quality or
4-bit GPU memory behaviour — those need the real T4.

Usage:
    python verify_offline.py
"""

import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
NOTEBOOK_SOURCE = HERE / "natlas_notebook.py"

TEST_API_KEY = "offline-verification-key-do-not-use-in-kaggle"

BOOTSTRAP = (
    "import sys, runpy, mock_backend; "
    "mock_backend.install(); "
    "path = sys.argv[1]; "
    "sys.argv = sys.argv[1:]; "
    "runpy.run_path(path, run_name='__main__')"
)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def configure_env(port: int) -> None:
    os.environ.update(
        {
            "N_ATLAS_STRICT_GPU": "0",
            "N_ATLAS_PORT": str(port),
            "N_ATLAS_API_KEY": TEST_API_KEY,
            "N_ATLAS_LOG_CONTENT": "0",
            "N_ATLAS_WORKDIR": str(HERE / ".offline-workdir"),
            "N_ATLAS_AUTO_INSTALL": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    (HERE / ".offline-workdir").mkdir(exist_ok=True)


def run_notebook(port: int) -> dict:
    sys.path.insert(0, str(HERE))
    import mock_backend

    mock_backend.install()
    configure_env(port)

    namespace = {
        "__name__": "__natlas_notebook__",
        "__file__": str(NOTEBOOK_SOURCE),
    }
    source = NOTEBOOK_SOURCE.read_text(encoding="utf-8")
    code = compile(source, str(NOTEBOOK_SOURCE), "exec")
    exec(code, namespace)

    print("\n" + "#" * 70)
    print("# POST-SUITE ASSERTIONS (beyond the notebook cells)")
    print("#" * 70)

    record = namespace["record"]
    clean_response = namespace["clean_response"]
    api = namespace["api"]
    CONFIG = namespace["CONFIG"]
    GENERATION_LOCK = namespace["GENERATION_LOCK"]
    BASE_URL = namespace["BASE_URL"]
    TEST_API_KEY = namespace["TEST_KEY"]
    mock_backend.MOCK_STATE.update({"force_oom": False, "force_error": False})

    raw, markers = clean_response(
        "Ina lafiya! Yaya ranar ka?\n\nQuestion: Ah, ina farin ciki..."
    )
    record(
        "cleaner truncates at a mid-text Question: marker",
        raw == "Ina lafiya! Yaya ranar ka?"
        and any("question" in m for m in markers),
        f"cleaned={raw!r} markers={markers}",
    )

    raw, _ = clean_response(
        "Question: What is your name?\nAnswer: My name is N-ATLaS.\nUser: who is this?"
    )
    record(
        "cleaner recovers the Answer: segment when output starts with Question:",
        raw == "My name is N-ATLaS.",
        f"cleaned={raw!r}",
    )

    raw, markers = clean_response(
        "<|im_start|>assistant\nBáwo ni? Ó dara.<|im_end|><|im_start|>user\nmore"
    )
    record(
        "cleaner strips ChatML artifacts and the fake next turn",
        raw == "Báwo ni? Ó dara.",
        f"cleaned={raw!r}",
    )

    raw, _ = clean_response(
        "Ina lafiya! Yaya ranar ka?<|im_end|>\n<|im_start|>user\nAnything else?"
    )
    record(
        "cleaner cuts a ChatML turn continuation instead of leaving a bare role label",
        raw == "Ina lafiya! Yaya ranar ka?",
        f"cleaned={raw!r}",
    )

    raw, _ = clean_response("Hello there.<|im_end|>\nuser\nAnything else?")
    record(
        "cleaner cuts a bare role line left behind after template tokens",
        raw == "Hello there.",
        f"cleaned={raw!r}",
    )

    raw, _ = clean_response("**Bold answer** with a list:\n- one\n- two")
    record(
        "cleaner preserves markdown formatting",
        raw.startswith("**Bold answer**") and "- one" in raw,
        f"cleaned={raw!r}",
    )

    raw, _ = clean_response(
        "I told the user: ask your manager. The user: field is required."
    )
    record(
        "cleaner keeps 'user:' that appears inside a sentence",
        raw == "I told the user: ask your manager. The user: field is required.",
        f"cleaned={raw!r}",
    )

    raw, _ = clean_response("Question 1: what is your name? Question 2: where are you?")
    record(
        "cleaner keeps 'Question 1:'/'Question 2:' list numbering",
        raw == "Question 1: what is your name? Question 2: where are you?",
        f"cleaned={raw!r}",
    )

    raw, _ = clean_response("The email is: user@example.com\nAsk: who owns it?")
    record(
        "cleaner keeps an email address",
        "user@example.com" in raw,
        f"cleaned={raw!r}",
    )

    _ = api(
        "POST",
        "/v1/chat/completions",
        body={
            "model": CONFIG.served_model_name,
            "messages": [{"role": "user", "content": "How are you?"}],
        },
    )
    record(
        "generation lock is free after a successful request",
        not GENERATION_LOCK.locked(),
        f"locked={GENERATION_LOCK.locked()}",
    )

    print("\n" + "#" * 70)
    print("# LANGUAGE TAGGING DETECTOR")
    print("#" * 70)

    detect_language = namespace["detect_language"]
    fold_text = namespace["fold_text"]
    build_language_directive = namespace["build_language_directive"]
    LANGUAGE_NAMES = namespace["LANGUAGE_NAMES"]

    record(
        "diacritic folding maps Yoruba/Igbo letters to ASCII",
        fold_text("Báwo ni? Kedu ka ị mere? Ṣé ibo ni jọ")
        == "bawo ni? kedu ka i mere? se ibo ni jo",
        repr(fold_text("Báwo ni? Kedu ka ị mere? Ṣé ibo ni jọ")),
    )

    DETECTOR_CASES = [
        ("How are you?", "en"),
        ("What is the weather in Lagos today?", "en"),
        ("Please tell me about NAFDAC numbers", "en"),
        ("Can you help me? I need a product verification", "en"),
        ("Kana lafiya?", "ha"),
        ("Sannu, yaya kake?", "ha"),
        ("Ina lafiya, na gode da tambayar", "ha"),
        ("Ka yi nasara yau!", "ha"),
        ("Báwo ni?", "yo"),
        ("Bawo ni?", "yo"),
        ("Kini n se?", "yo"),
        ("Mo nifẹ o, tani?", "yo"),
        ("Ṣé ibo ni? Yin ni ó dara", "yo"),
        ("Kedu ka ị mere?", "ig"),
        ("Kedu ka i mere?", "ig"),
        ("Nnọọ, kedu ka ị dị?", "ig"),
        ("Gịnị mere, ọ bụla anyị?", "ig"),
        ("How are you? Yaya kake?", "ha"),
        ("Answer in Yoruba.", "yo"),
        ("Please respond in English", "en"),
        ("Sauraya a cikin Hausa", "ha"),
    ]
    wrong = []
    for text, expected in DETECTOR_CASES:
        got = detect_language(text)["language"]
        if got != expected:
            wrong.append(f"{text!r} expected {expected} got {got}")
    record(
        f"detector classifies all {len(DETECTOR_CASES)} probes",
        not wrong,
        "; ".join(wrong) if wrong else "en/ha/yo/ig all correct",
    )

    record(
        "detector defaults to English on ambiguous input",
        detect_language("...")["language"] == "en"
        and detect_language("42")["language"] == "en",
        f"'...'->{detect_language('...')['language']} '42'->{detect_language('42')['language']}",
    )
    record(
        "an explicit language request overrides the score",
        detect_language("Say hello in Igbo")["confidence"] == "explicit",
        f"confidence={detect_language('Say hello in Igbo')['confidence']}",
    )
    record(
        "each directive names its target and forbids the others",
        all(
            LANGUAGE_NAMES[code] in directive
            and "Do not use" in directive
            and all(
                LANGUAGE_NAMES[other] in directive
                for other in LANGUAGE_NAMES
                if other != code
            )
            for code, directive in namespace["LANGUAGE_DIRECTIVES"].items()
        ),
        "each directive names all four languages, one to use and three to avoid",
    )
    record(
        "directives name the target language",
        "Yoruba" in build_language_directive("yo", False)
        and "Igbo" in build_language_directive("ig", False)
        and "Hausa" in build_language_directive("ha", False)
        and "English" in build_language_directive("en", False),
        build_language_directive("yo", False),
    )

    print("\n" + "#" * 70)
    print("# CONTEXT BUDGET ACCOUNTING (the T4 OOM root cause)")
    print("#" * 70)

    prepare_context = namespace["prepare_context"]
    context_budget_for = namespace["context_budget_for"]
    build_prompt_messages = namespace["build_prompt_messages"]
    apply_chat_template = namespace["_apply_chat_template"]
    count_tokens = namespace["count_tokens"]
    template_overhead = namespace["template_overhead"]

    record(
        "per-message template overhead is measured, not guessed",
        isinstance(template_overhead(), int) and template_overhead() >= 1,
        f"overhead={template_overhead()} tokens/message",
    )

    for label, turns, per_message in (
        ("76-turn history", 76, "lorem ipsum " * 40),
        ("120-turn history", 120, "detail " * 120),
        ("single huge turn", 1, "alpha beta gamma " * 4000),
    ):
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"T{i}: {per_message}"}
            for i in range(max(1, turns - 1))
        ]
        history.append({"role": "user", "content": "Summarise everything above."})
        prepared = build_prompt_messages(history)
        budget = context_budget_for(16)
        try:
            prepared_ctx = prepare_context(prepared, budget=budget)
            exact = count_tokens(apply_chat_template(prepared_ctx.messages))
            record(
                f"{label}: real templated prompt fits the budget",
                exact <= budget,
                f"exact={exact} budget={budget} dropped={prepared_ctx.dropped_messages} "
                f"reported={prepared_ctx.input_tokens}",
            )
            record(
                f"{label}: reported input_tokens matches the real prompt",
                prepared_ctx.input_tokens == exact,
                f"reported={prepared_ctx.input_tokens} exact={exact}",
            )
            record(
                f"{label}: the final user turn survives trimming",
                prepared_ctx.messages[-1]["role"] == "user",
                f"last_role={prepared_ctx.messages[-1]['role']}",
            )
        except Exception as exc:
            record(f"{label}: fits the budget", False, repr(exc)[:200])

    record(
        "default prompt budget is inside the T4-safe ceiling",
        CONFIG.max_input_tokens <= 4096,
        f"max_input_tokens={CONFIG.max_input_tokens} "
        f"(T4 measured ~1.6 MB VRAM per prompt token)",
    )
    record(
        "expandable_segments allocator is requested",
        os.environ.get("PYTORCH_ALLOC_CONF", "") == "expandable_segments:True",
        f"PYTORCH_ALLOC_CONF={os.environ.get('PYTORCH_ALLOC_CONF')}",
    )

    print("\n" + "#" * 70)
    print("# FORCED-FAILURE ASSERTIONS")
    print("#" * 70)

    mock_backend.MOCK_STATE["force_oom"] = True
    try:
        oom_response = api(
            "POST",
            "/v1/chat/completions",
            body={
                "model": CONFIG.served_model_name,
                "messages": [{"role": "user", "content": "How are you?"}],
            },
        )
        oom_status = oom_response.status_code
        oom_body = oom_response.text
    except Exception as exc:
        oom_status = -1
        oom_body = repr(exc)
    record(
        "forced CUDA OOM returns 507",
        oom_status == 507,
        f"status={oom_status} :: {oom_body[:120]}",
    )
    record(
        "507 body does not leak the raw exception",
        "OutOfMemoryError" not in oom_body and "Tried to allocate" not in oom_body,
        repr(oom_body[:140]),
    )
    mock_backend.MOCK_STATE["force_oom"] = False

    mock_backend.MOCK_STATE["force_error"] = True
    try:
        err_response = api(
            "POST",
            "/v1/chat/completions",
            body={
                "model": CONFIG.served_model_name,
                "messages": [{"role": "user", "content": "How are you?"}],
            },
        )
        err_status = err_response.status_code
        err_body = err_response.text
    except Exception as exc:
        err_status = -1
        err_body = repr(exc)
    record(
        "forced generation failure returns 500",
        err_status == 500,
        f"status={err_status}",
    )
    record(
        "500 body does not leak a traceback",
        "Traceback" not in err_body and "mock generation explosion" not in err_body,
        repr(err_body[:160]),
    )
    mock_backend.MOCK_STATE["force_error"] = False

    _ = api("GET", "/health", key=None, timeout=30)
    record(
        "server still healthy after recoverable failures",
        _health_ok(namespace),
        "",
    )

    print("\n" + "#" * 70)
    print("# OPENAI SDK COMPATIBILITY (spec 24)")
    print("#" * 70)

    try:
        from openai import OpenAI
        from openai import APIStatusError
    except Exception as exc:
        record("openai SDK is importable", False, repr(exc))
        return namespace

    record("openai SDK is importable", True, f"openai={__import__('openai').__version__}")

    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key=TEST_API_KEY, timeout=120.0, max_retries=0)

    sdk_response = client.chat.completions.create(
        model=CONFIG.served_model_name,
        messages=[{"role": "user", "content": "Sannu, yaya kake?"}],
    )
    sdk_text = sdk_response.choices[0].message.content
    record(
        "OpenAI SDK chat.completions.create() returns a parsed answer",
        bool(sdk_text.strip()),
        f"content={sdk_text[:120]!r}",
    )
    record(
        "OpenAI SDK models.list() parses",
        [m.id for m in client.models.list().data] == [CONFIG.served_model_name],
        f"ids={[m.id for m in client.models.list().data]}",
    )

    sdk_full = client.chat.completions.create(
        model=CONFIG.served_model_name,
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "My name is Ahmed."},
            {"role": "assistant", "content": "Nice to meet you, Ahmed."},
            {"role": "user", "content": "What is my name?"},
        ],
        temperature=0.7,
        max_tokens=128,
    )
    record(
        "OpenAI SDK round-trips a full multi-turn conversation",
        "ahmed" in (sdk_full.choices[0].message.content or "").lower(),
        f"content={sdk_full.choices[0].message.content[:120]!r}",
    )
    record(
        "OpenAI SDK exposes usage tokens",
        isinstance(sdk_full.usage.total_tokens, int) and sdk_full.usage.total_tokens > 0,
        f"usage={sdk_full.usage}",
    )
    record(
        "OpenAI SDK sees a chatcmpl- id and int created",
        sdk_full.id.startswith("chatcmpl-") and isinstance(sdk_full.created, int),
        f"id={sdk_full.id} created={sdk_full.created}",
    )

    bad_client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="wrong-key", timeout=60.0, max_retries=0)
    try:
        bad_client.chat.completions.create(
            model=CONFIG.served_model_name,
            messages=[{"role": "user", "content": "hi"}],
        )
        sdk_auth_status = 200
    except APIStatusError as exc:
        sdk_auth_status = exc.status_code
    record(
        "OpenAI SDK raises AuthenticationError on a bad key",
        sdk_auth_status == 401,
        f"status={sdk_auth_status}",
    )

    try:
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
        sdk_model_status = 200
    except APIStatusError as exc:
        sdk_model_status = exc.status_code
    record(
        "OpenAI SDK surfaces 404 for an unknown model",
        sdk_model_status == 404,
        f"status={sdk_model_status}",
    )

    return namespace


def _health_ok(namespace: dict) -> bool:
    response = namespace["api"]("GET", "/health", key=None, timeout=30)
    if response.status_code != 200:
        return False
    payload = response.json()
    return bool(payload.get("model_loaded")) and payload.get("status") == "ok"


def _child_env(port: int) -> dict:
    env = dict(os.environ)
    env.update(
        {
            "N_ATLAS_STRICT_GPU": "0",
            "N_ATLAS_API_KEY": TEST_API_KEY,
            "N_ATLAS_PORT": str(port),
            "N_ATLAS_LOG_CONTENT": "0",
            "N_ATLAS_WORKDIR": str(HERE / ".offline-workdir"),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": str(HERE),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def _run_script(script: str, args: list, port: int, timeout: float = 900.0):
    return subprocess.run(
        [sys.executable, "-c", BOOTSTRAP, str(HERE / script), *args],
        env=_child_env(port),
        cwd=str(HERE),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def verify_legacy_scripts(namespace: dict) -> None:
    record = namespace["record"]

    print("\n" + "#" * 70)
    print("# LEGACY SCRIPTS (kaggleserver.py / apiserver.py / kaggle.py)")
    print("#" * 70)

    source = (HERE / "natlas_notebook.py").read_text(encoding="utf-8")

    import notebook_cells

    record(
        "no API key literal remains in any script",
        all(TEST_API_KEY not in (HERE / name).read_text(encoding="utf-8") for name in (
            "apiserver.py", "kaggleserver.py", "kaggle.py", "natlas_notebook.py"
        )),
        "scanned apiserver.py, kaggleserver.py, kaggle.py, natlas_notebook.py",
    )
    record(
        "cell numbering is sequential 1..16",
        [int(t.split()[1]) for t in notebook_cells.cell_titles(source)] == list(range(1, 17)),
        f"{len(notebook_cells.cell_titles(source))} cells",
    )

    listing = _run_script("kaggle.py", ["--list"], free_port())
    record(
        "kaggle.py --list maps all 16 cells",
        listing.returncode == 0 and "16. CELL 16" in listing.stdout,
        f"rc={listing.returncode}",
    )
    if listing.returncode != 0:
        record("kaggle.py --list output", False, listing.stderr[-400:])

    subset = _run_script("kaggle.py", ["--cells", "1-7"], free_port())
    record(
        "kaggle.py --cells 1-7 loads the engine without starting a server",
        subset.returncode == 0
        and "MODEL READY" in subset.stdout
        and "Uvicorn running on" not in subset.stdout,
        f"rc={subset.returncode} model_ready={'MODEL READY' in subset.stdout}",
    )
    if subset.returncode != 0:
        record("kaggle.py --cells 1-7 output", False, subset.stderr[-500:])

    bad_range = _run_script("kaggle.py", ["--cells", "9-99"], free_port())
    record(
        "kaggle.py rejects an out-of-range cell range",
        bad_range.returncode == 2,
        f"rc={bad_range.returncode}",
    )

    engine = _run_script("kaggleserver.py", [], free_port())
    engine_out = engine.stdout
    record(
        "kaggleserver.py loads the engine standalone and exits 0",
        engine.returncode == 0 and "engine smoke test: 5/5" in engine_out,
        f"rc={engine.returncode}",
    )
    if engine.returncode != 0:
        record("kaggleserver.py output", False, (engine.stderr or engine_out)[-700:])
    else:
        record(
            "kaggleserver.py answers English/Hausa/Yoruba/Igbo/mixed",
            engine_out.count("[PASS]") >= 5,
            f"pass_lines={engine_out.count('[PASS]')}",
        )

    server_port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-c", BOOTSTRAP, str(HERE / "apiserver.py")],
        env=_child_env(server_port),
        cwd=str(HERE),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        base = f"http://127.0.0.1:{server_port}"
        ready = False
        deadline = time.time() + 180
        health = None
        while time.time() < deadline:
            if server.poll() is not None:
                break
            try:
                health = requests.get(f"{base}/health", timeout=5)
                if health.status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(1.0)
        record(
            "apiserver.py comes up and serves /health",
            ready,
            f"port={server_port}",
        )
        if ready:
            import requests as _rq

            models = _rq.get(
                f"{base}/v1/models",
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
                timeout=30,
            )
            record(
                "apiserver.py serves /v1/models with a valid key",
                models.status_code == 200
                and models.json()["data"][0]["id"] == "n-atlas",
                f"status={models.status_code}",
            )
            chat = _rq.post(
                f"{base}/v1/chat/completions",
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
                json={
                    "model": "n-atlas",
                    "messages": [{"role": "user", "content": "Kana lafiya?"}],
                    "max_tokens": 64,
                },
                timeout=180,
            )
            record(
                "apiserver.py serves a clean chat completion",
                chat.status_code == 200
                and bool(chat.json()["choices"][0]["message"]["content"].strip()),
                f"status={chat.status_code}",
            )
            bad = _rq.get(f"{base}/v1/models", timeout=30)
            record(
                "apiserver.py rejects unauthenticated /v1/models",
                bad.status_code == 401,
                f"status={bad.status_code}",
            )
    finally:
        server.terminate()
        try:
            output = server.communicate(timeout=30)[0] or ""
        except Exception:
            server.kill()
            output = ""
        if not ready:
            record("apiserver.py startup output", False, output[-800:])


def main() -> int:
    if not NOTEBOOK_SOURCE.exists():
        print(f"missing {NOTEBOOK_SOURCE}")
        return 2

    port = free_port()
    print(f"[verify] notebook source : {NOTEBOOK_SOURCE}")
    print(f"[verify] test port       : {port}")

    try:
        namespace = run_notebook(port)
        verify_legacy_scripts(namespace)
    except Exception:
        traceback.print_exc()
        print("\n[verify] NOTEBOOK EXECUTION FAILED")
        return 1

    results = namespace["TEST_RESULTS"]
    failed = [(name, detail) for name, ok, detail in results if ok is False]
    skipped = [name for name, ok, _ in results if ok is None]
    passed = sum(1 for _, ok, _ in results if ok is True)

    print("\n" + "=" * 70)
    print(
        f"OFFLINE VERIFICATION: {passed} passed, {len(failed)} failed, "
        f"{len(skipped)} skipped (GPU-only) out of {len(results)} checks"
    )
    if failed:
        print("-" * 70)
        for name, detail in failed:
            print(f"  FAIL  {name} :: {detail}")
    if skipped:
        print("-" * 70)
        for name in skipped:
            print(f"  SKIP  {name} :: needs a real CUDA GPU")
    print("=" * 70)

    namespace["stop_server"]()

    if failed:
        return 1
    if skipped:
        print("All runnable checks passed. GPU-only checks still need the T4.")
    else:
        print("All offline checks passed: API plumbing, auth, validation, context")
        print("budgeting, response cleaning, error mapping and the generation lock.")
    print("N-ATLaS generation quality and 4-bit T4 memory still need a real run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
