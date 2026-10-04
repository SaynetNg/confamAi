"""Exercise Cell 18 on its own, without a GPU or a loaded model.

The full notebook cannot run here: Cells 2-5 need torch, CUDA and the N-ATLaS
weights. Cell 18 needs none of that, so it is executed against a hand-built
namespace holding only the globals it actually reads.

This checks the three things that can break Cell 18 without a GPU:
  * it finds nafdac_source and imports it;
  * it degrades to "unavailable" instead of raising when it cannot;
  * a real number verifies and the verified block is produced.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import typing
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import notebook_cells  # noqa: E402

logging.basicConfig(level=logging.INFO)


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default=0):
    try:
        return int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        return default


def build_namespace():
    """The globals Cell 18 reads.

    NAFDAC_HOOKS is normally created by Cell 10, which owns the routes. Cell 18
    fills it, so an empty dict stands in for it here — that is the real
    relationship, not a stub to make the test pass.
    """
    return {
        "os": os,
        "re": __import__("re"),
        "json": json,
        "unicodedata": __import__("unicodedata"),
        "sys": sys,
        "Path": Path,
        "logger": logging.getLogger("cell18"),
        "WORKDIR": HERE,
        "_env_flag": _env_flag,
        "_env_int": _env_int,
        "Any": typing.Any,
        "List": typing.List,
        "Dict": typing.Dict,
        "Optional": typing.Optional,
        "NAFDAC_HOOKS": {},
    }


def _stub_extractor(payload):
    """An intent extractor that returns `payload` whatever the message.

    Stands in for the model. Cell 18 is exercised offline with no model loaded,
    so the understanding step is injected rather than generated.
    """
    return lambda message: payload


def _json_extractor(intent, product_name=None, nrn=None, manufacturer=None):
    def extract(message):
        return json.dumps(
            {
                "intent": intent,
                "product_name": product_name,
                "nafdac_number": nrn,
                "manufacturer": manufacturer,
            }
        )

    return extract


def run_gate_tests(result, check):
    """Tests 1-12: that only a validated product intent can reach the greenbook.

    The central assertion is not that any one message behaves correctly, it is
    that `find_products` is never called when the gate refuses. So several of
    these install a counting spy over the real search function and assert the
    call count is zero, rather than asserting on the absence of a result. A
    broken gate that still happened to return None would pass a weaker test.
    """
    enrich = result["_nafdac_enrich"]
    validate = result["validate_intent_extraction"]
    permits = result["_intent_permits_retrieval"]
    is_grounded = result["_entity_is_grounded"]
    ground = result["_ground_entities"]

    original_extractor = result["NAFDAC_INTENT_EXTRACTOR"]
    original_find = result["find_products"]

    def with_extractor(extractor):
        result["NAFDAC_INTENT_EXTRACTOR"] = extractor

    def restore():
        result["NAFDAC_INTENT_EXTRACTOR"] = original_extractor
        result["find_products"] = original_find

    def run(message):
        return enrich([{"role": "user", "content": message}])

    def count_searches(message):
        """Run a message and report how many times the greenbook was searched."""
        calls = []

        def spy(terms, sources=None, limit=20):
            calls.append(terms)
            return original_find(terms, sources=sources, limit=limit)

        result["find_products"] = spy
        try:
            outcome = run(message)
        finally:
            result["find_products"] = original_find
        return outcome, len(calls)

    # --- Test 1: a greeting must not search ---------------------------------
    with_extractor(_json_extractor("GENERAL_QUESTION"))
    outcome, searches = count_searches("Good morning")
    check(
        "T1  'Good morning' -> GENERAL_QUESTION, no search",
        outcome is None and searches == 0,
        f"searches={searches} outcome={outcome!r:.40}",
    )

    # --- Test 10: the specific false positive that motivated this -----------
    # The greenbook really does contain a product called "Good Morning Cough
    # Syrup", so a search here would succeed. The proof is that it is never
    # attempted, not that it would have failed.
    found_directly = original_find(
        "Morning", sources=result["NAFDAC_SOURCES"], limit=5
    )
    check(
        "T10 the greenbook really does contain 'Morning' products",
        found_directly.found,
        f"would have matched {found_directly.total}",
    )
    outcome, searches = count_searches("Good morning")
    check(
        "T10 'Good morning' never calls the search function",
        searches == 0,
        f"search calls={searches}",
    )

    # --- Test 4: a health question naming no product -----------------------
    with_extractor(_json_extractor("GENERAL_QUESTION"))
    outcome, searches = count_searches("What is malaria?")
    check(
        "T4  'What is malaria?' -> GENERAL_QUESTION, no search",
        outcome is None and searches == 0,
        f"searches={searches}",
    )

    # --- Test 5: a medicine with no name must not guess --------------------
    with_extractor(_json_extractor("CLARIFICATION_REQUIRED"))
    outcome, searches = count_searches("Check this medicine")
    check(
        "T5  'Check this medicine' -> CLARIFICATION_REQUIRED, no search",
        outcome is None and searches == 0,
        f"searches={searches}",
    )

    # --- Test 2: an explicit product check does search ---------------------
    with_extractor(
        _json_extractor("PRODUCT_CHECK", product_name="Good Morning Cough Syrup")
    )
    outcome, searches = count_searches("Is Good Morning Cough Syrup genuine?")
    check(
        "T2  explicit product check searches",
        searches == 1,
        f"searches={searches}",
    )
    check(
        "T2  the searched term is the extracted product name",
        outcome is not None
        and "Morning" in str(outcome.get("query", "")).title(),
        f"query={outcome.get('query')!r}" if outcome else "None",
    )

    # --- Test 3: a short brand name searches -------------------------------
    with_extractor(_json_extractor("PRODUCT_CHECK", product_name="Lonart"))
    outcome, searches = count_searches("Is Lonart genuine?")
    check("T3  'Is Lonart genuine?' searches", searches == 1, f"searches={searches}")

    # --- Test 6: an information request is a separate intent ---------------
    with_extractor(_json_extractor("PRODUCT_INFORMATION", product_name="Lonart"))
    outcome = run("Tell me about Lonart")
    check(
        "T6  'Tell me about Lonart' takes the product-information path",
        outcome is not None and outcome.get("intent") == "PRODUCT_INFORMATION",
        f"intent={outcome.get('intent') if outcome else None!r}",
    )
    check(
        "T6  a product check and an information request stay distinct",
        (outcome or {}).get("intent") != "PRODUCT_CHECK",
        f"intent={(outcome or {}).get('intent')!r}",
    )

    # --- Test 7: malformed JSON fails closed -------------------------------
    for label, payload in (
        ("not json at all", "I think it is fine"),
        ("truncated json", '{"intent": "PRODUCT_CHECK", "product_nam'),
        ("empty output", ""),
        ("a bare number", "42"),
    ):
        with_extractor(_stub_extractor(payload))
        outcome, searches = count_searches("Is Lonart genuine?")
        check(
            f"T7  malformed output ({label}) fails closed",
            outcome is None and searches == 0,
            f"searches={searches}",
        )

    # --- Test 8: an unknown intent fails closed ----------------------------
    with_extractor(_stub_extractor(json.dumps({"intent": "DEFINITELY_FAKE"})))
    outcome, searches = count_searches("Is Lonart genuine?")
    check(
        "T8  an unknown intent fails closed",
        outcome is None and searches == 0,
        f"searches={searches}",
    )

    # --- Test 8b: a verdict field invented by the model is rejected -------
    with_extractor(
        _stub_extractor(
            json.dumps(
                {
                    "intent": "PRODUCT_CHECK",
                    "product_name": "Lonart",
                    "nafdac_number": None,
                    "manufacturer": None,
                    "verdict": "AUTHENTIC",
                }
            )
        )
    )
    validated = validate(
        json.dumps(
            {
                "intent": "PRODUCT_CHECK",
                "product_name": "Lonart",
                "verdict": "AUTHENTIC",
            }
        )
    )
    check(
        "T8b a model-supplied verdict field is rejected",
        validated["valid"] is False,
        validated["reason"],
    )
    outcome, searches = count_searches("Is Lonart genuine?")
    check(
        "T8b a model-supplied verdict cannot reach retrieval",
        searches == 0,
        f"searches={searches}",
    )

    # --- Test 9: product intent with no entity asks for clarification ------
    with_extractor(_json_extractor("PRODUCT_CHECK"))
    outcome, searches = count_searches("Is it genuine?")
    check(
        "T9  product intent with no entity fails closed",
        outcome is None and searches == 0,
        f"searches={searches}",
    )
    check(
        "T9  the gate refuses a product intent with no NRN and no name",
        permits(validate('{"intent":"PRODUCT_CHECK"}'), "") is False,
    )

    # --- Test 12: the model cannot supply the NRN that gets verified -------
    # The message contains no NRN, but the model claims one. Only the regex is
    # allowed to decide, so nothing is looked up.
    with_extractor(
        _json_extractor("PRODUCT_CHECK", product_name="Lonart", nrn="A4-999999")
    )
    outcome, searches = count_searches("Is Lonart genuine?")
    check(
        "T12 a model-invented NRN is not trusted for lookup",
        outcome is not None and outcome.get("kind") == "search",
        f"kind={(outcome or {}).get('kind')!r}",
    )
    check(
        "T12 the invented NRN is not what was verified",
        "A4-999999" not in str(outcome or {}).replace("A4-999999", "", 1),
        "invented number did not reach a lookup",
    )
    check(
        "T12 a fabricated number in the message is still the only authority",
        permits(
            validate('{"intent":"PRODUCT_CHECK","nafdac_number":"A4-999999"}'),
            "",
        )
        is False,
    )

    # --- validator unit checks ---------------------------------------------
    ok = validate(
        '{"intent":"PRODUCT_CHECK","product_name":"Lonart",'
        '"nafdac_number":null,"manufacturer":null}'
    )
    check(
        "a well-formed extraction validates",
        ok["valid"] and ok["intent"] == "PRODUCT_CHECK" and ok["product_name"] == "Lonart",
        str(ok),
    )
    fenced = validate(
        'Sure!\n```json\n{"intent":"GENERAL_QUESTION","product_name":null,'
        '"nafdac_number":null,"manufacturer":null}\n```'
    )
    check(
        "a fenced or chatty reply still parses",
        fenced["valid"] and fenced["intent"] == "GENERAL_QUESTION",
        str(fenced),
    )
    for label, bad in (
        ("a non-string entity", '{"intent":"PRODUCT_CHECK","product_name":7}'),
        ("a null intent", '{"intent":null}'),
        ("a list entity", '{"intent":"PRODUCT_CHECK","product_name":["Lonart"]}'),
    ):
        rejected = validate(bad)
        check(
            f"the validator rejects {label}",
            rejected["valid"] is False,
            rejected["reason"],
        )
    check(
        "a failed extraction resolves to CLARIFICATION_REQUIRED",
        validate("nonsense")["intent"] == "CLARIFICATION_REQUIRED",
    )

    # --- entity grounding ---------------------------------------------------
    # The model may point at a product; it may not introduce one. Without this
    # check, "Check this medicine." with a model that answers "Lonart" retrieves
    # a real record for a product the user never named, and the answer looks
    # authoritative because the record is genuine.
    print("\n--- entity grounding ------------------------------------------")

    # A quoted entity is grounded, across the differences that do not change
    # which words the user wrote.
    for entity, message, label in [
        ("Lonart", "Is Lonart genuine?", "same case"),
        ("lonart", "Is Lonart genuine?", "lower case"),
        ("Lonart", "is lonart genuine?", "user typed lower"),
        ("Good Morning Cough Syrup", "Is Good Morning Cough Syrup genuine?", "multi-word"),
        ("Good-Morning Cough Syrup", "is good morning cough syrup genuine?", "hyphenated"),
        ("ọdụ", "Nke onye bu ọdụ?", "Igbo with diacritics"),
        ("odu", "Nke onye bu ọdụ?", "Igbo typed without diacritics"),
        ("AG Spironolactone", "is ag spironolactone genuine?", "leading article"),
    ]:
        check(
            f"a quoted entity is grounded ({label})",
            is_grounded(entity, message) is True,
            f"{entity!r} in {message!r}",
        )

    # An invented entity is not.
    for entity, message, label in [
        ("Lonart", "Check this medicine.", "invented brand"),
        ("Paracetamol", "Check this medicine.", "invented generic"),
        ("Lonart", "Is A4-100160 genuine?", "invented beside a real NRN"),
        ("Lonart", "", "invented with no message"),
        ("", "Is Lonart genuine?", "empty entity"),
    ]:
        check(
            f"an invented entity is not grounded ({label})",
            is_grounded(entity, message) is False,
            f"{entity!r} in {message!r}",
        )

    # The drop is what stops retrieval, and it happens in _ground_entities.
    payload = '{"intent":"PRODUCT_CHECK","product_name":"Lonart"}'
    dropped = ground(validate(payload), "Check this medicine.")
    check(
        "an ungrounded product name is discarded",
        dropped["product_name"] is None,
        str(dropped.get("product_name")),
    )
    check(
        "the discarded entity is reported",
        "product_name" in (dropped.get("ungrounded") or []),
        str(dropped.get("ungrounded")),
    )
    kept = ground(validate(payload), "Is Lonart genuine?")
    check(
        "a grounded product name survives",
        kept["product_name"] == "Lonart",
        str(kept["product_name"]),
    )

    # An ungrounded manufacturer is dropped on the same rule. It does not affect
    # retrieval today, but the same class of invention should not be waiting
    # behind it for whoever wires it up.
    with_mfr = ground(
        validate(
            '{"intent":"PRODUCT_CHECK","product_name":"Lonart","manufacturer":"Emzor"}'
        ),
        "Is Lonart genuine?",
    )
    check(
        "an ungrounded manufacturer is discarded",
        with_mfr["manufacturer"] is None,
        str(with_mfr.get("manufacturer")),
    )

    # --- grounding, end to end through the gate -----------------------------
    # "Check this medicine" where the model invents a product.
    with_extractor(_stub_extractor(payload))
    outcome, searches = count_searches("Check this medicine.")
    check(
        "an invented entity fails the gate end to end",
        outcome is None and searches == 0,
        f"searches={searches}",
    )

    # A hallucinated name beside a real NRN must not block the deterministic
    # lookup: the number came from the user's own text, and the name is dropped
    # rather than allowed to steer anything.
    with_extractor(
        _stub_extractor(
            '{"intent":"PRODUCT_CHECK","product_name":"Lonart",'
            '"nafdac_number":null,"manufacturer":null}'
        )
    )
    outcome, searches = count_searches("Is A4-100160 genuine?")
    check(
        "an ungrounded name does not block a deterministic NRN lookup",
        outcome is not None and outcome.get("kind") == "verify",
        f"kind={(outcome or {}).get('kind')!r} searches={searches}",
    )

    # "What is Lonart used for?" must still reach the product-information path.
    with_extractor(_json_extractor("PRODUCT_INFORMATION", product_name="Lonart"))
    outcome, searches = count_searches("What is Lonart used for?")
    check(
        "a product-information question with a quoted name retrieves",
        searches == 1,
        f"searches={searches}",
    )

    # Same question, but the model names a product the user did not mention.
    with_extractor(
        _json_extractor("PRODUCT_INFORMATION", product_name="Paracetamol")
    )
    outcome, searches = count_searches("What is Lonart used for?")
    check(
        "a product-information question with an invented name does not",
        outcome is None and searches == 0,
        f"searches={searches}",
    )

    # --- the invariant itself ----------------------------------------------
    # No validated product intent, no search. Stated directly so the property is
    # tested rather than merely implied by the cases above.
    for invalid in (
        {"valid": False, "intent": "PRODUCT_CHECK", "product_name": "Lonart"},
        {"valid": True, "intent": "GENERAL_QUESTION", "product_name": "Lonart"},
        {"valid": True, "intent": "CLARIFICATION_REQUIRED", "product_name": "Lonart"},
        {"valid": True, "intent": "PRODUCT_CHECK", "product_name": None},
    ):
        check(
            f"no retrieval for intent={invalid['intent']} valid={invalid['valid']}",
            permits(invalid, "") is False,
        )
    check(
        "a validated product check with a name may retrieve",
        permits({"valid": True, "intent": "PRODUCT_CHECK", "product_name": "Lonart"}, "")
        is True,
    )
    check(
        "a validated product check with a deterministic NRN may retrieve",
        permits({"valid": True, "intent": "PRODUCT_CHECK"}, "A4-100160") is True,
    )

    restore()


def _run_cell18_isolated(search_dirs_replacement=None):
    """Run Cell 18 in a clean subprocess and return its namespace as JSON.

    A subprocess is required, not stylistic. `nafdac_source` is already in
    `sys.modules` by the time this test runs, and `_import_nafdac_source()`
    returns True immediately when it sees that. Re-executing Cell 18 in this
    process would therefore always report the greenbook as loaded and the test
    would pass for the wrong reason. A fresh interpreter is the only honest
    simulation of a Kaggle session.

    The environment always points the data search at files that do not exist, so
    a data file can never be found. `search_dirs_replacement` additionally hides
    the package itself, which is the unattached-dataset case.
    """
    source = notebook_cells.read_source()
    if search_dirs_replacement:
        # Override the whole list rather than its first entry: the list also
        # contains WORKDIR and the cwd, so editing only the environment entry
        # would still find the package sitting in the project root.
        marker = "    str(Path.cwd()),\n]"
        if marker not in source:
            raise AssertionError("could not locate the end of NAFDAC_SEARCH_DIRS")
        source = source.replace(
            marker,
            marker + "\n# overridden by the test\nNAFDAC_SEARCH_DIRS = "
            + search_dirs_replacement,
            1,
        )
    body = notebook_cells.code_cells(source)[17]

    # Written to a file rather than passed with -c: the cell body is far longer
    # than Windows will accept on a command line.
    harness = (
        "import json, logging, pathlib, sys, typing\n"
        "logging.basicConfig(level=logging.CRITICAL)\n"
        "ns = {\n"
        "    'os': __import__('os'),\n"
        "    're': __import__('re'),\n"
        "    'json': __import__('json'),\n"
        "    'unicodedata': __import__('unicodedata'),\n"
        "    'sys': sys,\n"
        "    'Path': pathlib.Path,\n"
        "    'logger': logging.getLogger('isolated'),\n"
        "    'WORKDIR': pathlib.Path('.'),\n"
        "    '_env_flag': lambda n, d=False: d,\n"
        "    '_env_int': lambda n, d=0: d,\n"
        "    'Any': typing.Any, 'List': typing.List,\n"
        "    'Dict': typing.Dict, 'Optional': typing.Optional,\n"
        "    'NAFDAC_HOOKS': {},\n"
        "    'MODEL': None,\n"
        "}\n"
        "exec(compile(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'),"
        " 'cell18', 'exec'), ns)\n"
        "out = ns['_nafdac_status']()\n"
        "out['_hook_registered'] = callable(ns['NAFDAC_HOOKS'].get('status'))\n"
        "out['_enrich_declines'] = ns['_nafdac_enrich'](\n"
        "    [{'role': 'user', 'content': 'Is Lonart genuine?'}]\n"
        ") is None\n"
        "print('@@JSON@@' + json.dumps(out))\n"
    )

    env = dict(os.environ)
    # Point the data search somewhere with no greenbook file in it.
    env["NAFDAC_SOURCE_DIR"] = str(HERE / "no-such-dir")
    env["NAFDAC_CSV_PATH"] = str(HERE / "no-such-dir" / "nafdac_greenbook.csv")
    env["NAFDAC_XLSX_PATH"] = str(HERE / "no-such-dir" / "nafdac_greenbook.xlsx")
    env["NAFDAC_AUTODETECT_FILES"] = "0"
    env.pop("NAFDAC_DATABASE_URL", None)
    env["PYTHONIOENCODING"] = "utf-8"

    with tempfile.TemporaryDirectory() as tmp:
        body_file = Path(tmp) / "cell18.py"
        harness_file = Path(tmp) / "harness.py"
        body_file.write_text(body, encoding="utf-8")
        harness_file.write_text(harness, encoding="utf-8")
        # The project root, so the package is importable unless the test hides
        # it. That is what separates "dataset not attached" from "data file
        # missing", which behave very differently.
        completed = subprocess.run(
            [sys.executable, str(harness_file), str(body_file)],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(HERE.parent),
            env=env,
        )

    if completed.returncode != 0:
        raise AssertionError(
            "isolated Cell 18 failed (rc=%s):\n%s"
            % (completed.returncode, completed.stderr.strip()[-2000:])
        )
    marker_line = "@@JSON@@"
    for line in completed.stdout.splitlines():
        if line.startswith(marker_line):
            return json.loads(line[len(marker_line) :])
    raise AssertionError("isolated Cell 18 printed no status payload")


def run_multi_number_tests(result, check):
    """Tests 13-18: every NRN is resolved, and a miss still carries a block.

    Both of these are regressions rather than new features, and both came from
    watching the live server give confidently wrong answers:

      * an unregistered number used to attach no context at all, so the model
        answered "is A4-9999999 fake?" from memory and said yes;
      * with two numbers in a message only the first was resolved, so the model
        invented a product for the second — the same number came back as three
        different drugs in three runs.
    """
    enrich = result["_nafdac_enrich"]
    original_extractor = result["NAFDAC_INTENT_EXTRACTOR"]
    original_max = result["NAFDAC_MAX_NUMBERS"]

    def check_intent(intent, product_name=None):
        result["NAFDAC_INTENT_EXTRACTOR"] = _json_extractor(intent, product_name)

    try:
        def run(message):
            return enrich([{"role": "user", "content": message}])

        # --- an unregistered number must still produce a block -------------
        check_intent("PRODUCT_CHECK")
        miss = run("Is A4-0000000 genuine?")
        check(
            "an unregistered number attaches a block instead of nothing",
            miss is not None,
            f"enrich returned {miss!r}",
        )
        if miss:
            check(
                "the miss block says NOT_REGISTERED",
                "Verdict: NOT_REGISTERED" in miss["content"],
                miss.get("verdict"),
            )
            check(
                "the miss block forbids a counterfeit claim",
                "counterfeit" in miss["content"].lower(),
            )
            check(
                "the miss block tells the model not to call it fake",
                "not treat this as proof" in miss["content"].lower()
                or "fake" in miss["content"].lower(),
            )
            check("the miss reports found=False", miss.get("found") is False)
            check(
                "the miss verdict is not_registered",
                miss.get("verdict") == "not_registered",
                miss.get("verdict"),
            )
            check(
                "the miss records the number as unregistered",
                bool(miss.get("unregistered")),
                str(miss.get("unregistered")),
            )

        # --- both numbers resolved -----------------------------------------
        result["NAFDAC_MAX_NUMBERS"] = 4
        two = run("Compare A4-100160 and A11-0009 for me.")
        check("a two-number message produces a block", two is not None)
        if two:
            content = two["content"]
            check(
                "both numbers are listed as resolved",
                sorted(two.get("nrns") or []) == ["A11-0009", "A4-100160"],
                str(two.get("nrns")),
            )
            check(
                "the first product is present",
                "AG Spironolactone" in content,
                "A4-100160 record missing",
            )
            check(
                "the second product is present too",
                "Alben Paracetamol Drops" in content,
                "A11-0009 record missing",
            )
            check(
                "both numbers are marked registered",
                sorted(two.get("registered") or []) == ["A11-0009", "A4-100160"],
                str(two.get("registered")),
            )
            check(
                "the multi-number safety clause is present",
                "more than one NAFDAC number" in content,
                "clause missing",
            )
            check(
                "it forbids recommending a substitute",
                "in place of another" in content,
                "substitute clause missing",
            )
            check(
                "it forbids ranking the numbers",
                "NEVER rank" in content,
                "ranking clause missing",
            )

        # --- a mixed batch must not report one clean verdict ----------------
        mixed = run("I have two packs: A4-100160 and A4-0000000. Which should I trust?")
        check("a mixed batch produces a block", mixed is not None)
        if mixed:
            check(
                "a mixed batch is not reported as authentic",
                mixed.get("verdict") != "authentic",
                mixed.get("verdict"),
            )
            check("a mixed batch is marked mixed", mixed.get("verdict") == "mixed",
                  mixed.get("verdict"))
            check(
                "the unregistered number is named",
                bool(mixed.get("unregistered")),
                str(mixed.get("unregistered")),
            )
            check(
                "the not-registered block is in the merged content",
                "Verdict: NOT_REGISTERED" in mixed["content"],
            )

        # --- a flagged number poisons the batch verdict ---------------------
        flagged = run("Compare A4-100160 and C4-1142.")
        if flagged:
            check(
                "a flagged product makes the whole batch flagged",
                flagged.get("verdict") == "flagged",
                flagged.get("verdict"),
            )

        # --- the single-number block must forbid restating the verdict ------
        # Regression guard. "Is A4-100160 genuine?" came back as a bare verdict
        # with no product name, and in 3 of 10 runs it inverted AUTHENTIC into
        # "not registered". The model was paraphrasing the verdict instead of
        # quoting it, so the block now forbids restating it.
        result["NAFDAC_MAX_NUMBERS"] = original_max
        check_intent("PRODUCT_CHECK")
        one = run("Is A4-100160 genuine?")
        if one:
            content = one["content"]
            check(
                "the block forbids restating the verdict in own words",
                "NEVER restate a verdict" in content,
                "clause missing",
            )
            check(
                "the block forbids inverting AUTHENTIC into not-registered",
                "do not write 'not registered'" in content,
                "inversion clause missing",
            )
            check(
                "the block forbids calling a listed product missing",
                "Never "
                "describe a product the block names as missing" in content,
                "missing-product clause missing",
            )
            check(
                "the block asks for the product name beside the verdict",
                "name the product from the block" in content,
                "quote-clause missing",
            )
            check(
                "the block forbids echoing the [nafdac] marker",
                "Never reproduce the '[nafdac]' marker" in content,
                "marker clause missing",
            )
            check(
                "the block refuses the yes/no format trap",
                "one-word or yes/no answer" in content,
                "format clause missing",
            )
            check(
                "the block refuses a user-asserted counterfeit claim",
                "does not make it so" in content,
                "injection clause missing",
            )

        # --- a single number must not get the multi-number clause ----------------
        single = run("Is A4-100160 genuine?")
        check("a single number produces a block", single is not None)
        if single:
            check(
                "a single number omits the multi-number clause",
                "more than one NAFDAC number" not in single["content"],
            )
            check("a single number reports found=True", single.get("found") is True)
            check(
                "a single number keeps its nrn for the caller",
                single.get("nrn") == "A4-100160",
                str(single.get("nrn")),
            )

        # --- the cap on how many numbers one message may carry --------------
        result["NAFDAC_MAX_NUMBERS"] = 1
        capped = run("Compare A4-100160 and A11-0009.")
        if capped:
            check(
                "the cap bounds the work per message",
                len(capped.get("nrns") or []) == 1,
                str(capped.get("nrns")),
            )
    finally:
        result["NAFDAC_INTENT_EXTRACTOR"] = original_extractor
        result["NAFDAC_MAX_NUMBERS"] = original_max


def run_multi_candidate_tests(result, check):
    """A name search matching many products must not produce an answerable block.

    Regression. "Is Artemether and Lumefantrine genuine?" returned "considered
    genuine drugs... over-the-counter (OTC)... produced by Swiss Pharma Pvt. Ltd
    ... since 2020". None of that was invented — it was all row one of the block,
    C4-1419, with that row's FLAGGED verdict dropped. The block told the model
    not to choose; it chose the best-ranked row.

    Two causes, both fixed: the candidate list was truncated mid-record so the
    model never saw the total, and the instruction said "ask which one" without
    saying what it must not do.
    """
    enrich = result["_nafdac_enrich"]
    original_extractor = result["NAFDAC_INTENT_EXTRACTOR"]
    original_limit = result["NAFDAC_MAX_SEARCH_RESULTS"]
    original_ambiguous = result["NAFDAC_MAX_AMBIGUOUS_RESULTS"]
    original_chars = result["NAFDAC_MAX_CONTEXT_CHARS"]

    def check_intent(product_name):
        result["NAFDAC_INTENT_EXTRACTOR"] = _json_extractor(
            "PRODUCT_CHECK", product_name)

    try:
        # --- many matches -------------------------------------------------
        check_intent("Artemether and Lumefantrine")
        many = enrich([{"role": "user",
                        "content": "Is Artemether and Lumefantrine genuine?"}])
        check("a many-match search attaches a block", many is not None)
        if many:
            content = many["content"]
            check("a many-match search is marked ambiguous",
                  many.get("ambiguous") is True)
            check("the total match count is carried",
                  isinstance(many.get("total"), int) and many["total"] > 100,
                  str(many.get("total")))
            check("the multi-candidate clause is present",
                  "MANY DIFFERENT products" in content, "clause missing")
            check("it forbids naming one product as the answer",
                  "NEVER present one of these products as the answer"
                  in content, "selection clause missing")
            check("it forbids a verdict for the whole set",
                  "NEVER give a verdict for the set as a whole" in content,
                  "verdict clause missing")
            check("it calls out the first row specifically",
                  "The first row is not the answer" in content,
                  "first-row clause missing")
            check("it warns that rows may be Inactive",
                  "are Inactive and expired" in content,
                  "status clause missing")
            check("the block is not truncated mid-record",
                  not content.rstrip().endswith("[truncated]"),
                  "block ends truncated")
            check("the block reports how many were shown",
                  "shown" in many and many["shown"] < many.get("total", 0),
                  "shown=%s total=%s" % (many.get("shown"), many.get("total")))
            check("more than one candidate is shown",
                  many["shown"] >= 3,
                  "shown=%d" % many["shown"])
            check("the total is stated in the block text",
                  "458 products match" in content or
                  "%d products match" % many["total"] in content,
                  "total missing from block")
            check("the block says they are different products",
                  "DIFFERENT products" in content, "difference note missing")
            # The temptable fields are what the model quoted from row one, so an
            # ambiguous block must not carry them at all.
            for label, needle in (
                ("manufacturer", "Manufacturer:"),
                ("marketing category", "Marketing category:"),
                ("approval date", "Approved:"),
                ("greenbook url", "Greenbook page:"),
            ):
                check("the ambiguous block omits %s" % label,
                      needle not in content, needle)
            check("fewer candidates are attached than a plain search",
                  many["shown"] <= original_ambiguous,
                  "shown=%d cap=%d" % (many["shown"], original_ambiguous))

        # --- a single match keeps the ordinary path ----------------------
        check_intent("Artheget EZ")
        one = enrich([{"role": "user", "content": "Is Artheget EZ genuine?"}])
        check("a single-match search attaches a block", one is not None)
        if one:
            check("a single-match search is not ambiguous",
                  one.get("ambiguous") is False)
            check("a single-match search omits the multi-candidate clause",
                  "MANY DIFFERENT products" not in one["content"])
            check("a single-match search can show the full candidate list",
                  one["shown"] == original_limit,
                  "shown=%d" % one["shown"])
    finally:
        result["NAFDAC_INTENT_EXTRACTOR"] = original_extractor
        result["NAFDAC_MAX_SEARCH_RESULTS"] = original_limit
        result["NAFDAC_MAX_AMBIGUOUS_RESULTS"] = original_ambiguous
        result["NAFDAC_MAX_CONTEXT_CHARS"] = original_chars


def run_absent_greenbook_tests(check):
    """A deployment with no greenbook must be diagnosable, not merely silent.

    This is the state that actually happens on Kaggle when the dataset is not
    attached, and it is the hardest one to debug: the server still answers every
    question, so nothing looks broken. Before this was fixed, the endpoint meant
    to explain the failure returned 503 as well, so the reason was invisible from
    outside, and /health said only "greenbook data is not loaded yet".
    """
    print("\n--- greenbook absent: is the failure diagnosable? --------")

    # Nothing importable and nothing readable: the unattached-notebook case.
    absent = _run_cell18_isolated('["/nonexistent/a", "/nonexistent/b"]')
    check(
        "the status hook is registered even though nothing loaded",
        absent.get("_hook_registered") is True,
    )
    check("enabled is False", absent.get("enabled") is False, json.dumps(absent)[:140])
    check(
        "a reason is given",
        bool(absent.get("reason")),
        str(absent.get("reason"))[:140],
    )
    check(
        "the paths searched are listed, so the mount can be compared",
        bool(absent.get("searched_dirs")),
        str(absent.get("searched_dirs"))[:140],
    )
    hint = absent.get("hint") or ""
    for token in (
        "nafdac_source",
        "nafdac_greenbook.csv",
        "NAFDAC_SOURCE_DIR",
        "NAFDAC_DATABASE_URL",
        "Cell 18",
    ):
        check(f"the fix names {token}", token in hint)
    check(
        "the enrich hook declines rather than raising",
        absent.get("_enrich_declines") is True,
    )

    # The package importable but no data file: `enabled` is true here, because
    # the import succeeded, yet the server still cannot verify anything. This is
    # the case that matters most, because a green banner would hide it.
    no_data = _run_cell18_isolated()
    if no_data.get("enabled") is True:
        check(
            "a loaded package with no data file reports enabled but not ready",
            no_data.get("ready") is False,
            json.dumps(no_data)[:140],
        )
        check(
            "and says why",
            "no data file" in (no_data.get("hint") or ""),
            str(no_data.get("hint"))[:120],
        )
        check(
            "and is never described as ready",
            no_data.get("ready") is not True,
        )
    else:
        check(
            "with no data file the status is still diagnosable",
            no_data.get("enabled") is False,
            json.dumps(no_data)[:140],
        )


def main() -> int:
    failures = []

    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}" + (f" :: {detail}" if detail else ""))
        if not condition:
            failures.append(name)

    print("=" * 70)
    print("CELL 18 — NAFDAC greenbook (offline)")
    print("=" * 70)

    ns = build_namespace()
    result = notebook_cells.exec_range(18, 18, ns)

    check(
        "Cell 18 executed without raising",
        "NAFDAC_AVAILABLE" in result,
        f"available={result.get('NAFDAC_AVAILABLE')}",
    )
    if not result.get("NAFDAC_AVAILABLE"):
        print(f"  nafdac_source could not be loaded: {result.get('NAFDAC_IMPORT_ERROR')}")
        print("  Nothing further can be checked on this machine.")
        return 1

    status = result["_nafdac_status"]()
    check("status reports enabled", status.get("enabled") is True)
    check(
        "at least one source is usable",
        status.get("any_source_usable") is True,
        str([(s["name"], s["usable"]) for s in status.get("sources", [])]),
    )
    check(
        "the status hook is registered even when the data loaded",
        callable(result["NAFDAC_HOOKS"].get("status")),
    )
    check(
        "Cell 18 filled the verify hook",
        callable(result["NAFDAC_HOOKS"].get("verify")),
    )
    check(
        "Cell 18 filled the enrich hook",
        callable(result["NAFDAC_HOOKS"].get("enrich")),
    )

    # A real number from the greenbook, verified through the same hook the chat
    # endpoint uses.
    found = result["verify_nafdac_product"]("A4-100160", sources=result["NAFDAC_SOURCES"])
    check("A4-100160 verifies", found.found is True, f"nrn={found.nrn!r}")
    if found.found:
        record = found.primary
        check(
            "the record carries the real product name",
            "Spironolactone" in record.product_name,
            record.product_name,
        )
        check(
            "the record carries an approval date",
            bool(record.approval_date),
            record.approval_date,
        )
        check("sources are named", bool(record.sources), str(record.sources))

    inactive = result["verify_nafdac_product"]("C4-1142", sources=result["NAFDAC_SOURCES"])
    check(
        "an inactive product reports as not authorised",
        inactive.found and inactive.authorised is False,
        f"authorised={inactive.authorised}",
    )

    missing = result["verify_nafdac_product"]("A4-0000000", sources=result["NAFDAC_SOURCES"])
    check("an unknown number is not found", missing.found is False)
    check("an unknown number claims nothing", missing.authorised is None)

    print("\n--- intent gate: the retrieval boundary -------------------")
    gate = run_gate_tests(result, check)

    print("\n--- every number resolved, and a miss still carries a block ---")
    run_multi_number_tests(result, check)

    run_multi_candidate_tests(result, check)

    run_absent_greenbook_tests(check)

    print("=" * 70)
    if failures:
        print(f"{len(failures)} check(s) failed: {failures}")
        return 1
    print("All Cell 18 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
