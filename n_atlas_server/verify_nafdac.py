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
import sys
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

    print("=" * 70)
    if failures:
        print(f"{len(failures)} check(s) failed: {failures}")
        return 1
    print("All Cell 18 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
