"""N-ATLaS generation engine — standalone / Kaggle script.

Loads the N-ATLaS tokenizer, the 4-bit NF4 model, the system prompt, the
generation engine, the response cleaner and the conversation/context manager
(Cells 1-7 of natlas_notebook.py). No HTTP server is started here.

Run it directly for a smoke test:

    python kaggleserver.py

Or import it:

    from kaggleserver import generate_response
    print(generate_response([{"role": "user", "content": "Kana lafiya?"}]))
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import notebook_cells

_ENGINE = notebook_cells.exec_range(1, 7, globals())

# Re-bound explicitly so this module's public surface is greppable and static
# analysis can see it. Cell 4 loads the model, so importing this module loads it.
CONFIG = _ENGINE["CONFIG"]
GPU = _ENGINE["GPU"]
DEVICE = _ENGINE["DEVICE"]
QUANTIZATION_MODE = _ENGINE["QUANTIZATION_MODE"]
MODEL_LOAD_SECONDS = _ENGINE["MODEL_LOAD_SECONDS"]
MODEL = _ENGINE["MODEL"]
SERVER_ERROR_TYPES = (
    _ENGINE["ServerError"],
    _ENGINE["ModelUnavailable"],
    _ENGINE["ContextTooLarge"],
    _ENGINE["InvalidRequest"],
    _ENGINE["UnknownModel"],
    _ENGINE["AuthError"],
    _ENGINE["GenerationFailed"],
    _ENGINE["GPUOutOfMemory"],
    _ENGINE["QueueTimeout"],
)
ServerError = _ENGINE["ServerError"]
generate = _ENGINE["generate"]
generate_response = _ENGINE["generate_response"]
build_prompt_messages = _ENGINE["build_prompt_messages"]
prepare_context = _ENGINE["prepare_context"]
clean_response = _ENGINE["clean_response"]
gpu_memory = _ENGINE["_gpu_memory"]

__all__ = [
    "CONFIG",
    "GPU",
    "DEVICE",
    "MODEL",
    "QUANTIZATION_MODE",
    "MODEL_LOAD_SECONDS",
    "ServerError",
    "generate",
    "generate_response",
    "build_prompt_messages",
    "prepare_context",
    "clean_response",
    "run_smoke_test",
    "engine_info",
]

SMOKE_PROMPTS = [
    ("English", "How are you?"),
    ("Hausa", "Kana lafiya?"),
    ("Yoruba", "Báwo ni?"),
    ("Igbo", "Kedu ka ị mere?"),
    ("Mixed", "How are you? Yaya kake?"),
]

REFUSAL_PHRASES = (
    "i can't understand",
    "i cannot understand",
    "i do not understand",
    "unable to understand",
    "i'm sorry",
    "i am sorry",
    "as an ai",
    "i don't speak",
)


def run_smoke_test(
    prompts: Optional[List[Tuple[str, str]]] = None,
    max_new_tokens: int = 128,
) -> bool:
    """Generate one reply per prompt and report PASS/FAIL. Returns all_passed."""
    prompts = prompts or SMOKE_PROMPTS
    results: List[Tuple[str, bool, str]] = []

    for label, prompt in prompts:
        try:
            answer = generate_response(
                [{"role": "user", "content": prompt}], max_new_tokens=max_new_tokens
            )
        except ServerError as exc:
            results.append((label, False, f"{type(exc).__name__}: {exc.detail}"))
            print(f"[FAIL] {label:<8} {prompt!r} :: {type(exc).__name__}: {exc.detail}")
            continue

        refused = any(phrase in answer.lower() for phrase in REFUSAL_PHRASES)
        passed = bool(answer.strip()) and not refused
        results.append((label, passed, answer))
        print(f"[{'PASS' if passed else 'FAIL'}] {label:<8} {prompt!r}")
        print(f"       -> {answer}")

    passed_count = sum(1 for _, ok, _ in results if ok)
    print("-" * 70)
    print(f"engine smoke test: {passed_count}/{len(results)} prompts answered")
    if DEVICE == "cuda":
        print(f"quantization: {QUANTIZATION_MODE}")
        print(f"load seconds: {MODEL_LOAD_SECONDS}")
        print(f"gpu allocated: {gpu_memory()['allocated_gb']} GB")
    return passed_count == len(results)


def engine_info() -> Dict[str, object]:
    return {
        "model": CONFIG.model_id,
        "served_as": CONFIG.served_model_name,
        "device": DEVICE,
        "gpu": GPU.get("gpu_name", "none"),
        "quantization": QUANTIZATION_MODE,
        "model_load_seconds": MODEL_LOAD_SECONDS,
        "context_budget": CONFIG.context_budget,
        "max_new_tokens": CONFIG.max_new_tokens,
        "system_prompt_source": CONFIG.system_prompt_source,
        "api_key_source": CONFIG.api_key_source,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("N-ATLaS ENGINE (kaggleserver.py)")
    print("=" * 70)
    for _key, _value in engine_info().items():
        print(f"{_key:<22} {_value}")
    print("=" * 70)
    raise SystemExit(0 if run_smoke_test() else 1)
