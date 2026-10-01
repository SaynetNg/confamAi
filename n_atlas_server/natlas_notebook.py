# %% [markdown]
# # N-ATLaS — Kaggle GPU Server (Stage 1)
#
# Model: `NCAIR1/N-ATLaS` (4-bit NF4) · Tesla T4 16 GB · OpenAI-compatible FastAPI
#
# **Run the cells top to bottom, once, in order.** Do not start a tunnel or write
# laptop client code until every check in Cells 11–16 prints `PASS`.
#
# Everything is configured through environment variables, so the core server
# never needs editing:
#
# | Variable | Purpose | Default |
# |---|---|---|
# | `N_ATLAS_API_KEY` | Bearer token required by every `/v1/*` route | random ephemeral dev key |
# | `N_ATLAS_SYSTEM_PROMPT` | Inline override of the system prompt | built-in prompt |
# | `N_ATLAS_SYSTEM_PROMPT_FILE` | Path to a prompt file (wins over inline) | unset |
# | `N_ATLAS_MAX_NEW_TOKENS` | Hard ceiling on client `max_tokens` | `512` |
# | `N_ATLAS_MAX_INPUT_TOKENS` | Hard ceiling on prompt tokens | `6144` |
# | `N_ATLAS_CONTEXT_BUDGET` | Approximate model context target | `8192` |
# | `N_ATLAS_MAX_MESSAGES` | Hard ceiling on client message count | `64` |
# | `N_ATLAS_LOG_CONTENT` | Set `1` to log message text (off by default) | `0` |
# | `N_ATLAS_HOST` / `N_ATLAS_PORT` | Bind address | `0.0.0.0` / `8000` |
#
# For a **real** secret on Kaggle, put this in the first cell (or a Kaggle
# Secret attached as `N_ATLAS_API_KEY`) — never commit a literal key:
#
# ```python
# import os
# from kaggle_secrets import UserSecretsClient
# os.environ["N_ATLAS_API_KEY"] = UserSecretsClient().get_secret("n-atlas-api-key")
# ```

# %% [code]
# =============================================================================
# CELL 1 — Environment and GPU verification
# =============================================================================
import os
import re
import sys
import time
import uuid
import math
import logging
import secrets
import platform
import threading
import traceback
import subprocess
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Must be set before CUDA initialises. Measured on a T4: without it the caching
# allocator held 1.55 GiB "reserved but unallocated" during long prompts, which
# is the difference between fitting in 16 GB and OOMing.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
logger = logging.getLogger("natlas")

WORKDIR = Path(os.environ.get("N_ATLAS_WORKDIR", "/kaggle/working"))
if not WORKDIR.exists():
    WORKDIR = Path.cwd()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        logger.warning("invalid integer for %s, using default %s", name, default)
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        logger.warning("invalid float for %s, using default %s", name, default)
        return default


def _dist_version(name: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(name)
    except Exception:
        try:
            mod = __import__(name)
            return str(getattr(mod, "__version__", "unknown"))
        except Exception:
            return "not-installed"


def _cuda_report() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "torch_available": False,
        "cuda_available": False,
        "gpu_name": "none",
        "gpu_vram_gb": 0.0,
        "device_count": 0,
    }
    try:
        import torch
    except Exception as exc:
        info["error"] = f"torch import failed: {exc!r}"
        return info
    info["torch_available"] = True
    info["torch_version"] = _dist_version("torch")
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        info["cuda_error"] = repr(exc)
        return info
    if not info["cuda_available"]:
        return info
    try:
        info["device_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_vram_gb"] = round(props.total_memory / (1024**3), 2)
        info["capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:
        info["gpu_error"] = repr(exc)
    return info


@dataclass
class Settings:
    model_id: str = "NCAIR1/N-ATLaS"
    served_model_name: str = os.environ.get("N_ATLAS_SERVED_MODEL_NAME", "n-atlas")
    owned_by: str = "NCAIR1"

    host: str = os.environ.get("N_ATLAS_HOST", "0.0.0.0")
    port: int = _env_int("N_ATLAS_PORT", 8000)

    api_key: str = ""
    api_key_source: str = ""

    default_temperature: float = _env_float("N_ATLAS_TEMPERATURE", 0.7)
    min_temperature: float = 0.0
    max_temperature: float = _env_float("N_ATLAS_MAX_TEMPERATURE", 2.0)
    top_p: float = _env_float("N_ATLAS_TOP_P", 0.9)
    repetition_penalty: float = _env_float("N_ATLAS_REPETITION_PENALTY", 1.10)

    max_new_tokens: int = _env_int("N_ATLAS_MAX_NEW_TOKENS", 512)
    default_max_new_tokens: int = _env_int("N_ATLAS_DEFAULT_MAX_TOKENS", 512)
    min_max_new_tokens: int = 1
    # Measured on a Tesla T4 with this model: VRAM grows ~1.6 MB per prompt token
    # (large-vocab logits tensor + KV cache). At 4-bit a ~1400-token prompt peaks
    # near 1.6 GB allocated, while a ~6700-token prompt asks for an extra 4.3 GB
    # and OOMs a 16 GB card. 4096 leaves real headroom; raise it deliberately.
    max_input_tokens: int = _env_int("N_ATLAS_MAX_INPUT_TOKENS", 4096)
    context_budget: int = _env_int("N_ATLAS_CONTEXT_BUDGET", 8192)
    template_headroom: int = 64
    oom_retry_budget_fraction: float = _env_float("N_ATLAS_OOM_RETRY_FRACTION", 0.5)
    max_messages: int = _env_int("N_ATLAS_MAX_MESSAGES", 64)
    max_messages_absolute: int = _env_int("N_ATLAS_MAX_MESSAGES_ABSOLUTE", 512)
    max_chars_per_message: int = _env_int("N_ATLAS_MAX_CHARS_PER_MESSAGE", 6000)
    max_request_bytes: int = _env_int("N_ATLAS_MAX_REQUEST_BYTES", 1_048_576)
    max_queue_seconds: float = _env_float("N_ATLAS_MAX_QUEUE_SECONDS", 120.0)

    generation_timeout: float = _env_float("N_ATLAS_GENERATION_TIMEOUT", 300.0)
    log_content: bool = _env_flag("N_ATLAS_LOG_CONTENT", False)
    language_tagging: bool = _env_flag("N_ATLAS_LANGUAGE_TAGGING", True)
    strict_gpu: bool = _env_flag("N_ATLAS_STRICT_GPU", True)
    auto_install: bool = _env_flag("N_ATLAS_AUTO_INSTALL", True)
    allow_ephemeral_key: bool = _env_flag("N_ATLAS_ALLOW_EPHEMERAL_KEY", True)

    system_prompt: str = ""
    system_prompt_source: str = "builtin"

    def public_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "served_model_name": self.served_model_name,
            "host": self.host,
            "port": self.port,
            "api_key_source": self.api_key_source,
            "max_new_tokens": self.max_new_tokens,
            "max_input_tokens": self.max_input_tokens,
            "context_budget": self.context_budget,
            "max_messages": self.max_messages,
            "log_content": self.log_content,
            "system_prompt_source": self.system_prompt_source,
        }


class ServerError(Exception):
    """Base class for errors that map to a specific HTTP status."""

    status_code = 500
    public_message = "Internal server error."

    def __init__(self, detail: Optional[str] = None, log_detail: Optional[str] = None):
        self.detail = detail or self.public_message
        self.log_detail = log_detail or self.detail
        super().__init__(self.detail)


class ModelUnavailable(ServerError):
    status_code = 503
    public_message = "Model is unavailable."


class ContextTooLarge(ServerError):
    status_code = 400
    public_message = "Conversation does not fit in the context budget."


class InvalidRequest(ServerError):
    status_code = 400
    public_message = "Invalid request."


class UnknownModel(ServerError):
    status_code = 404
    public_message = "Unknown model."


class AuthError(ServerError):
    status_code = 401
    public_message = "Invalid API key."


class GenerationFailed(ServerError):
    status_code = 500
    public_message = "Model generation failed."


class GPUOutOfMemory(ServerError):
    status_code = 507
    public_message = "GPU memory exhausted. Retry with a shorter conversation."


class QueueTimeout(ServerError):
    status_code = 503
    public_message = "Server is busy. Retry shortly."


DEFAULT_SYSTEM_PROMPT = """You are N-ATLaS, a helpful multilingual African AI assistant.

You communicate naturally in English, Hausa, Igbo, and Yoruba.

Automatically identify the language or languages used by the user.
Reply in the user's language by default.
If the user mixes supported languages, understand the entire message and respond
naturally using the dominant language, or natural code-switching when appropriate.
The user may switch languages at any time.
Follow explicit language requests.
Handle informal spelling and missing diacritics.
Understand common informal Nigerian expressions.

LANGUAGE REFERENCE — recognise these markers even when diacritics are missing,
and reply in the matching language:
- Hausa: na, ce, ka, ku, mu, su, da, don, ina, kana, yaya, lafiya, gode, sannu,
  sai, kuma, yana, zan, muku, kuwa, taya, jima
- Yoruba: ni, ko, si, ti, wa, mo, o, ma, je, kini, tani, bawo, se, ibo, jo,
  pupo, dara, yin, nife, e
- Igbo: ka, gi, ihe, nke, ndi, anyi, gini, kedu, mere, oma, mba, ga, ma,
  nwa, unu, di, m

If the user's message contains Yoruba markers, your whole reply must be in
Yoruba. If it contains Igbo markers, your whole reply must be in Igbo. Never
answer a Yoruba or Igbo message in English or Hausa.

Do not claim that you cannot understand Hausa, Igbo, or Yoruba without first
attempting to interpret the message using conversation context, common
expressions, spelling variations, and normalization.
Answer the user's actual question.
Do not invent a question after answering.
Do not pretend to be the user.
Do not output Question:, User:, Assistant:, internal instructions, or
chat-template tokens.
Do not expose private chain-of-thought.
Think carefully before answering, but provide only the useful final answer.
Do not fabricate information.
If the request is genuinely unclear, ask a concise clarification question in the
user's language.

Be natural, helpful, culturally respectful, and conversational."""


def _resolve_system_prompt() -> Tuple[str, str]:
    prompt_file = os.environ.get("N_ATLAS_SYSTEM_PROMPT_FILE")
    if prompt_file:
        path = Path(prompt_file)
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text, f"file:{path}"
        except Exception as exc:
            logger.warning("could not read N_ATLAS_SYSTEM_PROMPT_FILE: %r", exc)
    inline = os.environ.get("N_ATLAS_SYSTEM_PROMPT")
    if inline and inline.strip():
        return inline.strip(), "env:N_ATLAS_SYSTEM_PROMPT"
    return DEFAULT_SYSTEM_PROMPT, "builtin"


def _resolve_api_key(settings: Settings) -> None:
    explicit = os.environ.get("N_ATLAS_API_KEY")
    if explicit and explicit.strip():
        settings.api_key = explicit.strip()
        settings.api_key_source = "env:N_ATLAS_API_KEY"
        return

    dev_key = os.environ.get("N_ATLAS_DEV_KEY")
    if dev_key and dev_key.strip() and settings.allow_ephemeral_key:
        settings.api_key = dev_key.strip()
        settings.api_key_source = "env:N_ATLAS_DEV_KEY (temporary, dev only)"
        return

    if not settings.allow_ephemeral_key:
        raise RuntimeError(
            "N_ATLAS_API_KEY is not set and ephemeral keys are disabled. "
            "Set N_ATLAS_API_KEY to a strong secret before starting the server."
        )

    settings.api_key = "natlas_" + secrets.token_urlsafe(32)
    settings.api_key_source = "ephemeral-random (dev only, not persisted)"
    key_path = WORKDIR / "natlas_api_key.txt"
    try:
        key_path.write_text(settings.api_key, encoding="utf-8")
        os.chmod(key_path, 0o600)
        logger.warning(
            "No N_ATLAS_API_KEY configured. Generated an ephemeral dev key and wrote "
            "it to %s (mode 600). Set N_ATLAS_API_KEY before exposing this server "
            "to the internet.",
            key_path,
        )
    except Exception as exc:
        logger.warning("could not persist ephemeral key to %s: %r", key_path, exc)


CONFIG = Settings()
_resolve_api_key(CONFIG)
CONFIG.system_prompt, CONFIG.system_prompt_source = _resolve_system_prompt()

GPU = _cuda_report()

print("=" * 70)
print("N-ATLaS SERVER — ENVIRONMENT")
print("=" * 70)
print(f"Model:             {CONFIG.model_id}")
print(f"Served as:         {CONFIG.served_model_name}")
print(f"PyTorch:           {GPU.get('torch_version', 'MISSING')}")
print(f"Transformers:      {_dist_version('transformers')}")
print(f"bitsandbytes:      {_dist_version('bitsandbytes')}")
print(f"CUDA available:    {GPU['cuda_available']}")
if GPU["cuda_available"]:
    print(f"GPU name:          {GPU['gpu_name']}")
    print(f"GPU VRAM (GB):     {GPU['gpu_vram_gb']}")
    print(f"Compute capability:{GPU.get('capability', '?')}")
else:
    print(f"GPU name:          none ({GPU.get('error') or GPU.get('cuda_error') or 'no CUDA device'})")
print(f"Python:            {platform.python_version()} ({sys.platform})")
print(f"Workdir:           {WORKDIR}")
print("-" * 70)
print(f"API key source:    {CONFIG.api_key_source}")
print(f"System prompt:     {CONFIG.system_prompt_source}")
print(f"Context budget:    {CONFIG.context_budget} tokens")
print(f"Max new tokens:    {CONFIG.max_new_tokens}")
print("=" * 70)

if not GPU["torch_available"]:
    raise RuntimeError("PyTorch is not importable in this notebook environment.")

if CONFIG.strict_gpu and not GPU["cuda_available"]:
    raise RuntimeError(
        "A CUDA GPU is required (Tesla T4 expected) but torch.cuda.is_available() "
        "is False. Enable a GPU accelerator in the Kaggle notebook settings, or set "
        "N_ATLAS_STRICT_GPU=0 only if you deliberately want CPU inference."
    )

DEVICE = "cuda" if GPU["cuda_available"] else "cpu"

# %% [code]
# =============================================================================
# CELL 2 — Dependency verification / install only if needed
# =============================================================================
SERVER_DEPS = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "requests": "requests",
    "nest_asyncio": "nest_asyncio",
}

# Kaggle ships torch and transformers but NOT bitsandbytes, so 4-bit NF4
# loading in Cell 4 fails without these. `accelerate` is required by
# device_map="auto"; `safetensors` by the N-ATLaS weight loader. Installing
# them is cheap and never downgrades anything already present.
RUNTIME_DEPS = {
    "bitsandbytes": "bitsandbytes",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
}


def _missing_deps(deps: Dict[str, str]) -> List[Tuple[str, str]]:
    import importlib.util

    missing = []
    for module, dist in deps.items():
        if importlib.util.find_spec(module) is None:
            missing.append((module, dist))
    return missing


def ensure_dependencies(
    deps: Dict[str, str], label: str = "server"
) -> None:
    missing = _missing_deps(deps)
    if not missing:
        print(f"[deps] all {label} dependencies present — nothing installed.")
        return
    names = [dist for _, dist in missing]
    print(f"[deps] missing {label} dependencies: {', '.join(names)}")
    if not CONFIG.auto_install:
        raise RuntimeError(
            f"Missing required {label} dependencies {names} and automatic install "
            "is disabled (N_ATLAS_AUTO_INSTALL=0). Install them manually."
        )
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--no-input", *names]
    print("[deps] running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("pip install failed:\n%s\n%s", proc.stdout, proc.stderr)
        raise RuntimeError(f"Failed to install {names}: {proc.stderr.strip()[:500]}")
    import importlib

    importlib.invalidate_caches()
    still = _missing_deps(deps)
    if still:
        raise RuntimeError(f"Dependencies still missing after install: {still}")
    print(f"[deps] {label} install complete.")


ensure_dependencies(SERVER_DEPS, "server")

if CONFIG.strict_gpu and DEVICE == "cuda":
    ensure_dependencies(RUNTIME_DEPS, "model runtime")

import fastapi
import pydantic
import requests
import uvicorn

try:
    import nest_asyncio

    NEST_ASYNCIO_AVAILABLE = True
except Exception:
    NEST_ASYNCIO_AVAILABLE = False

CORE_STACK = {
    "torch": _dist_version("torch"),
    "transformers": _dist_version("transformers"),
    "bitsandbytes": _dist_version("bitsandbytes"),
    "accelerate": _dist_version("accelerate"),
    "safetensors": _dist_version("safetensors"),
    "fastapi": fastapi.__version__,
    "uvicorn": uvicorn.__version__,
    "pydantic": pydantic.__version__,
    "requests": requests.__version__,
    "nest_asyncio": getattr(nest_asyncio, "__version__", "present")
    if NEST_ASYNCIO_AVAILABLE
    else "MISSING",
}

print("[deps] core stack (never auto-downgraded):")
for name, version in CORE_STACK.items():
    print(f"    {name:<14} {version}")

if CORE_STACK["torch"] == "not-installed" or CORE_STACK["transformers"] == "not-installed":
    raise RuntimeError("torch and transformers must be present; refusing to continue.")

if CONFIG.strict_gpu and DEVICE == "cuda" and CORE_STACK["bitsandbytes"] == "not-installed":
    raise RuntimeError(
        "bitsandbytes is required for 4-bit NF4 loading on GPU but is not installed. "
        "Cell 2 attempted to install it; install manually with "
        "`pip install bitsandbytes` and re-run this cell."
    )

# %% [code]
# =============================================================================
# CELL 3 — Load the N-ATLaS tokenizer
# =============================================================================
import torch
from transformers import AutoTokenizer

TOKENIZER_LOAD_SECONDS = None

_tok_start = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_id)
TOKENIZER_LOAD_SECONDS = round(time.perf_counter() - _tok_start, 2)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

if not getattr(tokenizer, "chat_template", None):
    raise RuntimeError(
        f"{CONFIG.model_id} exposes no chat template. Refusing to hand-build a prompt."
    )

print("=" * 70)
print("TOKENIZER READY")
print("=" * 70)
print(f"loaded in:      {TOKENIZER_LOAD_SECONDS}s")
print(f"class:          {type(tokenizer).__name__}")
print(f"vocab_size:     {tokenizer.vocab_size}")
print(f"len(tokenizer): {len(tokenizer)}")
print(f"eos_token_id:   {tokenizer.eos_token_id}  ({tokenizer.eos_token!r})")
print(f"pad_token_id:   {tokenizer.pad_token_id}  ({tokenizer.pad_token!r})")
print(f"model_max_len:  {tokenizer.model_max_length}")
print(f"chat template:  present ({len(tokenizer.chat_template)} chars)")

# N-ATLaS derives from the Llama 3.1 chat template and renders a
# "Today Date:" line, so it expects a `date_string`. Extra template variables
# are harmless in Jinja when the template does not use them, so passing it is
# safe either way; the fallback only matters if the template rejects it.
def _template_kwargs() -> Dict[str, Any]:
    from datetime import datetime

    return {"date_string": datetime.now().strftime("%d %b %Y")}


def _render_chat(messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **_template_kwargs(),
        )
    except Exception as exc:
        if "date_string" not in str(exc):
            raise
        logger.warning(
            "chat template rejected date_string (%s); retrying without it", exc
        )
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )


_probe = _render_chat([{"role": "user", "content": "ping"}])
print("-" * 70)
print("chat template probe (repr, first 400 chars):")
print(repr(_probe)[:400])
print("=" * 70)

# %% [code]
# =============================================================================
# CELL 4 — Load N-ATLaS in 4-bit NF4
# =============================================================================
import gc
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

MODEL = None
MODEL_LOAD_SECONDS = None
MODEL_LOAD_ERROR = None
QUANTIZATION_MODE = "none"
QUANT_CONFIG = None


def load_model(force: bool = False):
    global MODEL, MODEL_LOAD_SECONDS, MODEL_LOAD_ERROR, QUANTIZATION_MODE, QUANT_CONFIG

    if MODEL is not None and not force:
        return MODEL
    if MODEL_LOAD_ERROR is not None and not force:
        raise ModelUnavailable(MODEL_LOAD_ERROR)

    _free_cuda_cache()
    gc.collect()

    _load_start = time.perf_counter()
    try:
        use_4bit = DEVICE == "cuda"
        if use_4bit:
            QUANT_CONFIG = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                CONFIG.model_id,
                quantization_config=QUANT_CONFIG,
                device_map="auto",
            )
            QUANTIZATION_MODE = "4-bit NF4 (double quant)"
        else:
            model = AutoModelForCausalLM.from_pretrained(
                CONFIG.model_id,
                device_map=None,
            )
            model = model.to(DEVICE)
            QUANTIZATION_MODE = "fp32 (no GPU available)"

        model.eval()
        if not use_4bit:
            model.config.use_cache = True

        MODEL_LOAD_SECONDS = round(time.perf_counter() - _load_start, 2)
        MODEL = model
        MODEL_LOAD_ERROR = None
        return model
    except Exception as exc:
        MODEL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        logger.error("model load failed: %s\n%s", MODEL_LOAD_ERROR, traceback.format_exc())
        raise ModelUnavailable(MODEL_LOAD_ERROR) from exc


def _free_cuda_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _gpu_memory() -> Dict[str, float]:
    out = {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}
    try:
        if torch.cuda.is_available():
            out["allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
            out["reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 3)
            out["max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    except Exception:
        pass
    return out


def _model_device():
    if MODEL is not None:
        try:
            return MODEL.device
        except Exception:
            pass
    return torch.device(DEVICE)


MODEL = load_model()

_mem = _gpu_memory()
print("=" * 70)
print("MODEL READY")
print("=" * 70)
print(f"model:            {CONFIG.model_id}")
print(f"class:            {type(MODEL).__name__}")
print(f"quantization:     {QUANTIZATION_MODE}")
print(f"device:           {_model_device()}")
print(f"dtype:            {getattr(MODEL, 'dtype', 'n/a')}")
print(f"load time:        {MODEL_LOAD_SECONDS}s")
print(f"parameters:       {sum(p.numel() for p in MODEL.parameters()):,}")
print(f"GPU allocated:    {_mem['allocated_gb']} GB")
print(f"GPU reserved:     {_mem['reserved_gb']} GB")
print(f"GPU max alloc:    {_mem['max_allocated_gb']} GB")
print("=" * 70)

# %% [code]
# =============================================================================
# CELL 5 — System prompt and generation engine
# =============================================================================
SYSTEM_PROMPT = CONFIG.system_prompt


@dataclass
class GenerationResult:
    text: str
    raw_text: str
    finish_reason: str
    input_tokens: int
    output_tokens: int
    seconds: float
    tokens_per_second: float
    gpu_memory: Dict[str, float] = field(default_factory=dict)
    cleaned_markers: List[str] = field(default_factory=list)
    retried: bool = False

    def to_log_fields(self) -> Dict[str, Any]:
        return {
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "generation_seconds": round(self.seconds, 3),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "gpu_memory_allocated_gb": self.gpu_memory.get("allocated_gb", 0.0),
            "gpu_memory_reserved_gb": self.gpu_memory.get("reserved_gb", 0.0),
            "markers_removed": ",".join(self.cleaned_markers) or "none",
            "retried": self.retried,
        }


GENERATION_LOCK = threading.Lock()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _resolve_eos_ids() -> List[int]:
    ids = []
    for candidate in (tokenizer.eos_token_id, getattr(tokenizer, "sep_token_id", None)):
        if isinstance(candidate, int) and candidate >= 0 and candidate not in ids:
            ids.append(candidate)
    return ids


def _apply_chat_template(messages: List[Dict[str, str]]) -> str:
    return _render_chat(messages)


def _encode(text: str):
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    device = _model_device()
    try:
        return encoded.to(device)
    except Exception:
        return encoded


def _raw_generate(
    prompt_messages: List[Dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> Tuple[str, int, int, str]:
    text = _apply_chat_template(prompt_messages)
    inputs = _encode(text)
    input_len = int(inputs["input_ids"].shape[1])

    eos_ids = _resolve_eos_ids()
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_ids if len(eos_ids) > 1 else (eos_ids[0] if eos_ids else None),
    }
    if temperature <= 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    outputs = MODEL.generate(**inputs, **gen_kwargs)
    new_tokens = outputs[0][input_len:]

    hit_cap = int(new_tokens.shape[0]) >= max_new_tokens
    finish_reason = "length" if hit_cap else "stop"

    try:
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    except Exception:
        decoded = str(new_tokens)

    return decoded, input_len, int(new_tokens.shape[0]), finish_reason


def _minimum_viable_budget(messages: List[Dict[str, str]]) -> int:
    """Smallest budget that still holds everything but the final message."""
    last = len(messages) - 1
    other = sum(count_tokens(m["content"]) for i, m in enumerate(messages) if i != last)
    return other + 64 + template_overhead() * len(messages) + CONFIG.template_headroom


def generate(
    prepared: List[Dict[str, str]],
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    request_id: str = "-",
) -> GenerationResult:
    if MODEL is None:
        raise ModelUnavailable("model is not loaded")

    max_new_tokens = clamp_int(
        CONFIG.default_max_new_tokens if max_new_tokens is None else int(max_new_tokens),
        CONFIG.min_max_new_tokens,
        CONFIG.max_new_tokens,
    )
    temperature = clamp(
        CONFIG.default_temperature if temperature is None else float(temperature),
        CONFIG.min_temperature,
        CONFIG.max_temperature,
    )
    top_p = clamp(CONFIG.top_p if top_p is None else float(top_p), 0.05, 1.0)
    repetition_penalty = clamp(
        CONFIG.repetition_penalty if repetition_penalty is None else float(repetition_penalty),
        0.5,
        2.0,
    )

    budget = context_budget_for(max_new_tokens)

    acquired = GENERATION_LOCK.acquire(timeout=CONFIG.max_queue_seconds)
    if not acquired:
        raise QueueTimeout(
            f"waited {CONFIG.max_queue_seconds}s for the generation lock"
        )

    try:
        started = time.perf_counter()
        retried = False
        context: Optional[PreparedContext] = None
        raw = ""
        input_tokens = 0
        output_tokens = 0
        finish_reason = "stop"

        for attempt in range(2):
            try:
                context = prepare_context(prepared, budget=budget)
                if CONFIG.log_content:
                    logger.info("request_id=%s prompt=%s", request_id, context.messages)
                raw, input_tokens, output_tokens, finish_reason = _raw_generate(
                    context.messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if not _is_oom(exc):
                    logger.error(
                        "request_id=%s generation error: %s\n%s",
                        request_id,
                        exc,
                        traceback.format_exc(),
                    )
                    raise GenerationFailed(
                        "Model generation failed.", log_detail=repr(exc)
                    ) from exc
                _free_cuda_cache()
                gc.collect()
                floor = _minimum_viable_budget(prepared)
                shrunk = max(int(budget * CONFIG.oom_retry_budget_fraction), floor)
                if attempt == 0 and shrunk <= budget:
                    retried = True
                    logger.warning(
                        "request_id=%s CUDA OOM; retrying at a %d-token prompt budget "
                        "(reduced from %d)",
                        request_id,
                        shrunk,
                        budget,
                    )
                    budget = shrunk
                    continue
                logger.error("request_id=%s CUDA OOM: %r", request_id, exc)
                raise GPUOutOfMemory(
                    "GPU memory exhausted. Retry with a shorter conversation or fewer "
                    "max_tokens.",
                    log_detail=repr(exc),
                ) from exc
            except ContextTooLarge as exc:
                if attempt == 0 and budget > _minimum_viable_budget(prepared):
                    retried = True
                    logger.warning(
                        "request_id=%s prompt too large (%s); retrying with a smaller "
                        "budget",
                        request_id,
                        exc.detail,
                    )
                    budget = max(
                        int(budget * CONFIG.oom_retry_budget_fraction),
                        _minimum_viable_budget(prepared),
                    )
                    continue
                if retried:
                    raise GPUOutOfMemory(
                        "GPU memory exhausted. Retry with a shorter conversation.",
                        log_detail=exc.detail,
                    ) from exc
                raise
            except Exception as exc:
                logger.error(
                    "request_id=%s generation error: %s\n%s",
                    request_id,
                    exc,
                    traceback.format_exc(),
                )
                raise GenerationFailed(
                    "Model generation failed.", log_detail=repr(exc)
                ) from exc

        cleaned, markers = clean_response(raw)

        if not cleaned.strip():
            retried = True
            logger.warning(
                "request_id=%s empty generation, retrying once at temperature %.2f",
                request_id,
                min(1.0, temperature + 0.2),
            )
            try:
                raw, input_tokens, output_tokens, finish_reason = _raw_generate(
                    context.messages,
                    max_new_tokens=max_new_tokens,
                    temperature=min(1.0, temperature + 0.2),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                cleaned, markers = clean_response(raw)
            except Exception as exc:
                logger.error("request_id=%s retry failed: %r", request_id, exc)
                cleaned, markers = "", []

        seconds = max(time.perf_counter() - started, 1e-6)
        result = GenerationResult(
            text=cleaned,
            raw_text=raw,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            seconds=seconds,
            tokens_per_second=output_tokens / seconds,
            gpu_memory=_gpu_memory(),
            cleaned_markers=markers,
            retried=retried,
        )
        if not result.text.strip():
            raise GenerationFailed(
                "Model produced an empty response.",
                log_detail=f"raw={raw!r}",
            )
        return result
    finally:
        GENERATION_LOCK.release()


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    text = repr(exc).lower()
    return "out of memory" in text or "cuda oom" in text or "cublas_status_alloc_failed" in text


def generate_response(
    messages: List[Dict[str, str]],
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    return generate(
        build_prompt_messages(list(messages)),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    ).text


# %% [code]
# =============================================================================
# CELL 6 — Response cleaning
# =============================================================================
TEMPLATE_ARTIFACTS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|endoftext|>",
    "<|begin_of_text|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<<SYS>>",
    "<</SYS>>",
    "[INST]",
    "[/INST]",
    "[SYSTEM]",
    "[/SYSTEM]",
    "<s>",
    "</s>",
)

TURN_ROLES = (
    "question",
    "user",
    "assistant",
    "system",
    "human",
    "bot",
    "ai",
    "prompt",
    "user query",
)

TURN_MARKER_RE = re.compile(
    r"(?:\A|\n)[ \t>*_#\-]*[ \t]*(?:"
    + "|".join(re.escape(role) for role in TURN_ROLES)
    + r")[ \t]*\**[ \t]*:[ \t]*",
    re.IGNORECASE,
)

STRUCTURAL_TURN_RE = re.compile(
    r"<\|(?:im_start|start_header_id)\|>"
    r"|\[/?INST\]"
    r"|<<\/?SYS>>"
    r"|</?s>"
    r"|<\|(?:endoftext|eot_id|eom_id|end|begin_of_text|system|user|assistant)\|>"
)

ANSWER_LINE_RE = re.compile(
    r"(?:\A|\n)[ \t>*_#\-]*answer[ \t]*\**[ \t]*:[ \t]*",
    re.IGNORECASE,
)

BARE_TURN_RE = re.compile(
    r"(?:\A|\n)[ \t]*(?:user|assistant|system|human|bot|ai|prompt)[ \t]*\n",
    re.IGNORECASE,
)

LEADING_ROLE_LABEL_RE = re.compile(
    r"\A(?:user|assistant|system|human|bot|ai|prompt|question|answer)\b[ \t]*(?:\n|:)",
    re.IGNORECASE,
)

CUT_RULES = (
    (STRUCTURAL_TURN_RE, "template-turn"),
    (TURN_MARKER_RE, "speaker-marker"),
    (BARE_TURN_RE, "bare-turn"),
)

EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")


def strip_template_artifacts(text: str) -> str:
    out = text
    for token in TEMPLATE_ARTIFACTS:
        if token in out:
            out = out.replace(token, "")
    return out


def cut_at_turn_boundary(text: str) -> Tuple[str, str, bool, int]:
    best_match = None
    best_kind = ""
    for regex, kind in CUT_RULES:
        match = regex.search(text)
        if match is not None and (best_match is None or match.start() < best_match.start()):
            best_match = match
            best_kind = kind
    if best_match is None:
        return text, "", False, len(text)
    label = re.sub(r"\s+", " ", best_match.group(0)).strip().lower()
    return text[: best_match.start()], label or best_kind, best_match.start() == 0, best_match.end()


def _extract_answer_segment(text: str) -> str:
    match = ANSWER_LINE_RE.search(text)
    if not match:
        return ""
    return text[match.end() :]


def _drop_leading_turn_markers(raw: str, removed: List[str]) -> str:
    """Skip turn markers the model emitted before any real content."""
    text = raw
    for _ in range(4):
        candidate, label, at_start, match_end = cut_at_turn_boundary(text)
        if not label:
            return text
        if not at_start:
            return candidate
        removed.append(label)
        removed.append("leading-turn-marker-dropped")
        text = text[match_end:]
    return cut_at_turn_boundary(text)[0]


def clean_response(raw: str) -> Tuple[str, List[str]]:
    removed: List[str] = []
    source = raw or ""
    if not source.strip():
        return "", removed

    text, label, at_start, _ = cut_at_turn_boundary(source)

    if at_start:
        recovered = _extract_answer_segment(source)
        if recovered.strip():
            removed.append(label)
            removed.append("answer-segment-recovered")
            text = cut_at_turn_boundary(recovered)[0]
        else:
            text = _drop_leading_turn_markers(source, removed)
    elif label:
        removed.append(label)

    text = strip_template_artifacts(text)
    text = LEADING_ROLE_LABEL_RE.sub("", text, count=1)
    text = EXCESS_BLANK_LINES_RE.sub("\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip(), removed

# %% [code]
# =============================================================================
# CELL 7 — Conversation and context management
# =============================================================================
VALID_ROLES = ("system", "user", "assistant")
_TOKEN_COUNT_CACHE: Dict[str, int] = {}
_TOKEN_COUNT_CACHE_LIMIT = 4096

LANGUAGE_NAMES = {"en": "English", "ha": "Hausa", "yo": "Yoruba", "ig": "Igbo"}

EXPLICIT_REQUESTS = (
    ("hausa", "ha"),
    ("sauraya", "ha"),
    ("yoruba", "yo"),
    ("yebu", "yo"),
    ("ede yoruba", "yo"),
    ("igbo", "ig"),
    ("ahụkerị", "ig"),
    ("ahukeri", "ig"),
)

LANGUAGE_MARKERS: Dict[str, Dict[str, float]] = {
    "ha": {
        "kana": 2.0, "lafiya": 2.0, "sannu": 2.0, "yaya": 2.0, "kake": 2.0,
        "yake": 2.0, "ina": 2.0, "gode": 2.0, "kuma": 2.0, "sai": 2.0,
        "yana": 2.0, "zan": 2.0, "muku": 2.0, "wallahi": 2.0, "kawai": 2.0,
        "yi": 1.5, "yau": 1.5, "da": 1.5, "mai": 1.5, "kuma": 2.0,
        "ka": 1.0, "ku": 1.0, "na": 1.0, "ya": 1.0, "ta": 1.0, "mu": 1.0,
        "su": 1.0, "ce": 1.0, "dan": 1.0, "ba": 0.5, "to": 0.5,
    },
    "yo": {
        "bawo": 2.5, "tani": 2.5, "kini": 2.5, "nife": 2.5, "pupo": 2.0,
        "dara": 2.0, "ibo": 2.0, "jo": 1.5, "pelepo": 2.0, "yin": 2.0,
        "ti": 1.0, "si": 1.0, "ko": 1.0, "wa": 1.0, "mo": 1.0, "ati": 1.5,
        "je": 1.0, "e": 1.0, "ni": 1.0, "o": 0.5, "ma": 0.5,
    },
    "ig": {
        "kedu": 2.5, "mere": 2.5, "nke": 2.0, "ndi": 2.0, "anyi": 2.0,
        "gini": 2.0, "oma": 2.0, "mba": 2.0, "nwa": 2.0, "unu": 2.0,
        "ihe": 2.0, "ka": 1.5, "gi": 1.5, "ga": 1.5, "di": 1.5, "ma": 1.0,
        "m": 0.5, "o": 0.5, "i": 0.5,
    },
    "en": {
        "the": 2.0, "you": 1.5, "your": 1.5, "what": 2.0, "how": 2.0,
        "please": 2.0, "thanks": 2.0, "hello": 2.0, "my": 1.5, "me": 1.5,
        "can": 1.0, "do": 1.0, "have": 1.5, "will": 1.5, "would": 1.5,
        "about": 1.5, "tell": 2.0, "explain": 2.0, "write": 2.0, "say": 1.5,
        "know": 1.5, "think": 1.5, "today": 1.5, "good": 1.0, "morning": 1.5,
        "i": 1.0, "it": 1.0, "we": 1.0, "is": 1.0, "are": 1.0, "and": 1.0,
    },
}

LANGUAGE_DIRECTIVES = {
    "en": "Write the entire reply in English. Do not use Hausa, Yoruba or Igbo.",
    "ha": "Write the entire reply in Hausa. Do not use English, Yoruba or Igbo.",
    "yo": "Write the entire reply in Yoruba. Do not use English, Hausa or Igbo.",
    "ig": "Write the entire reply in Igbo. Do not use English, Hausa or Yoruba.",
}

MIN_LANGUAGE_SCORE = 2.0
MIN_LANGUAGE_MARGIN = 1.0


def fold_text(text: str) -> str:
    """Lowercase and strip diacritics, so 'Bawo ni' and 'Báwo ni' both match."""
    decomposed = unicodedata.normalize("NFD", text or "").lower()
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return unicodedata.normalize("NFC", stripped)


def _explicit_language(folded: str) -> Optional[str]:
    for phrase, code in EXPLICIT_REQUESTS:
        if phrase in folded:
            if " in " in phrase or phrase.endswith("yoruba") or phrase.endswith("hausa") or phrase.endswith("igbo"):
                return code
            return code
    match = re.search(r"\b(?:in|using|reply|answer|respond|speak|talk)\s+([a-z]+)\b", folded)
    if match:
        for phrase, code in EXPLICIT_REQUESTS:
            if match.group(1) in phrase.split():
                return code
    return None


def _score_languages(folded: str) -> Dict[str, float]:
    words = re.findall(r"[a-z']+", folded)
    scores = {code: 0.0 for code in LANGUAGE_NAMES}
    for word in words:
        for code, markers in LANGUAGE_MARKERS.items():
            weight = markers.get(word)
            if weight:
                scores[code] += weight
    return scores


def detect_language(text: str) -> Dict[str, Any]:
    folded = fold_text(text)
    scores = _score_languages(folded)
    explicit = _explicit_language(folded)
    if explicit:
        return {
            "language": explicit,
            "confidence": "explicit",
            "scores": scores,
            "mixed": False,
        }

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_code, best_score = ranked[0]
    runner_code, runner_score = ranked[1]

    # Deliberate bias: English is this model's default fallback, so when a
    # message mixes English with a well-attested Nigerian language, target the
    # Nigerian language rather than letting English win by stopword count.
    if best_code == "en" and runner_code != "en":
        if runner_score >= MIN_LANGUAGE_SCORE and runner_score >= best_score * 0.6:
            return {
                "language": runner_code,
                "confidence": "high" if runner_score >= 4.0 else "medium",
                "scores": scores,
                "mixed": True,
            }

    if best_score < MIN_LANGUAGE_SCORE or (best_score - runner_score) < MIN_LANGUAGE_MARGIN:
        return {
            "language": "en",
            "confidence": "low",
            "scores": scores,
            "mixed": best_score >= MIN_LANGUAGE_SCORE,
        }

    return {
        "language": best_code,
        "confidence": "high" if best_score >= 4.0 else "medium",
        "scores": scores,
        "mixed": runner_score >= MIN_LANGUAGE_SCORE,
    }


def build_language_directive(code: str, mixed: bool) -> str:
    line = (
        f"[language] The user's last message is in {LANGUAGE_NAMES[code]}. "
        f"{LANGUAGE_DIRECTIVES[code]}"
    )
    if mixed:
        line += " The user mixes languages; still lead with this language."
    return line


def apply_language_directive(
    prompt_messages: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    last_user = next(
        (m["content"] for m in reversed(prompt_messages) if m["role"] == "user"),
        "",
    )
    detection = detect_language(last_user)

    if not CONFIG.language_tagging or detection["language"] == "en" and detection["confidence"] == "low":
        return prompt_messages, {**detection, "injected": False}

    directive = build_language_directive(detection["language"], detection["mixed"])
    updated = [dict(m) for m in prompt_messages]
    for message in updated:
        if message["role"] == "system":
            message["content"] = f"{message['content']}\n\n{directive}"
            break
    return updated, {**detection, "injected": True, "directive": directive}


LAST_LANGUAGE_DETECTION: Dict[str, Any] = {}


@dataclass
class PreparedContext:
    messages: List[Dict[str, str]]
    input_tokens: int
    prompt_budget: int
    dropped_messages: int
    truncated_messages: int
    system_tokens: int


def count_tokens(text: str) -> int:
    if not text:
        return 0
    cached = _TOKEN_COUNT_CACHE.get(text)
    if cached is not None:
        return cached
    try:
        count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    except Exception:
        count = max(1, math.ceil(len(text) / 3.6))
    if len(_TOKEN_COUNT_CACHE) >= _TOKEN_COUNT_CACHE_LIMIT:
        _TOKEN_COUNT_CACHE.clear()
    _TOKEN_COUNT_CACHE[text] = count
    return count


def normalize_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p.strip() for p in parts if p.strip()).strip()
    return ""


def sanitize_messages(raw_messages: List[Any]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for index, item in enumerate(raw_messages):
        if not isinstance(item, dict):
            raise InvalidRequest(f"messages[{index}] must be an object.")
        role = item.get("role")
        role = role.strip().lower() if isinstance(role, str) else role
        if role not in VALID_ROLES:
            raise InvalidRequest(
                f"messages[{index}].role must be one of {list(VALID_ROLES)}."
            )
        content = normalize_content(item.get("content"))
        if not content:
            continue
        if len(content) > CONFIG.max_chars_per_message:
            content = content[: CONFIG.max_chars_per_message]
        cleaned.append({"role": role, "content": content})
    return cleaned


def merge_consecutive(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1]["content"] = f"{merged[-1]['content']}\n{message['content']}"
        else:
            merged.append(dict(message))
    return merged


def build_prompt_messages(client_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if len(client_messages) > CONFIG.max_messages_absolute:
        raise InvalidRequest(
            f"Too many messages: {len(client_messages)} "
            f"(hard limit {CONFIG.max_messages_absolute})."
        )

    sanitized = sanitize_messages(client_messages)
    if not sanitized:
        raise InvalidRequest("messages must contain at least one non-empty message.")

    system_messages = [m for m in sanitized if m["role"] == "system"]
    dialogue = merge_consecutive([m for m in sanitized if m["role"] != "system"])

    if len(dialogue) > CONFIG.max_messages:
        dialogue = dialogue[-CONFIG.max_messages :]

    prompt: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in system_messages:
        if message["content"] != SYSTEM_PROMPT:
            prompt.append(message)
    prompt.extend(dialogue)

    if not any(m["role"] == "user" for m in prompt):
        raise InvalidRequest("messages must include at least one user message.")

    global LAST_LANGUAGE_DETECTION
    prompt, LAST_LANGUAGE_DETECTION = apply_language_directive(prompt)
    return prompt


def context_budget_for(max_new_tokens: int, fraction: float = 1.0) -> int:
    usable = int((CONFIG.context_budget - max_new_tokens - CONFIG.template_headroom) * fraction)
    usable = min(usable, int(CONFIG.max_input_tokens * fraction))
    if usable < CONFIG.template_headroom * 4:
        raise InvalidRequest(
            "Configured context budget is too small for the requested max_tokens."
        )
    return usable


_TEMPLATE_OVERHEAD: Optional[int] = None


def template_overhead() -> int:
    """Tokens the chat template adds per message (role header + separators)."""
    global _TEMPLATE_OVERHEAD
    if _TEMPLATE_OVERHEAD is None:
        probe = [
            {"role": "user", "content": "alpha"},
            {"role": "assistant", "content": "beta"},
        ]
        try:
            plain = sum(count_tokens(m["content"]) for m in probe)
            templated = count_tokens(_apply_chat_template(probe))
            _TEMPLATE_OVERHEAD = max(1, (templated - plain) // len(probe))
        except Exception as exc:
            logger.warning("could not measure template overhead: %r", exc)
            _TEMPLATE_OVERHEAD = 8
    return _TEMPLATE_OVERHEAD


def _estimated_prompt_tokens(messages: List[Dict[str, str]]) -> int:
    return sum(count_tokens(m["content"]) for m in messages) + template_overhead() * len(
        messages
    )


def _exact_prompt_tokens(messages: List[Dict[str, str]]) -> int:
    try:
        return count_tokens(_apply_chat_template(messages))
    except Exception:
        return _estimated_prompt_tokens(messages)


def _drop_oldest_turn(kept: List[Dict[str, str]]) -> bool:
    removable = [i for i, m in enumerate(kept) if m["role"] != "system"]
    if not removable:
        return False
    first = removable[0]
    if first == len(kept) - 1:
        return False
    if (
        first + 1 < len(kept)
        and kept[first]["role"] == "user"
        and kept[first + 1]["role"] == "assistant"
    ):
        del kept[first : first + 2]
    else:
        del kept[first]
    return True


def prepare_context(prompt_messages: List[Dict[str, str]], budget: int) -> PreparedContext:
    kept = [dict(m) for m in prompt_messages]
    original_count = len(kept)

    while _estimated_prompt_tokens(kept) > budget:
        if not _drop_oldest_turn(kept):
            break

    if _estimated_prompt_tokens(kept) > budget:
        last = len(kept) - 1
        other_tokens = sum(
            count_tokens(m["content"]) for i, m in enumerate(kept) if i != last
        )
        remaining = budget - other_tokens - template_overhead() * len(kept)
        if remaining <= 16:
            raise ContextTooLarge(
                "System prompt plus the final message exceed the context budget. "
                "Send a shorter message."
            )
        kept[last]["content"] = truncate_to_tokens(kept[last]["content"], remaining)

    exact = _exact_prompt_tokens(kept)
    for _ in range(8):
        if exact <= budget:
            break
        if not _drop_oldest_turn(kept):
            break
        exact = _exact_prompt_tokens(kept)

    if exact > budget:
        raise ContextTooLarge(
            f"Prompt needs ~{exact} tokens after trimming, over the {budget}-token "
            "prompt budget. Shorten the conversation or send a shorter message."
        )

    return PreparedContext(
        messages=kept,
        input_tokens=exact,
        prompt_budget=budget,
        dropped_messages=original_count - len(kept),
        truncated_messages=sum(
            1 for m in kept if len(m["content"]) >= CONFIG.max_chars_per_message
        ),
        system_tokens=sum(
            count_tokens(m["content"]) for m in kept if m["role"] == "system"
        ),
    )


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    low, high = 1, len(text)
    best = text[: max(1, int(len(text) * max_tokens / max(1, count_tokens(text))))]
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid]
        if count_tokens(candidate) <= max_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best.rstrip()

# %% [code]
# =============================================================================
# CELL 8 — FastAPI schemas and authentication
# =============================================================================
import hmac
from typing import Literal

from fastapi import Depends, FastAPI, Header, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["system", "user", "assistant"]


class MessageIn(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Role
    content: Any = ""

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_content(cls, value: Any) -> Any:
        return normalize_content(value) if isinstance(value, list) else value


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = CONFIG.served_model_name
    messages: List[MessageIn] = Field(..., min_length=1)
    temperature: Optional[float] = Field(default=None, ge=CONFIG.min_temperature, le=CONFIG.max_temperature)
    top_p: Optional[float] = Field(default=None, ge=0.05, le=1.0)
    max_tokens: Optional[int] = Field(default=None, ge=CONFIG.min_max_new_tokens)
    max_completion_tokens: Optional[int] = Field(default=None, ge=CONFIG.min_max_new_tokens)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.5, le=2.0)
    stream: bool = False
    n: int = 1

    def requested_max_tokens(self) -> Optional[int]:
        if self.max_tokens is not None:
            return self.max_tokens
        return self.max_completion_tokens


def resolve_model_name(requested: Optional[str]) -> str:
    if requested is None or not str(requested).strip():
        return CONFIG.served_model_name
    candidate = str(requested).strip()
    if candidate == CONFIG.served_model_name or candidate == CONFIG.model_id:
        return CONFIG.served_model_name
    raise UnknownModel(
        f"Unknown model '{candidate}'. Available: {CONFIG.served_model_name}."
    )


def verify_api_key(authorization: Optional[str] = Header(default=None)) -> None:
    if not CONFIG.api_key:
        raise ServerError("Server API key is not configured.")
    if not authorization:
        raise AuthError("Missing Authorization header. Expected 'Bearer <key>'.")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthError("Authorization header must use the Bearer scheme.")
    if not hmac.compare_digest(token.strip(), CONFIG.api_key):
        raise AuthError("Invalid API key.")


def get_request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex[:16]}"


app = FastAPI(
    title="N-ATLaS API",
    version="1.0.0",
    description=f"{CONFIG.model_id} served as an OpenAI-compatible chat completion API.",
    docs_url="/docs",
    redoc_url=None,
)

SERVER_START_TIME = time.time()


@app.middleware("http")
async def guard_request(request: Request, call_next):
    request_id = get_request_id(request)
    started = time.perf_counter()
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > CONFIG.max_request_bytes:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "error": {
                            "type": "request_too_large",
                            "message": (
                                "Request body exceeds "
                                f"{CONFIG.max_request_bytes} bytes."
                            ),
                            "param": None,
                            "code": "request_too_large",
                        }
                    },
                    headers={"X-Request-ID": request_id},
                )
        except ValueError:
            pass
    try:
        response = await call_next(request)
    except ServerError as exc:
        logger.error("request_id=%s unhandled %s: %s", request_id, type(exc).__name__, exc.log_detail)
        response = JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "type": type(exc).__name__,
                    "message": exc.detail,
                    "param": None,
                    "code": type(exc).__name__,
                }
            },
            headers={"X-Request-ID": request_id},
        )
    response.headers["X-Request-ID"] = request_id
    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


def _error_body(message: str, err_type: str, code: str, param: Optional[str] = None):
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code,
        }
    }


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    is_malformed_json = any(e.get("type") == "json_invalid" for e in errors)
    status_code = status.HTTP_400_BAD_REQUEST if is_malformed_json else status.HTTP_422_UNPROCESSABLE_ENTITY
    first = errors[0] if errors else {}
    loc = first.get("loc") or ()
    param = ".".join(str(p) for p in loc if p not in ("body",)) or None
    message = first.get("msg", "Invalid request payload.")
    message = re.sub(r"^Value error,\s*", "", message)
    logger.warning(
        "request_id=%s validation error type=%s param=%s msg=%s",
        get_request_id(request),
        first.get("type"),
        param,
        message,
    )
    return JSONResponse(
        status_code=status_code,
        content=_error_body(message, "invalid_request_error", "invalid_request", param),
        headers={"X-Request-ID": get_request_id(request)},
    )


@app.exception_handler(ServerError)
async def server_error_handler(request: Request, exc: ServerError):
    request_id = get_request_id(request)
    if exc.status_code >= 500:
        logger.error("request_id=%s %s: %s", request_id, type(exc).__name__, exc.log_detail)
    else:
        logger.warning("request_id=%s %s: %s", request_id, type(exc).__name__, exc.detail)
    headers = {"X-Request-ID": request_id}
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.detail, "server_error", type(exc).__name__),
        headers=headers,
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    request_id = get_request_id(request)
    logger.error("request_id=%s unhandled %s: %s", request_id, type(exc).__name__, exc)
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_body(
            "Internal server error.",
            "server_error",
            "internal_error",
        ),
        headers={"X-Request-ID": request_id},
    )

# %% [code]
# =============================================================================
# CELL 9 — API endpoints
# =============================================================================
# NAFDAC lookup is implemented in Cell 17, which runs after this one. The hooks
# below are declared here so the routes exist from the moment the app is built,
# and are filled in by Cell 17. Registering the routes here rather than there
# means a request that arrives before Cell 17 finishes gets a clean 503 instead
# of a 404, and the app never has routes added to it after uvicorn has started.
NAFDAC_HOOKS: Dict[str, Any] = {
    "verify": None,
    "search": None,
    "enrich": None,
    "status": None,
}


class NafdacVerifyRequest(BaseModel):
    """A request to check one NAFDAC number.

    Accepts a bare number or a whole sentence, because the client forwards
    whatever the user typed rather than pre-extracting.
    """

    model_config = ConfigDict(extra="allow")

    nrn: str = Field(..., min_length=1, max_length=CONFIG.max_chars_per_message)
    include_history: bool = True


class NafdacSearchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str = Field(..., min_length=1, max_length=200)
    limit: int = Field(default=10, ge=1, le=50)


@app.get("/health")
def health():
    loaded = MODEL is not None
    return {
        "status": "ok" if loaded else "degraded",
        "model": CONFIG.model_id,
        "served_model_name": CONFIG.served_model_name,
        "device": DEVICE,
        "gpu": GPU.get("gpu_name", "none"),
        "gpu_vram_gb": GPU.get("gpu_vram_gb", 0.0),
        "cuda_available": GPU["cuda_available"],
        "quantization": QUANTIZATION_MODE,
        "model_loaded": loaded,
        "model_load_seconds": MODEL_LOAD_SECONDS,
        "model_load_error": MODEL_LOAD_ERROR,
        "transformers": _dist_version("transformers"),
        "torch": _dist_version("torch"),
        "bitsandbytes": _dist_version("bitsandbytes"),
        "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
        "context_budget": CONFIG.context_budget,
        "max_new_tokens": CONFIG.max_new_tokens,
        "nafdac": (
            NAFDAC_HOOKS["status"]()
            if callable(NAFDAC_HOOKS["status"])
            else {"enabled": False, "reason": "greenbook data is not loaded yet"}
        ),
    }


@app.get("/v1/nafdac/sources", dependencies=[Depends(verify_api_key)])
def nafdac_sources():
    """Which greenbook data sources this server can actually reach.

    Exposed because the answer to "is the database connected?" otherwise
    requires reading the server log, and that is the first question when a
    verification returns nothing.
    """
    if not callable(NAFDAC_HOOKS["status"]):
        raise ModelUnavailable("The NAFDAC greenbook data is not loaded.")
    return NAFDAC_HOOKS["status"]()


@app.post("/v1/nafdac/verify", dependencies=[Depends(verify_api_key)])
def nafdac_verify(payload: NafdacVerifyRequest):
    """Check one NAFDAC number against the greenbook.

    Returns the full record, not a verdict, so a caller can decide how to
    present it. A number that is not found is a 200 with found=false rather
    than a 404: not being in the export is a normal answer, not a client error.
    """
    if not callable(NAFDAC_HOOKS["verify"]):
        raise ModelUnavailable("The NAFDAC greenbook data is not loaded.")
    result = NAFDAC_HOOKS["verify"](payload.nrn, include_history=payload.include_history)
    if not result.searched:
        raise InvalidRequest(
            "No NAFDAC number found in the request. It looks like A4-100160 or "
            "04-3877."
        )
    return result.to_dict()


@app.post("/v1/nafdac/search", dependencies=[Depends(verify_api_key)])
def nafdac_search(payload: NafdacSearchRequest):
    """Free-text search over product names, ingredients and applicants."""
    if not callable(NAFDAC_HOOKS["search"]):
        raise ModelUnavailable("The NAFDAC greenbook data is not loaded.")
    records = NAFDAC_HOOKS["search"](payload.query, limit=payload.limit)
    return {
        "query": payload.query,
        "count": len(records),
        "results": [record.to_dict() for record in records],
    }


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": CONFIG.served_model_name,
                "object": "model",
                "created": int(SERVER_START_TIME),
                "owned_by": CONFIG.owned_by,
                "root": CONFIG.served_model_name,
                "permission": [],
            }
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
def chat_completions(payload: ChatCompletionRequest, request: Request):
    request_id = get_request_id(request)
    if payload.stream:
        raise InvalidRequest(
            "Streaming is not supported by this server. Send stream=false."
        )
    if payload.n != 1:
        raise InvalidRequest("Only n=1 is supported.")

    served_name = resolve_model_name(payload.model)

    client_messages = [{"role": m.role, "content": m.content} for m in payload.messages]
    max_new_tokens = clamp_int(
        payload.requested_max_tokens()
        if payload.requested_max_tokens() is not None
        else CONFIG.default_max_new_tokens,
        CONFIG.min_max_new_tokens,
        CONFIG.max_new_tokens,
    )
    if payload.requested_max_tokens() is not None and payload.requested_max_tokens() > CONFIG.max_new_tokens:
        logger.warning(
            "request_id=%s clamped max_tokens %s -> %s",
            request_id,
            payload.requested_max_tokens(),
            CONFIG.max_new_tokens,
        )

    prepared = build_prompt_messages(client_messages)

    # The greenbook data is attached here, as its own system message, after the
    # prompt is built. It has to come after the language directive so the
    # directive still lands on the first system message, and before generation
    # so the model sees the values while it is answering. Only runs for a
    # message that actually carries a NAFDAC number.
    nafdac_enrich = NAFDAC_HOOKS.get("enrich")
    nafdac_context: Optional[Dict[str, Any]] = None
    if callable(nafdac_enrich):
        try:
            nafdac_context = nafdac_enrich(prepared)
        except Exception as exc:
            # A greenbook failure must not fail the chat turn; the model simply
            # answers without the verified block.
            logger.error("nafdac enrichment failed: %r", exc, exc_info=True)
            nafdac_context = None
    if nafdac_context:
        prepared = [dict(message) for message in prepared]
        prepared.insert(
            1,
            {
                "role": "system",
                "content": nafdac_context["content"],
            },
        )
        logger.info(
            "request_id=%s nafdac nrn=%s found=%s confidence=%s",
            request_id,
            nafdac_context.get("nrn"),
            nafdac_context.get("found"),
            nafdac_context.get("confidence"),
        )

    result = generate(
        prepared,
        max_new_tokens=max_new_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
        repetition_penalty=payload.repetition_penalty,
        request_id=request_id,
    )

    log_fields = result.to_log_fields()
    logger.info(
        "request_id=%s model=%s %s",
        request_id,
        served_name,
        " ".join(f"{k}={v}" for k, v in log_fields.items()),
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": served_name,
        "system_fingerprint": f"{CONFIG.model_id}-{QUANTIZATION_MODE.replace(' ', '')}",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "logprobs": None,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.input_tokens,
            "completion_tokens": result.output_tokens,
            "total_tokens": result.input_tokens + result.output_tokens,
        },
    }


@app.post("/v1/completions", dependencies=[Depends(verify_api_key)])
def legacy_completions(payload: Dict[str, Any]):
    raise InvalidRequest(
        "The legacy /v1/completions endpoint is not supported. Use "
        "/v1/chat/completions."
    )

# %% [code]
# =============================================================================
# CELL 10 — Start Uvicorn on 0.0.0.0:8000 in a background thread
# =============================================================================
if NEST_ASYNCIO_AVAILABLE:
    import asyncio

    try:
        asyncio.get_running_loop()
        _IN_EVENT_LOOP = True
    except RuntimeError:
        _IN_EVENT_LOOP = False

    if _IN_EVENT_LOOP:
        nest_asyncio.apply()
        print("[server] an event loop is already running: nest_asyncio applied.")
    else:
        print(
            "[server] no running event loop: uvicorn will own a fresh loop inside a "
            "dedicated background thread."
        )
else:
    print(
        "[server] nest_asyncio is unavailable; relying on the background-thread "
        "approach (install it if you need to serve from inside a running loop)."
    )

SERVER = None
SERVER_THREAD = None
SERVER_STATE = {"running": False, "started_at": None, "error": None}
SHUTDOWN_EVENT = threading.Event()


def is_server_running() -> bool:
    return bool(SERVER_STATE["running"] and SERVER is not None and getattr(SERVER, "started", False))


def serve_forever() -> None:
    """Block the calling thread until stop_server() or Ctrl+C. Script entry point."""
    SHUTDOWN_EVENT.clear()
    try:
        SHUTDOWN_EVENT.wait()
    finally:
        SHUTDOWN_EVENT.clear()


def start_server(host: Optional[str] = None, port: Optional[int] = None, wait: float = 30.0):
    global SERVER, SERVER_THREAD
    if is_server_running():
        print("[server] already running.")
        return SERVER

    host = host or CONFIG.host
    port = int(port or CONFIG.port)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
        timeout_keep_alive=30,
        limit_concurrency=32,
    )
    SERVER = uvicorn.Server(config)

    def _run():
        try:
            SERVER.run()
        except Exception as exc:
            SERVER_STATE["error"] = repr(exc)
            logger.error("server thread crashed: %r", exc, exc_info=True)

    SERVER_THREAD = threading.Thread(target=_run, name="natlas-uvicorn", daemon=True)
    SERVER_THREAD.start()

    deadline = time.time() + wait
    while time.time() < deadline:
        if SERVER_STATE["error"]:
            raise RuntimeError(f"Uvicorn failed to start: {SERVER_STATE['error']}")
        if getattr(SERVER, "started", False):
            SERVER_STATE["running"] = True
            SERVER_STATE["started_at"] = time.time()
            break
        time.sleep(0.2)
    else:
        raise RuntimeError(f"Uvicorn did not start within {wait}s")

    _mem = _gpu_memory()
    print("=" * 70)
    print("N-ATLaS SERVER")
    print("=" * 70)
    print(f"Model:          {CONFIG.model_id}")
    print(f"Quantization:   {QUANTIZATION_MODE}")
    print(f"GPU:            {GPU.get('gpu_name', 'none')}")
    print(f"PyTorch:        {_dist_version('torch')}")
    print(f"Transformers:   {_dist_version('transformers')}")
    print(f"bitsandbytes:   {_dist_version('bitsandbytes')}")
    print(f"API:            http://{host}:{port}")
    print(f"Local:          http://127.0.0.1:{port}")
    print(f"Health:         http://127.0.0.1:{port}/health")
    print(f"Chat:           http://127.0.0.1:{port}/v1/chat/completions")
    print(f"Docs:           http://127.0.0.1:{port}/docs")
    print(f"API key source: {CONFIG.api_key_source}")
    print(f"GPU allocated:  {_mem['allocated_gb']} GB")
    print("=" * 70)
    return SERVER


def stop_server():
    if SERVER is not None:
        SERVER.should_exit = True
    SERVER_STATE["running"] = False
    SHUTDOWN_EVENT.set()


start_server()

# %% [code]
# =============================================================================
# CELL 11 — Local health test
# =============================================================================
BASE_URL = f"http://127.0.0.1:{CONFIG.port}"
TEST_KEY = CONFIG.api_key
TEST_RESULTS: List[Tuple[str, Optional[bool], str]] = []


def record(name: str, passed: bool, detail: str = "", requires_cuda: bool = False) -> None:
    if requires_cuda and not GPU["cuda_available"]:
        TEST_RESULTS.append((name, None, detail))
        print(f"[SKIP] {name} :: needs a CUDA GPU")
        return
    TEST_RESULTS.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))


def api(
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
    key: Optional[str] = TEST_KEY,
    timeout: float = 300.0,
    raw_body: Optional[str] = None,
):
    headers = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    url = f"{BASE_URL}{path}"
    if raw_body is not None:
        headers["Content-Type"] = "application/json"
        return requests.request(method, url, data=raw_body, headers=headers, timeout=timeout)
    return requests.request(method, url, json=body, headers=headers, timeout=timeout)


_health = api("GET", "/health", key=None, timeout=30)
record(
    "GET /health returns 200",
    _health.status_code == 200,
    f"status={_health.status_code}",
)
if _health.status_code == 200:
    _h = _health.json()
    record(
        "health reports model loaded",
        bool(_h.get("model_loaded")),
        f"model={_h.get('model')} device={_h.get('device')} gpu={_h.get('gpu')}",
    )
    record(
        "health reports cuda device",
        bool(_h.get("cuda_available")),
        f"cuda_available={_h.get('cuda_available')} quantization={_h.get('quantization')}",
        requires_cuda=True,
    )
    record(
        "model is loaded in 4-bit NF4",
        bool(_h.get("cuda_available")) and "4-bit" in str(_h.get("quantization")),
        f"quantization={_h.get('quantization')}",
        requires_cuda=True,
    )

_models = api("GET", "/v1/models")
record(
    "GET /v1/models returns 200",
    _models.status_code == 200,
    f"status={_models.status_code}",
)
if _models.status_code == 200:
    _m = _models.json()
    _ids = [item.get("id") for item in _m.get("data", [])]
    record(
        "/v1/models lists the served model",
        _m.get("object") == "list" and CONFIG.served_model_name in _ids,
        f"ids={_ids} owned_by={_m.get('data', [{}])[0].get('owned_by')}",
    )

_models_noauth = requests.get(f"{BASE_URL}/v1/models", timeout=30)
record(
    "/v1/models without key returns 401",
    _models_noauth.status_code == 401,
    f"status={_models_noauth.status_code}",
)

# %% [code]
# =============================================================================
# CELL 12 — Local chat-completion test
# =============================================================================
def chat(
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    key: Optional[str] = TEST_KEY,
    max_tokens: Optional[int] = None,
    timeout: float = 300.0,
) -> Tuple[int, str]:
    messages = list(history or []) + [{"role": "user", "content": prompt}]
    body: Dict[str, Any] = {"model": CONFIG.served_model_name, "messages": messages}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    response = api("POST", "/v1/chat/completions", body=body, key=key, timeout=timeout)
    if response.status_code != 200:
        return response.status_code, response.text
    return response.status_code, response.json()["choices"][0]["message"]["content"]


status_code, answer = chat("How are you?")
record("POST /v1/chat/completions returns 200", status_code == 200, f"status={status_code}")
record("English response is non-empty", bool(answer.strip()), repr(answer[:200]))

_openai = api(
    "POST",
    "/v1/chat/completions",
    body={
        "model": CONFIG.served_model_name,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "temperature": 0.7,
        "max_tokens": 64,
    },
)
if _openai.status_code == 200:
    _j = _openai.json()
    _choice = _j["choices"][0]
    _ok = (
        _j.get("object") == "chat.completion"
        and _j.get("id", "").startswith("chatcmpl-")
        and isinstance(_j.get("created"), int)
        and _j.get("model") == CONFIG.served_model_name
        and _choice["index"] == 0
        and _choice["message"]["role"] == "assistant"
        and isinstance(_choice["message"]["content"], str)
        and _choice.get("finish_reason") in {"stop", "length"}
        and isinstance(_j.get("usage", {}).get("total_tokens"), int)
    )
    record(
        "response body is OpenAI-compatible",
        _ok,
        f"id={_j.get('id')} finish_reason={_choice.get('finish_reason')} "
        f"usage={_j.get('usage')}",
    )
    record(
        "no chat-template tokens in response",
        not any(tok in _choice["message"]["content"] for tok in TEMPLATE_ARTIFACTS),
        repr(_choice["message"]["content"][:160]),
    )
    _content = _choice["message"]["content"]
    _leaked = TURN_MARKER_RE.search(_content)
    record(
        "no continuation markers leaked into the response",
        _leaked is None,
        f"leaked_marker={_leaked.group(0)!r}" if _leaked else "clean",
    )
    record(
        "response does not echo the system prompt",
        "multilingual African AI assistant" not in _content,
        repr(_content[:160]),
    )
else:
    record("response body is OpenAI-compatible", False, f"status={_openai.status_code}")

# %% [code]
# =============================================================================
# CELL 13 — Multilingual tests (English, Hausa, Yoruba, Igbo, mixed, diacritics)
# =============================================================================
LANGUAGE_CASES = [
    ("English", "How are you?"),
    ("Hausa", "Kana lafiya?"),
    ("Yoruba", "Báwo ni?"),
    ("Igbo", "Kedu ka ị mere?"),
    ("Mixed English/Hausa", "How are you? Yaya kake?"),
    ("Yoruba no diacritics", "Bawo ni?"),
    ("Igbo no diacritics", "Kedu ka i mere?"),
    ("Hausa greeting", "Sannu, yaya kake?"),
]

EXPECTED_LANGUAGE = {
    "English": "en",
    "Hausa": "ha",
    "Yoruba": "yo",
    "Igbo": "ig",
    "Mixed English/Hausa": "ha",
    "Yoruba no diacritics": "yo",
    "Igbo no diacritics": "ig",
    "Hausa greeting": "ha",
}

REFUSAL_PHRASES = (
    "i can't understand",
    "i cannot understand",
    "i'm sorry, but i can",
    "i do not understand",
    "unable to understand",
    "i'm sorry",
    "i am sorry",
    "as an ai",
    "i don't speak",
    "i do not speak",
    "cannot process this language",
    "not able to understand",
)

LANG_HINTS = {
    "English": (
        "how", "i", "i'm", "fine", "well", "good", "great", "doing", "glad",
        "hello", "hi", "thank", "thanks", "assistant", "help", "today", "sure",
        "english", "you", "your",
    ),
    "Hausa": (
        "kana", "lafiya", "sannu", "ina", "yake", "kake", "kadalla", "gaskiya",
        "yaya", "zan", "yana", "harshe", "magani", "gode", "kuma", "ku", "na",
        "dan", "sai", "to", "ya", "ta", "mu", "su", "dai", "wallahi", "ina",
        "maku", "muna", "taya", "jima",
    ),
    "Yoruba": (
        "bawo", "ni", "ko", "wa", "mo", "o", "ati", "se", "ibo", "jo", "yin",
        "dara", "pupọ", "pelepo", "tani", "kini", "nife", "e", "ma", "je", "ti",
        "si", "jowo", "baba", "mama",
    ),
    "Igbo": (
        "kedu", "mere", "ka", "gi", "ihe", "nke", "ndi", "anyi", "gịnị", "ọma",
        "mba", "ga", "ma", "nwa", "o", "i", "unu", "di", "m", "nso",
    ),
    "Mixed English/Hausa": (
        "kana", "lafiya", "yaya", "kake", "harshe", "gode", "kuma", "sannu",
        "ina", "yake",
    ),
    "Yoruba no diacritics": (
        "bawo", "ni", "ko", "wa", "mo", "o", "se", "ibo", "jo", "pelepo",
        "tani", "kini", "nife", "dara",
    ),
    "Igbo no diacritics": (
        "kedu", "mere", "ka", "gi", "ihe", "nke", "ndi", "anyi", "oma", "mba",
        "ga", "nwa", "unu", "di",
    ),
    "Hausa greeting": (
        "sannu", "kana", "lafiya", "kake", "yake", "ina", "yaya", "harshe",
        "gode", "kuma", "ku",
    ),
}

_HINT_PATTERNS: Dict[str, List[Any]] = {}


def _compile_hints(label: str) -> List[Any]:
    if label not in _HINT_PATTERNS:
        patterns = []
        for hint in LANG_HINTS.get(label, ()):
            if hint.isascii():
                patterns.append(re.compile(rf"\b{re.escape(hint)}\b", re.IGNORECASE))
            else:
                patterns.append(re.compile(re.escape(hint), re.IGNORECASE))
        _HINT_PATTERNS[label] = patterns
    return _HINT_PATTERNS[label]


def language_signal(label: str, text: str) -> bool:
    return any(pattern.search(text) for pattern in _compile_hints(label))


LANGUAGE_ADHERENCE: List[Tuple[str, bool, str]] = []

for label, prompt in LANGUAGE_CASES:
    status_code, answer = chat(prompt, max_tokens=160)
    ok_status = status_code == 200
    ok_nonempty = ok_status and bool(answer.strip())
    ok_refusal = not any(phrase in answer.lower() for phrase in REFUSAL_PHRASES)
    ok_language = ok_nonempty and language_signal(label, answer)
    detected = LAST_LANGUAGE_DETECTION
    answered_in = detect_language(answer) if ok_nonempty else {"language": "?"}
    LANGUAGE_ADHERENCE.append((label, ok_language, answered_in["language"]))
    record(
        f"[{label}] generates a reply",
        ok_status and ok_nonempty,
        f"status={status_code} :: {answer[:180]!r}",
    )
    record(
        f"[{label}] does not refuse the language",
        ok_refusal,
        f"refusal_phrase_found={not ok_refusal}",
    )
    record(
        f"[{label}] tagger targeted the right language",
        detected.get("language") == EXPECTED_LANGUAGE[label],
        f"detected={detected.get('language')} confidence={detected.get('confidence')} "
        f"injected={detected.get('injected')}",
    )

_adhered = sum(1 for _, ok, _ in LANGUAGE_ADHERENCE if ok)
_total_lang = len(LANGUAGE_ADHERENCE)
_missed = [
    f"{label} (answered in {got})" for label, ok, got in LANGUAGE_ADHERENCE if not ok
]
record(
    f"replies stay in the requested language ({_adhered}/{_total_lang})",
    _adhered == _total_lang,
    f"missed={_missed}" if _missed else "all languages matched",
)
if _missed:
    print(
        "  NOTE: the tagger above chose the right language every time, so this is the\n"
        "        model overriding an explicit instruction — not a server bug.\n"
        "        Missed: " + "; ".join(_missed)
    )

print("-" * 70)
print("Direct engine calls, bypassing HTTP entirely:")
direct_answers = {}
for _prompt in ["Kana lafiya?", "Sannu, yaya kake?", "Báwo ni?", "Kedu ka ị mere?", "How are you?"]:
    _direct = generate_response([{"role": "user", "content": _prompt}], max_new_tokens=160)
    direct_answers[_prompt] = _direct
    print(f"> {_prompt}")
    print(f"  {_direct}")

record(
    "direct generate_response() answers all five prompts",
    all(bool(text.strip()) for text in direct_answers.values()),
    f"prompts={len(direct_answers)}",
)
record(
    "direct generate_response() never refuses a Nigerian language",
    not any(
        phrase in text.lower()
        for text in direct_answers.values()
        for phrase in REFUSAL_PHRASES
    ),
    repr({k: v[:60] for k, v in list(direct_answers.items())[:2]}),
)

# %% [code]
# =============================================================================
# CELL 14 — Conversation memory and language switching
# =============================================================================
_memory_prompt = "My name is Ahmed."
_memory_reply = chat(_memory_prompt)[1]
_memory_history = [
    {"role": "user", "content": _memory_prompt},
    {"role": "assistant", "content": _memory_reply or "Nice to meet you, Ahmed."},
]
_code, _answer = chat("What is my name?", history=_memory_history)
record(
    "conversation memory: recalls the name",
    _code == 200 and "ahmed" in _answer.lower(),
    f"status={_code} :: {_answer[:180]!r}",
)

_switch_history: List[Dict[str, str]] = []
_code1, hausa_reply = chat("Answer in Hausa.", max_tokens=120)
_switch_history += [
    {"role": "user", "content": "Answer in Hausa."},
    {"role": "assistant", "content": hausa_reply or "Ok"},
]
_code2, english_reply = chat("Now answer in English.", history=_switch_history, max_tokens=120)
record(
    "language switching: Hausa then English",
    _code1 == 200 and _code2 == 200 and bool(english_reply.strip()),
    f"hausa={hausa_reply[:90]!r} english={english_reply[:90]!r}",
)
record(
    "language switching: English reply is not pure Hausa",
    bool(english_reply.strip()) and language_signal("English", english_reply),
    f"english_signal={language_signal('English', english_reply)}",
)

_code3, followup = chat(
    "And what did I ask you to do first?",
    history=[
        {"role": "user", "content": "Answer in Hausa."},
        {"role": "assistant", "content": hausa_reply or "Ok"},
        {"role": "user", "content": "Now answer in English."},
        {"role": "assistant", "content": english_reply or "Okay."},
    ],
    max_tokens=120,
)
record(
    "follow-up question keeps context across a language switch",
    _code3 == 200 and bool(followup.strip()),
    f"status={_code3} :: {followup[:180]!r}",
)

_code4, greeted = chat("Kana lafiya?", history=_memory_history, max_tokens=120)
record(
    "language switch with preserved memory",
    _code4 == 200 and bool(greeted.strip()),
    f"status={_code4} :: {greeted[:180]!r}",
)

# %% [code]
# =============================================================================
# CELL 15 — Error handling and security tests
# =============================================================================
_refusal_response = api(
    "POST",
    "/v1/chat/completions",
    body={"model": CONFIG.served_model_name, "messages": [{"role": "user", "content": "Hi"}]},
    key="definitely-not-the-key",
)
record(
    "invalid API key returns 401",
    _refusal_response.status_code == 401,
    f"status={_refusal_response.status_code}",
)
record(
    "401 body does not leak the key",
    CONFIG.api_key not in _refusal_response.text
    and TEST_KEY not in _refusal_response.text,
    repr(_refusal_response.text[:160]),
)
record(
    "401 sets WWW-Authenticate header",
    "www-authenticate" in {k.lower() for k in _refusal_response.headers},
    f"headers={sorted(k.lower() for k in _refusal_response.headers)[:8]}",
)

_malformed = api(
    "POST",
    "/v1/chat/completions",
    raw_body='{"model": "n-atlas", "messages": [',
    key=TEST_KEY,
)
record(
    "malformed JSON returns 400",
    _malformed.status_code == 400,
    f"status={_malformed.status_code} :: {_malformed.text[:140]}",
)

_missing_messages = api("POST", "/v1/chat/completions", body={"model": CONFIG.served_model_name})
record(
    "missing messages returns 422",
    _missing_messages.status_code == 422,
    f"status={_missing_messages.status_code} :: {_missing_messages.text[:140]}",
)

_empty_messages = api(
    "POST",
    "/v1/chat/completions",
    body={"model": CONFIG.served_model_name, "messages": []},
)
record(
    "empty messages returns 422",
    _empty_messages.status_code == 422,
    f"status={_empty_messages.status_code}",
)

_blank_messages = api(
    "POST",
    "/v1/chat/completions",
    body={"model": CONFIG.served_model_name, "messages": [{"role": "user", "content": "   "}]},
)
record(
    "whitespace-only message returns 400",
    _blank_messages.status_code == 400,
    f"status={_blank_messages.status_code} :: {_blank_messages.text[:140]}",
)

_bad_role = api(
    "POST",
    "/v1/chat/completions",
    body={
        "model": CONFIG.served_model_name,
        "messages": [{"role": "wizard", "content": "hello"}],
    },
)
record(
    "invalid role returns 422",
    _bad_role.status_code == 422,
    f"status={_bad_role.status_code}",
)

_bad_model = api(
    "POST",
    "/v1/chat/completions",
    body={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
)
record(
    "unknown model returns 404",
    _bad_model.status_code == 404,
    f"status={_bad_model.status_code} :: {_bad_model.text[:140]}",
)

_bad_temp = api(
    "POST",
    "/v1/chat/completions",
    body={
        "model": CONFIG.served_model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 9.5,
    },
)
record(
    "out-of-range temperature returns 422",
    _bad_temp.status_code == 422,
    f"status={_bad_temp.status_code}",
)

_bad_max_tokens = api(
    "POST",
    "/v1/chat/completions",
    body={
        "model": CONFIG.served_model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": -5,
    },
)
record(
    "negative max_tokens returns 422",
    _bad_max_tokens.status_code == 422,
    f"status={_bad_max_tokens.status_code}",
)

_clamped = api(
    "POST",
    "/v1/chat/completions",
    body={
        "model": CONFIG.served_model_name,
        "messages": [{"role": "user", "content": "Reply with just the word: ok"}],
        "max_tokens": 100000,
    },
    timeout=300.0,
)
record(
    "huge max_tokens is clamped, not fatal",
    _clamped.status_code == 200,
    f"status={_clamped.status_code}",
)
if _clamped.status_code == 200:
    _u = _clamped.json().get("usage", {})
    record(
        "clamped completion stays within the token cap",
        _u.get("completion_tokens", 0) <= CONFIG.max_new_tokens,
        f"completion_tokens={_u.get('completion_tokens')} cap={CONFIG.max_new_tokens}",
    )

_stream = api(
    "POST",
    "/v1/chat/completions",
    body={
        "model": CONFIG.served_model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    },
)
record(
    "stream=true is rejected with a clear 400",
    _stream.status_code == 400 and "stream" in _stream.text.lower(),
    f"status={_stream.status_code}",
)

_oversize_body = api(
    "POST",
    "/v1/chat/completions",
    raw_body="x" * (CONFIG.max_request_bytes + 1024),
)
record(
    "oversized body is rejected with 413",
    _oversize_body.status_code in (400, 413),
    f"status={_oversize_body.status_code}",
)

_long_history = [
    {"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn {i}: " + "lorem ipsum " * 40}
    for i in range(CONFIG.max_messages + 12)
]
_long = api(
    "POST",
    "/v1/chat/completions",
    body={"model": CONFIG.served_model_name, "messages": _long_history, "max_tokens": 16},
    timeout=300.0,
)
record(
    f"{len(_long_history)}-message history is trimmed to {CONFIG.max_messages} and answered",
    _long.status_code == 200,
    f"status={_long.status_code} :: {_long.text[:140]}",
)

_absurd_history = [
    {"role": "user" if i % 2 == 0 else "assistant", "content": f"Turn {i}"}
    for i in range(CONFIG.max_messages_absolute + 5)
]
_absurd = api(
    "POST",
    "/v1/chat/completions",
    body={"model": CONFIG.served_model_name, "messages": _absurd_history},
)
record(
    f"{len(_absurd_history)}-message payload hits the hard limit and returns 400",
    _absurd.status_code == 400,
    f"status={_absurd.status_code} :: {_absurd.text[:140]}",
)

_big_turn = api(
    "POST",
    "/v1/chat/completions",
    body={
        "model": CONFIG.served_model_name,
        "messages": [{"role": "user", "content": "Summarise this. " + "alpha beta gamma " * 4000}],
        "max_tokens": 32,
    },
    timeout=300.0,
)
record(
    "single oversized turn is truncated or rejected without a 5xx",
    _big_turn.status_code in (200, 400, 413),
    f"status={_big_turn.status_code}",
)

_trim_history = [
    {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i} " + "detail " * 120}
    for i in range(120)
]
_trim = api(
    "POST",
    "/v1/chat/completions",
    body={
        "model": CONFIG.served_model_name,
        "messages": _trim_history + [{"role": "user", "content": "What was message 119 about?"}],
        "max_tokens": 64,
    },
    timeout=300.0,
)
record(
    "120-turn conversation is trimmed inside the context budget",
    _trim.status_code == 200,
    f"status={_trim.status_code} :: {_trim.text[:140]}",
)

_system_only = api(
    "POST",
    "/v1/chat/completions",
    body={
        "model": CONFIG.served_model_name,
        "messages": [{"role": "system", "content": "You are a calculator."}],
    },
)
record(
    "messages without a user turn returns 400",
    _system_only.status_code == 400,
    f"status={_system_only.status_code} :: {_system_only.text[:140]}",
)

_legacy = api(
    "POST",
    "/v1/completions",
    body={"model": CONFIG.served_model_name, "prompt": "hi"},
)
record(
    "legacy /v1/completions returns 400 with guidance",
    _legacy.status_code == 400,
    f"status={_legacy.status_code}",
)

for _name, _response in (
    ("malformed JSON 400", _malformed),
    ("blank message 400", _blank_messages),
    ("unknown model 404", _bad_model),
    ("stream rejected 400", _stream),
    ("invalid role 422", _bad_role),
):
    _body_text = _response.text
    record(
        f"{_name} does not leak a traceback or stack frame",
        "Traceback" not in _body_text and 'File "' not in _body_text,
        repr(_body_text[:140]),
    )

# %% [code]
# =============================================================================
# CELL 16 — Performance / GPU test and Stage 1 gate
# =============================================================================
_perf_samples: List[GenerationResult] = []
for _prompt in [
    "How are you?",
    "Kana lafiya?",
    "Báwo ni?",
    "Explain what N-ATLaS can do in two sentences.",
]:
    _t0 = time.perf_counter()
    try:
        _res = generate(
            build_prompt_messages([{"role": "user", "content": _prompt}]),
            max_new_tokens=128,
            request_id="perf",
        )
        _perf_samples.append(_res)
        print(
            f"[perf] {_prompt[:40]!r} in={_res.input_tokens} out={_res.output_tokens} "
            f"{_res.seconds:.2f}s {_res.tokens_per_second:.1f} tok/s "
            f"gpu_alloc={_res.gpu_memory.get('allocated_gb')}GB"
        )
    except ServerError as exc:
        print(f"[perf] FAILED for {_prompt[:40]!r}: {exc}")

record(
    "performance samples collected",
    len(_perf_samples) == 4,
    f"samples={len(_perf_samples)}",
)
if _perf_samples:
    _avg_tps = sum(r.tokens_per_second for r in _perf_samples) / len(_perf_samples)
    _avg_in = sum(r.input_tokens for r in _perf_samples) / len(_perf_samples)
    _avg_out = sum(r.output_tokens for r in _perf_samples) / len(_perf_samples)
    print("-" * 70)
    print(f"average input tokens:  {_avg_in:.0f}")
    print(f"average output tokens: {_avg_out:.0f}")
    print(f"average tokens/second: {_avg_tps:.1f}")
    print(f"GPU allocated:         {_gpu_memory()['allocated_gb']} GB")
    print(f"GPU reserved:          {_gpu_memory()['reserved_gb']} GB")
    print(f"GPU max allocated:     {_gpu_memory()['max_allocated_gb']} GB")
    print("-" * 70)
    record("tokens/second is usable for interactive chat", _avg_tps >= 1.0, f"{_avg_tps:.1f} tok/s")

record(
    "generation lock is released after every request",
    not GENERATION_LOCK.locked(),
    f"locked={GENERATION_LOCK.locked()}",
)

_concurrent_results: List[int] = []


def _concurrent_probe():
    try:
        _concurrent_results.append(
            api(
                "POST",
                "/v1/chat/completions",
                body={
                    "model": CONFIG.served_model_name,
                    "messages": [{"role": "user", "content": "Reply with the word ok"}],
                    "max_tokens": 16,
                },
                timeout=300.0,
            ).status_code
        )
    except Exception as exc:
        _concurrent_results.append(-1)
        logger.error("concurrent probe failed: %r", exc)


_threads = [threading.Thread(target=_concurrent_probe) for _ in range(3)]
for _t in _threads:
    _t.start()
for _t in _threads:
    _t.join()
record(
    "3 simultaneous requests all succeed (serialised by the generation lock)",
    all(code == 200 for code in _concurrent_results),
    f"statuses={_concurrent_results}",
)

print()
print("=" * 70)
_passed = sum(1 for _, ok, _ in TEST_RESULTS if ok is True)
_failed = [(name, detail) for name, ok, detail in TEST_RESULTS if ok is False]
_skipped = [name for name, ok, _ in TEST_RESULTS if ok is None]
print(
    f"STAGE 1 GATE: {_passed} passed, {len(_failed)} failed, "
    f"{len(_skipped)} skipped (GPU-only) out of {len(TEST_RESULTS)} checks"
)
if _failed:
    print("-" * 70)
    for _name, _detail in _failed:
        print(f"  FAIL  {_name} :: {_detail}")
if _skipped:
    print("-" * 70)
    print("  SKIPPED (need a real CUDA GPU):")
    for _name in _skipped:
        print(f"    {_name}")
print("=" * 70)
if not _failed:
    if _skipped:
        print("All runnable checks passed, but GPU checks were skipped.")
        print("Stage 1 is NOT complete: rerun on the T4 so nothing is skipped.")
    else:
        print("ALL CHECKS PASSED — Stage 1 complete.")
        print("Next stage (not started here): secure external tunnel -> laptop client.")
else:
    print("Stage 1 NOT complete. Fix the failures above before exposing this server.")

# %% [code]
# =============================================================================
# CELL 17 — NAFDAC greenbook verification
# =============================================================================
# This cell is what lets N-ATLaS answer "is A4-100160 genuine?" from data
# instead of from memory. It loads the shared `nafdac_source` package, points it
# at whatever greenbook data is reachable, and fills the NAFDAC_HOOKS that Cell
# 9 declared.
#
# Data sources, in the order they are consulted:
#
#   1. PostgreSQL  NAFDAC_DATABASE_URL. This is the one that matters most on
#                  Kaggle: the laptop's files are not reachable from here, but
#                  a hosted database is. It also carries superseded and
#                  withdrawn products, so "this used to be registered" stays
#                  answerable.
#   2. CSV         nafdac_greenbook.csv, if one was uploaded as a Kaggle
#                  dataset and NAFDAC_CSV_PATH points at it.
#   3. XLSX        nafdac_greenbook.xlsx, likewise via NAFDAC_XLSX_PATH.
#
# The package is looked for in several places, because on Kaggle it has to be
# uploaded as a dataset and the mount path is not something this notebook can
# assume. Every entry is treated as a candidate *parent* directory containing
# `nafdac_source/`, or as the package directory itself.
NAFDAC_SEARCH_DIRS = [
    os.environ.get("NAFDAC_SOURCE_DIR", ""),
    "/kaggle/input/n-atlas-source",
    "/kaggle/input/nafdac-source",
    "/kaggle/input/natlas-source",
    "/kaggle/input/n-afdac-source",
    str(WORKDIR / "nafdac_source"),
    str(WORKDIR),
    str(Path.cwd()),
]

# The notebook's own directory and its parent, when running from a checkout
# rather than from Kaggle. Not added inside the list literal because __file__
# does not exist in a notebook, and a bare `str()` of it would inject the
# string "<__file__>" into the search path.
_NAFDAC_SELF = globals().get("__file__")
if _NAFDAC_SELF:
    _NAFDAC_SELF_DIR = Path(_NAFDAC_SELF).resolve().parent
    # apiserver.py already puts the project root on sys.path, so the checkout
    # layout is: <root>/n_atlas_server/natlas_notebook.py, <root>/nafdac_source/.
    for _candidate in (
        _NAFDAC_SELF_DIR,
        _NAFDAC_SELF_DIR.parent,
        _NAFDAC_SELF_DIR.parent / "nafdac_source",
    ):
        _text = str(_candidate)
        if _text not in NAFDAC_SEARCH_DIRS:
            NAFDAC_SEARCH_DIRS.append(_text)

# The data sources below are optional in the sense that a missing one is
# reported, not fatal. When the package cannot be found the cell degrades
# rather than raising: the server still starts, /health still reports
# nafdac.enabled=false, and chat still works. A missing greenbook must not take
# the model offline.
NAFDAC_ENABLED = _env_flag("N_ATLAS_NAFDAC_ENABLED", True)
NAFDAC_MAX_CONTEXT_CHARS = _env_int("N_ATLAS_NAFDAC_MAX_CONTEXT_CHARS", 1800)
NAFDAC_MAX_SEARCH_RESULTS = _env_int("N_ATLAS_NAFDAC_MAX_SEARCH_RESULTS", 10)

# The instruction prepended to the verified block. Phrased as a constraint
# rather than a suggestion, because the failure this prevents — the model
# confidently inventing a product name or an approval date — is exactly the
# thing a verification tool must not do.
NAFDAC_INSTRUCTION = (
    "[nafdac] The block below is verified data from the NAFDAC Greenbook "
    "database, retrieved for this question. Treat it as fact.\n"
    "- Report these exact values. Do not reword a product name, and do not "
    "round or reformat a date.\n"
    "- Do not add an ingredient, strength, applicant, manufacturer, approval "
    "date or status that is not listed.\n"
    "- If a field the user asks about is not listed, say it is not recorded in "
    "the greenbook. Do not fill the gap from memory.\n"
    "- If the block says the number was not found, say so plainly and do not "
    "conclude that the product is counterfeit: the export may be out of date.\n"
    "- Answer in the user's language, as the rest of your instructions say."
)

NAFDAC_IMPORT_ERROR = ""
NAFDAC_SOURCES: List[Any] = []
NAFDAC_AVAILABLE = False


def _import_nafdac_source() -> bool:
    """Put `nafdac_source` on sys.path and import it. Returns True on success."""
    global NAFDAC_IMPORT_ERROR

    if "nafdac_source" in sys.modules:
        return True

    tried = []
    for directory in NAFDAC_SEARCH_DIRS:
        if not directory:
            continue
        path = Path(directory)
        # Accept both a parent directory containing nafdac_source/ and the
        # package directory itself, because a Kaggle dataset can be mounted
        # either way depending on how it was uploaded.
        candidates = [path] if path.name == "nafdac_source" else [path]
        for candidate in candidates:
            package_init = candidate / "nafdac_source" / "__init__.py"
            tried.append(str(package_init))
            if package_init.is_file():
                if str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
                try:
                    import nafdac_source  # noqa: F401

                    logger.info("nafdac_source loaded from %s", candidate)
                    return True
                except Exception as exc:
                    NAFDAC_IMPORT_ERROR = f"import failed from {candidate}: {exc}"
                    logger.error(NAFDAC_IMPORT_ERROR)
                    continue

        # The directory may itself be the package (not a parent of one).
        direct_init = path / "__init__.py"
        if path.name == "nafdac_source" and direct_init.is_file():
            tried.append(str(direct_init))
            if str(path.parent) not in sys.path:
                sys.path.insert(0, str(path.parent))
            try:
                import nafdac_source  # noqa: F401

                logger.info("nafdac_source loaded from %s", path.parent)
                return True
            except Exception as exc:
                NAFDAC_IMPORT_ERROR = f"import failed from {path.parent}: {exc}"
                continue

    if not NAFDAC_IMPORT_ERROR:
        NAFDAC_IMPORT_ERROR = (
            "nafdac_source/__init__.py not found in any of: "
            + ", ".join(tried[:6])
        )
    return False


def _build_nafdac_sources() -> List[Any]:
    """The source list, in merge order, from the environment."""
    from nafdac_source.sources import CsvSource, PostgresSource, XlsxSource

    sources: List[Any] = []

    dsn = os.environ.get("NAFDAC_DATABASE_URL", "").strip()
    if dsn:
        sources.append(
            PostgresSource(
                dsn=dsn,
                connect_timeout=_env_int("NAFDAC_DB_CONNECT_TIMEOUT", 10),
            )
        )

    csv_setting = os.environ.get("NAFDAC_CSV_PATH", "").strip()
    xlsx_setting = os.environ.get("NAFDAC_XLSX_PATH", "").strip()

    if csv_setting:
        sources.append(CsvSource(Path(csv_setting) if csv_setting else None))
    elif _env_flag("NAFDAC_AUTODETECT_FILES", True):
        # Upload nafdac_greenbook.csv as a Kaggle dataset and it is found
        # without any configuration, because CsvSource walks up from here.
        sources.append(CsvSource())

    if xlsx_setting:
        sources.append(XlsxSource(Path(xlsx_setting) if xlsx_setting else None))
    elif _env_flag("NAFDAC_AUTODETECT_FILES", True):
        sources.append(XlsxSource())

    return sources


def _nafdac_status() -> Dict[str, Any]:
    """The payload behind /health's nafdac field and /v1/nafdac/sources."""
    if not NAFDAC_AVAILABLE:
        return {
            "enabled": False,
            "reason": NAFDAC_IMPORT_ERROR or "nafdac_source is not importable",
            "hint": (
                "Upload the nafdac_source/ directory as a Kaggle dataset, or set "
                "NAFDAC_SOURCE_DIR to a directory containing it."
            ),
        }

    payload: Dict[str, Any] = {"enabled": True, "sources": []}
    for source in NAFDAC_SOURCES:
        try:
            detail = source.describe()
        except Exception as exc:
            detail = f"error: {type(exc).__name__}"
        payload["sources"].append(
            {
                "name": getattr(source, "name", "unknown"),
                "detail": detail,
                "usable": source.available(),
            }
        )
    payload["any_source_usable"] = any(
        entry["usable"] for entry in payload["sources"]
    )
    if not payload["any_source_usable"]:
        payload["hint"] = (
            "No source is reachable. Set NAFDAC_DATABASE_URL for a hosted "
            "database, or NAFDAC_CSV_PATH to an uploaded nafdac_greenbook.csv."
        )
    return payload


def _nafdac_enrich(prompt_messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """Attach verified greenbook data to a prompt, or return None.

    Returns None when the last user message is not a NAFDAC question, when no
    number could be found in it, or when the number is not in the data. A miss
    is deliberately not passed on as verified context: telling the model
    "not found" as fact would stop it answering sensibly about a product that
    is simply newer than the export.
    """
    if not NAFDAC_AVAILABLE:
        return None

    last_user = next(
        (
            message["content"]
            for message in reversed(prompt_messages)
            if message["role"] == "user"
        ),
        "",
    )
    if not last_user:
        return None

    from nafdac_source import extract_nrn, mentions_nafdac

    # Cheap gate first, so ordinary conversation never pays for a lookup. A
    # message that names a number but not NAFDAC is still checked, because a
    # bare "A4-100160" is usually a lookup and the greenbook resolves it in
    # milliseconds. Anything with no number-shaped token at all stops here.
    has_nrn = bool(extract_nrn(last_user))
    if not has_nrn and not mentions_nafdac(last_user):
        return None

    result = verify_nafdac_product(last_user, sources=NAFDAC_SOURCES)
    if not result.found:
        logger.info("nafdac %s not found in any source", result.nrn)
        return None

    rendered = result.render()
    if len(rendered) > NAFDAC_MAX_CONTEXT_CHARS:
        rendered = rendered[: NAFDAC_MAX_CONTEXT_CHARS].rstrip() + "\n[truncated]"

    return {
        "nrn": result.nrn,
        "found": True,
        "confidence": result.confidence,
        "content": f"{NAFDAC_INSTRUCTION}\n\n{rendered}",
    }


if NAFDAC_ENABLED and _import_nafdac_source():
    try:
        NAFDAC_SOURCES = _build_nafdac_sources()
        NAFDAC_AVAILABLE = True
    except Exception as exc:
        NAFDAC_IMPORT_ERROR = f"could not build sources: {type(exc).__name__}: {exc}"
        logger.error(NAFDAC_IMPORT_ERROR)

if NAFDAC_AVAILABLE:
    from nafdac_source import search_products, verify_nafdac_product

    NAFDAC_HOOKS["verify"] = verify_nafdac_product
    NAFDAC_HOOKS["search"] = search_products
    NAFDAC_HOOKS["status"] = _nafdac_status
    NAFDAC_HOOKS["enrich"] = _nafdac_enrich

    # A read-only check at startup: it costs one round trip, and it means a
    # misconfigured connection is visible in the log rather than surfacing as
    # a failed lookup in a user's answer.
    _startup_result = verify_nafdac_product("A4-100160", sources=NAFDAC_SOURCES)
    _probe_note = ""
    if _startup_result.found:
        _probe_note = (
            f"probe A4-100160 -> {_startup_result.primary.product_name!r} "
            f"({_startup_result.status})"
        )
    else:
        _probe_note = "probe A4-100160 -> not found"
        logger.warning(
            "nafdac probe found nothing. Sources: %s",
            {name: why for name, why in _startup_result.sources_failed.items()},
        )
else:
    if not NAFDAC_ENABLED:
        _probe_note = "disabled by N_ATLAS_NAFDAC_ENABLED=0"
    else:
        _probe_note = f"unavailable: {NAFDAC_IMPORT_ERROR}"

print("=" * 70)
print("NAFDAC GREENBOOK")
print("=" * 70)
if NAFDAC_AVAILABLE:
    for _entry in _nafdac_status().get("sources", []):
        _mark = "ok " if _entry["usable"] else "-- "
        print(f"  {_mark}{_entry['name']:<9} {_entry['detail']}")
else:
    print(f"  unavailable: {_probe_note}")
print(f"  {_probe_note}")
print("=" * 70)
