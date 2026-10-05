# %% [markdown]
# # ConfamAI — QLoRA fine-tuning (Notebook A)
#
# This notebook trains the ConfamAI adapter and pushes it to Hugging Face. It is
# the *training* half of a two-notebook split, and it serves nothing:
#
# | | This notebook (A) | `n_atlas_server/natlas_kaggle.ipynb` (B) |
# |---|---|---|
# | Job | trains the adapter, pushes it, ends | serves chat requests |
# | Dataset | `confam-training` | `n-atlas-source` |
# | Lifetime | one run, ~2 h, then stop it | as long as you leave it running |
#
# Your laptop client only ever talks to notebook B. B picks up the adapter by
# setting `N_ATLAS_ADAPTER_ID`, so nothing else has to change.
#
# ## Before you run
#
# 1. Attach the Kaggle dataset `sabuwa/confam-training`, which carries
#    `confam_training_dataset.jsonl`. It mounts at
#    /kaggle/input/datasets/sabuwa/confam-training/, and Cell 1 finds it there
#    or anywhere else it lands.
# 2. Attach a Hugging Face token with **write** permission — Cell 8 pushes to
#    the Hub and cannot work without it. Either add it as a Kaggle secret named
#    `HF_TOKEN` (Add-ons -> Secrets -> New Secret), or set `HF_TOKEN` in the
#    environment before running. Do not paste it into a cell: a literal in this
#    file travels with the notebook when it is uploaded or shared.
# 3. Accelerator: **GPU T4 x2**. A T4 has no bfloat16 support, which is why Cell 5
#    pins the compute dtype to float16. The second T4 is deliberately left idle —
#    Cell 1 explains why training on both of them does not work.
# 4. Run every cell in order, starting from Cell 1. Cell 1 has to run before
#    anything imports torch, because it is what limits the process to one GPU.
#    Cell 9 tells you whether it worked before you touch notebook B.
#
# The dataset is checked in Cell 4, before the 5-minute model load, so a bad file
# fails in seconds rather than after the weights are in VRAM.

# %%
# =============================================================================
# CELL 1 — Configuration and input discovery
# =============================================================================
# Everything the run needs, resolved here so a wrong path fails immediately with
# an explanation instead of somewhere in the trainer.
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Pin the process to ONE GPU, here, before anything imports torch — Cell 2 is
# what imports it, so this is the last cell that can still do this cheaply.
#
# It is not about fitting the model: a 4-bit 8B plus LoRA ranks fits one T4 with
# room to spare. It is about the TRAINER. With more than one GPU visible in a
# plain-notebook launch, transformers wraps the model in torch.nn.DataParallel
# (Trainer._wrap_model, whenever args.n_gpu > 1), and Trainer.compute_loss adds
# `num_items_in_batch` — a 0-dim tensor — to the batch before calling the model.
# DataParallel.scatter cannot chunk a 0-dim tensor across replicas, so the FIRST
# training step dies with:
#
#     RuntimeError: chunk expects at least a 1-dimensional tensor
#
# (huggingface/transformers#37151, huggingface/trl#2338.)
#
# `device_map` in Cell 5 does not save this run: it puts the weights on cuda:0,
# but it is args.n_gpu — a count of VISIBLE devices, not of devices in use — that
# decides whether the model gets wrapped at all.
#
# The cost is the second T4 sitting idle. DataParallel was the only way to reach
# it from inside a notebook cell, and it is the thing that is broken; the
# supported multi-GPU path is `accelerate launch` (DDP), which a Kaggle cell
# cannot use. See time.md §3.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# A T4 has 16 GB and this run needs about 14.5 of it at batch 2, so the CUDA
# caching allocator spends the last gigabyte on fragmentation rather than on
# tensors. expandable_segments lets it grow and shrink one segment instead of
# carving up a fixed pool, which is what turns "1.02 GiB free" into a successful
# step when the arithmetic is nearly — but not quite — enough. setdefault, so a
# value already in the environment wins.
#
# Must be in place before the first allocation, which is another reason it lives
# here rather than next to the trainer in Cell 7.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = os.environ.get("N_ATLAS_MODEL_ID", "NCAIR1/N-ATLaS")

# Where this notebook PUSHES the trained adapter. Notebook B reads the same id
# from N_ATLAS_ADAPTER_ID. Both must match exactly, including the account name.
#
# "Saynetng" is the account slug from huggingface.co/Saynetng — note the
# lower-case "ng". It does not exist yet: the push in Cell 8 creates it. If your
# HF_TOKEN belongs to a different account, change this and the account on the
# token together, or the push will fail with a permission error after the
# training has run.
ADAPTER_ID = os.environ.get(
    "N_ATLAS_ADAPTER_ID", "Saynetng/confam-ai-adapter"
)

DATASET_FILENAME = "confam_training_dataset.jsonl"

# Where this dataset actually lives on Kaggle, for the attached dataset
# `sabuwa/confam-training`. Kaggle mounts an attached dataset either flat at
# /kaggle/input/<slug> or, when it is owner-scoped, at
# /kaggle/input/datasets/<owner>/<slug> — both layouts are in use, and this one
# is the second. The exact path is tried first so a run never depends on glob
# order, then both layouts are searched by filename for the case where the
# account or slug differs.
DATASET_MOUNT = (
    "/kaggle/input/datasets/sabuwa/confam-training/confam_training_dataset.jsonl"
)

# The system prompt length dominates this dataset: every record repeats it, and it
# is the bulk of the ~1100 tokens a record comes to. 2048 leaves ample room for the
# verified block and the answer; 1024 would truncate the assistant turn off the end,
# which is the only part that teaches anything. Measured in Cell 6, in tokens.
MAX_SEQ_LENGTH = int(os.environ.get("N_ATLAS_MAX_SEQ", "2048"))

MAX_STEPS = int(os.environ.get("N_ATLAS_TRAIN_STEPS", "200"))

# Per-device batch 1 with 8 accumulation steps, NOT 2 x 4, and the reason is
# arithmetic rather than taste: the last thing before the loss is the logits
# tensor, which is BATCH_SIZE x MAX_SEQ_LENGTH x vocab_size. Llama-3's vocab is
# 128256, so at 2048 padded positions one sequence's fp16 logits are 501 MB, and
# cross_entropy upcasts them to fp32 and keeps a softmax buffer alive for the
# backward pass — about 2 GB of VRAM per sequence in the batch, on top of the
# ~7 GB the 4-bit model and its fp32 embedding layer already hold.
#
# At batch 2 that is ~12 GB before the 1 GB final allocation, and a T4's usable
# 14.56 GB is not enough: the first step dies in ForCausalLMLoss with CUDA OOM.
# At batch 1 it fits, and because TRL pads to a fixed length rather than to the
# longest record in the batch, batch 1 also means no padding at all.
#
# The effective batch is 8 either way, so 200 steps still means 1600 samples and
# the learning-rate schedule is unchanged. The cost is wall clock: 8 sequential
# micro-steps per optimizer step instead of 4 larger ones.
BATCH_SIZE = int(os.environ.get("N_ATLAS_TRAIN_BATCH", "1"))
GRAD_ACCUM = int(os.environ.get("N_ATLAS_TRAIN_ACCUM", "8"))
LEARNING_RATE = float(os.environ.get("N_ATLAS_TRAIN_LR", "2e-4"))

OUTPUT_DIR = os.environ.get(
    "N_ATLAS_TRAIN_OUTPUT", "/kaggle/working/confam-ai-adapter"
)

LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

START_TIME = time.time()


def mounted_inputs():
    """Every dataset directory visible to this notebook, as full paths.

    Walks the owner-scoped layout too. `/kaggle/input/*` alone lists only
    "datasets" on such an account, which tells you nothing about which datasets
    are attached.
    """
    root = Path("/kaggle/input")
    if not root.is_dir():
        return []
    found = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        found.append(str(entry))
        if entry.name != "datasets":
            continue
        for owner in sorted(entry.iterdir()):
            if not owner.is_dir():
                continue
            for slug in sorted(owner.iterdir()):
                if slug.is_dir():
                    found.append(str(slug))
    return found


def find_dataset() -> Path:
    """Locate the JSONL wherever Kaggle mounted it.

    The exact mount path first, then both Kaggle layouts by filename, then the
    working directory so the cell is also usable outside Kaggle.
    """
    candidates = [Path(DATASET_MOUNT)]
    for base in ("/kaggle/input", "/kaggle/working", "."):
        candidates.extend(sorted(Path(base).glob(f"**/{DATASET_FILENAME}")))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    mounted = mounted_inputs()
    raise SystemExit(
        f"{DATASET_FILENAME} is not attached to this notebook.\n\n"
        "Attach the Kaggle dataset that contains it (Add Data), or point\n"
        "DATASET_MOUNT in Cell 1 at the path it was mounted to.\n"
        "Expected here:\n  %s\n"
        "Datasets visible to this notebook:\n  %s"
        % (DATASET_MOUNT,
           "\n  ".join(mounted) if mounted else "(nothing under /kaggle/input)")
    )


DATASET_PATH = find_dataset()

print("=" * 70)
print("CONFAMAI — QLoRA FINE-TUNING")
print("=" * 70)
print(f"base model:     {MODEL_ID}")
print(f"adapter out:    {ADAPTER_ID}")
print(f"gpus:            cuda:0 only "
      f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
print(f"dataset:        {DATASET_PATH} "
      f"({DATASET_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"max_seq_length: {MAX_SEQ_LENGTH}")
print(f"steps:          {MAX_STEPS}  "
      f"(effective batch {BATCH_SIZE * GRAD_ACCUM}, "
      f"~{MAX_STEPS * BATCH_SIZE * GRAD_ACCUM} samples seen)")
print(f"output:         {OUTPUT_DIR}")

# %%
# =============================================================================
# CELL 2 — Install libraries
# =============================================================================
# subprocess rather than a `!pip` magic: `!` is IPython-only, so this cell stays
# valid Python and the notebook can be compiled and validated offline.
#
# trl is pinned because Cell 7 uses the 0.12 argument names (`max_seq_length=`,
# `tokenizer=`). Later TRL renamed them to `max_length=` and `processing_class=`,
# and the old names are forwarded to TrainingArguments, which then fails with an
# error about an unknown argument that explains nothing.
TRL_VERSION = "0.12.2"

print("[deps] installing…")
proc = subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--quiet", "--no-input",
        "-U", "transformers", "datasets", "accelerate", "bitsandbytes",
        "peft", f"trl=={TRL_VERSION}",
    ],
    capture_output=True,
    text=True,
)
if proc.returncode != 0:
    raise RuntimeError(f"pip install failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
import importlib

importlib.invalidate_caches()

import torch  # noqa: E402
import transformers  # noqa: E402
import trl  # noqa: E402
import peft  # noqa: E402
import bitsandbytes  # noqa: E402

print(f"[deps] transformers {transformers.__version__}")
print(f"[deps] trl         {trl.__version__}")
print(f"[deps] peft        {peft.__version__}")
print(f"[deps] torch       {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "No GPU. Set the notebook accelerator to GPU T4 x2 and re-run."
    )

# Exactly one visible GPU, checked here because it is the difference between a
# run that works and one that dies on the first step. See Cell 1: with 2 visible
# devices the trainer wraps the model in torch.nn.DataParallel, and DataParallel
# cannot scatter the 0-dim num_items_in_batch tensor the loss divides by.
#
# torch caches the device count on first use, so if anything imported torch
# before Cell 1 ran, this is 2 however Cell 1 was written. Failing here costs
# seconds; the same mistake found in Cell 7 costs the 5-minute model load first.
_gpu_count = torch.cuda.device_count()
if _gpu_count != 1:
    raise SystemExit(
        "%d GPUs visible; this notebook trains on exactly 1. Cell 1 sets "
        "CUDA_VISIBLE_DEVICES=0 for that reason, but torch has already cached "
        "the device count.\n"
        "Restart the kernel and run every cell in order from Cell 1 — the "
        "notebook's first cell must be the one that hides cuda:1."
        % _gpu_count
    )

# %%
# =============================================================================
# CELL 3 — Hugging Face login
# =============================================================================
# Required, not optional: Cell 8 pushes the adapter, and an unauthenticated push
# fails after two hours of training. Checked here so the failure costs seconds.
#
# Three sources, tried in this order:
#   1. HF_TOKEN in the environment (Kaggle runtime settings, or `export` in a
#      shell). Useful when the Secrets panel is unavailable.
#   2. The HF_TOKEN Kaggle secret.
#   3. Nothing — the error below names all three ways out.
#
# The token is never written to disk and never printed. Nothing here belongs in
# the .ipynb either: a token pasted into a cell travels with the file when it is
# uploaded or shared, and a revoked token does not come back out of a diff.
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
_token_source = "environment"

if not HF_TOKEN:
    _token_source = "kaggle secret"
    try:
        from kaggle_secrets import UserSecretsClient

        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN").strip()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Could not get HF_TOKEN from the environment or Kaggle secrets "
            "(%s).\n"
            "Cell 8 pushes the adapter to the Hub, so a token is required.\n"
            "Either add it as a Kaggle secret (Add-ons -> Secrets -> New "
            "Secret, name HF_TOKEN),\n"
            "or set HF_TOKEN in the environment before running this notebook.\n"
            "Use a Hugging Face token with write permission; the username on "
            "it is the first\n"
            "half of ADAPTER_ID ('%s')."
            % (type(exc).__name__, ADAPTER_ID.split("/")[0])
        )

from huggingface_hub import login  # noqa: E402

login(token=HF_TOKEN)
print("[hf] logged in with the HF_TOKEN %s" % _token_source)

# %%
# =============================================================================
# CELL 4 — Load and check the dataset
# =============================================================================
# The invariants that make this dataset safe to train on, re-checked here on the
# file Kaggle actually has. They are the same ones prepare_training_data.py
# enforces at build time; re-checking at training time is what catches a stale
# upload, which is the realistic failure.
#
#   1. every record shares ONE system message (a mixed-persona file teaches the
#      model two identities and the system prompt can no longer correct it);
#   2. no record states a verdict without a verified block attached;
#   3. no record claims a product is counterfeit;
#   4. no record claims a pack is "safe".
#
# Check 3 is negation-aware on purpose. "This does not by itself mean the product
# is counterfeit" is required behaviour — the project forbids *claiming*
# counterfeit, and telling the user the opposite is the whole point of the
# NOT REGISTERED answer. A plain substring test rejects the sentence the assistant
# is supposed to say.
VERDICT_WORDS = ("AUTHENTIC", "FLAGGED", "NOT REGISTERED", "NOT_REGISTERED")
COUNTERFEIT_CLAIM = ("is counterfeit", "are counterfeit", "is poisoned",
                     "known counterfeit", "confirmed fake")

import re  # noqa: E402

_CLAIM_RE = re.compile(r"\b(?:is|are|was|were)\s+(?:counterfeit|poisoned)\b",
                       re.IGNORECASE)
_NEGATION_RE = re.compile(r"\b(?:not|never|no|n't|cannot|do not|does not)\b",
                          re.IGNORECASE)


def asserts_counterfeit(text):
    """True when the text CLAIMS a product is counterfeit.

    A negation earlier in the same sentence clears it: the rule is "never claim
    counterfeit", not "never write the word".
    """
    for sentence in re.split(r"(?<=[.!?])\s+", str(text or "")):
        match = _CLAIM_RE.search(sentence)
        if not match:
            continue
        if not _NEGATION_RE.search(sentence[:match.start()][-90:]):
            return True
    return False

records = [
    json.loads(line)
    for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

problems = []
system_messages = {
    record["messages"][0]["content"]
    for record in records
    if record["messages"] and record["messages"][0]["role"] == "system"
}
if len(system_messages) != 1:
    problems.append(
        "%d distinct system prompts; every record must carry the same one"
        % len(system_messages)
    )

verdicts = 0
with_block = 0
for number, record in enumerate(records, start=1):
    messages = record["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        problems.append("record %d does not end with an assistant turn" % number)
        continue
    blocks = [
        message["content"]
        for message in messages
        if message["role"] == "system" and message is not messages[0]
    ]
    has_block = any(block.startswith("[nafdac]") for block in blocks)
    with_block += has_block
    answer = messages[-1]["content"]
    lowered = answer.lower()
    if any(word in answer for word in VERDICT_WORDS):
        verdicts += 1
        if not has_block:
            problems.append(
                "record %d states a verdict with no [nafdac] block" % number
            )
    for phrase in ("known counterfeit", "confirmed fake", "safe for use"):
        if phrase in lowered:
            problems.append("record %d claims %r" % (number, phrase))
    if asserts_counterfeit(answer):
        problems.append("record %d asserts a product is counterfeit" % number)

prompt_text = next(iter(system_messages), "")
print("=" * 70)
print(f"records            : {len(records)}")
print(f"with verified block: {with_block}")
print(f"negatives (no block): {len(records) - with_block}")
print(f"verdict answers    : {verdicts}")
print(f"system prompt      : {len(prompt_text)} chars")
print(f"first line         : {prompt_text.splitlines()[0] if prompt_text else '(none)'}")
print("=" * 70)

if problems:
    print("INVALID DATASET — refusing to train on it:")
    for problem in problems[:20]:
        print("  - %s" % problem)
    raise SystemExit(
        "%d problem(s). Do not train on this file; rebuild it with "
        "prepare_training_data.py." % len(problems)
    )
print("VALID: one persona, every verdict backed by data, no counterfeit or "
      "safety claims.")

# %%
# =============================================================================
# CELL 5 — Load the base model in 4-bit NF4, wrapped in LoRA
# =============================================================================
# The 'Q' and the 'LoRA' of QLoRA: the base weights stay frozen in 4-bit and only
# small adapter matrices train, which is what makes an 8B model fit a T4.
#
# compute_dtype is float16, never bfloat16: the T4 is Turing-class and has no
# bf16 support. Notebook B loads the same model the same way, so training and
# serving agree on precision.
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

print(f"[model] loading {MODEL_ID} in 4-bit NF4…")
_load_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if not getattr(tokenizer, "chat_template", None):
    raise SystemExit(
        f"{MODEL_ID} exposes no chat template. Cell 6 renders records with it, "
        "and hand-building a prompt instead would train on a format the server "
        "never produces."
    )

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Pin the weights to ONE gpu as well.
#
# Cell 1 already hid cuda:1 from the process, so this is belt and braces — and it
# is the half that governs the WEIGHTS. device_map="auto" does not replicate the
# model, it SHARDS the layers across every visible GPU. With T4 x2 that puts the
# last layers — and therefore the logits — on cuda:1, while the per-batch token
# count the loss divides by is a scalar tensor living on cuda:0. transformers then
# dies in the loss with "Expected all tensors to be on the same device, but found
# at least two devices, cuda:1 and cuda:0". Pinning to a single device keeps the
# logits and the loss scalar together, which is the only arrangement this trainer
# supports.
#
# The trainer needs the same single-GPU arrangement for an unrelated reason, and
# Cell 1 is where that is handled: what killed the previous run was DataParallel,
# not sharding.
#
# The cost is throughput, not correctness: a 4-bit 8B model plus LoRA ranks fits
# one T4, and time.md §3 explains why the second T4 is unreachable from here.
_visible = torch.cuda.device_count()
if _visible > 1:
    print(f"[model] WARNING: {_visible} GPUs visible. Cell 1 sets "
          "CUDA_VISIBLE_DEVICES=0 for this reason, and without it the trainer "
          "wraps the model in DataParallel and the first step fails. The weights "
          "are still pinned to cuda:0 below.")
_device_map = {"": 0}

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map=_device_map,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

model = get_peft_model(
    model,
    LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    ),
)

print(f"[model] ready in {time.perf_counter() - _load_start:.0f}s")
print(f"[model] gpu: {next(model.parameters()).device} "
      f"({_visible} visible)")
model.print_trainable_parameters()
print("Expect a trainable percentage near 0.1%. If it is much larger, LoRA is "
      "not attached to the frozen base and the VRAM budget will not hold.")

# %%
# =============================================================================
# CELL 6 — Render the dataset with the model's own chat template
# =============================================================================
# Records are rendered with the template the server will use at inference, not a
# hand-rolled format: records carrying a verified block have two system messages
# in a row, which is exactly the shape notebook B builds at request time.
#
# The length measurement is not decoration. Every record repeats the same system
# prompt, so a too-small max_seq_length truncates the ASSISTANT turn off the end
# and silently trains on nothing. Measured here, before training, in tokens.
from datasets import Dataset  # noqa: E402


def render(example):
    """One record as text, keyed so datasets.map can write it back.

    The dict wrapper is required, not stylistic: map() only accepts a dict, a
    pyarrow table or None, and returning the bare string this used to return
    raises TypeError on the first record. The key has to be "text" because that
    is what SFTConfig(dataset_text_field="text") reads in Cell 7, and what the
    length measurement below indexes.
    """
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
    }


print("[data] rendering 643 records with the model's chat template…")
render_start = time.perf_counter()
train_dataset = Dataset.from_list(records).map(
    render, remove_columns=["messages"], desc="rendering"
)

# Measure in TOKENS, not characters. train_dataset["text"] is a list of strings,
# so len() on it returns a character count — which is roughly 4x the token count
# and made every one of the 643 records look like it overflowed. The rendered
# text already carries the template's own BOS/EOS markers as literal special
# tokens, so add_special_tokens=False avoids counting a second BOS on top.
lengths = [
    len(tokenizer(text, add_special_tokens=False)["input_ids"])
    for text in train_dataset["text"]
]
lengths.sort()
MEASURED_MAX_TOKENS = lengths[-1]
print(f"[data] rendered in {time.perf_counter() - render_start:.0f}s")
print(f"[data] tokens  p50={lengths[len(lengths) // 2]}  "
      f"p90={lengths[int(len(lengths) * 0.9)]}  max={lengths[-1]}")
print(f"[data] max_seq_length={MAX_SEQ_LENGTH}, "
      f"{sum(1 for n in lengths if n > MAX_SEQ_LENGTH)} records over it, "
      f"every batch padded to {MAX_SEQ_LENGTH}")

if MEASURED_MAX_TOKENS > MAX_SEQ_LENGTH:
    raise SystemExit(
        "%d tokens at the longest, over max_seq_length=%d. Truncation would cut "
        "the assistant turn — the only part of a record that teaches anything. "
        "Raise N_ATLAS_MAX_SEQ or shorten confam_prompt.txt."
        % (MEASURED_MAX_TOKENS, MAX_SEQ_LENGTH)
    )

# %%
# =============================================================================
# CELL 7 — Train
# =============================================================================
# Effective batch is BATCH_SIZE x GRAD_ACCUM = 8, so the default 200 steps cover
# all 643 records about 2.5 times. Two or three passes is right for LoRA on a
# dataset this size; much more and it memorises the synthetic product names.
from trl import SFTTrainer  # noqa: E402
from transformers import TrainingArguments  # noqa: E402

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=1,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    fp16=True,
    gradient_checkpointing=True,
    # use_reentrant=False is the mode PEFT documents for adapters: the reentrant
    # one silently drops adapter gradients in some PEFT versions and recomputes
    # with a peak that is harder to reason about. Cell 5 already enabled
    # checkpointing on the base model; this is the same switch, set where the
    # trainer actually uses it.
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    seed=42,
    run_name="confamai-qlora",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    packing=False,
    args=training_args,
)

# The one value that decides whether Trainer wraps the model in
# torch.nn.DataParallel. With n_gpu > 1 the first step dies inside
# DataParallel.scatter, which cannot chunk the 0-dim num_items_in_batch tensor
# that compute_loss adds to the batch — "chunk expects at least a 1-dimensional
# tensor". Cell 1 hides cuda:1 so this is 1; checking it here names the failure
# in one line instead of leaving a stack trace to read. See Cell 1.
if trainer.args.n_gpu != 1:
    raise SystemExit(
        "training args report n_gpu=%d. This notebook trains on one GPU, and "
        "more than one visible device breaks the first step in DataParallel. "
        "Run Cell 1 first, so CUDA_VISIBLE_DEVICES=0 reaches the process."
        % trainer.args.n_gpu
    )

# Print the two numbers that decide whether the loss fits, because the failure
# mode is an opaque CUDA OOM inside cross_entropy rather than anything the log
# explains. The logits are the peak: fp16 from the lm_head, then an fp32
# upcast plus a softmax buffer kept for the backward pass, which is the
# "Tried to allocate 1.04 GiB" in the Cell 7 failure at batch 2.
_logits_f16 = BATCH_SIZE * MAX_SEQ_LENGTH * model.config.vocab_size * 2
_total = torch.cuda.get_device_properties(0).total_memory
_loaded = torch.cuda.memory_allocated(0) / 1024**3
print(f"[train] {len(train_dataset)} records, {MAX_STEPS} steps, "
      f"effective batch {BATCH_SIZE * GRAD_ACCUM} "
      f"({BATCH_SIZE} per step x {GRAD_ACCUM} accum)")
print(f"[train] weights resident {_loaded:.1f} GiB of {_total / 1024**3:.1f} GiB; "
      f"logits {_logits_f16 / 1024**2:.0f} MiB per sequence at "
      f"{MAX_SEQ_LENGTH} padded positions, ~3x that in the loss")
print("If this run OOMs, N_ATLAS_TRAIN_BATCH=1 is already the floor — lower "
      "N_ATLAS_MAX_SEQ to Cell 6's measured max next, which is a linear saving.")
trainer.train()

print("=" * 70)
print(f"[train] done in {(time.time() - START_TIME) / 60:.0f} minutes")
print(f"[train] checkpoints in {OUTPUT_DIR}")

# %%
# =============================================================================
# CELL 8 — Push the adapter to the Hub
# =============================================================================
# This is the step that creates N_ATLAS_ADAPTER_ID for notebook B. Only the
# adapter is pushed, not the base weights: the repo should be tens of megabytes.
from huggingface_hub import HfApi  # noqa: E402

trainer.model.push_to_hub(ADAPTER_ID, token=HF_TOKEN)

info = HfApi().model_info(ADAPTER_ID)
size_mb = sum(
    (s.size or 0) for s in (info.siblings or [])
) / 1024 / 1024
files = sorted(s.rfilename for s in (info.siblings or []))

print("=" * 70)
print(f"[push] {ADAPTER_ID}")
print(f"[push] {len(files)} files, {size_mb:.1f} MB")
for name in files[:12]:
    print(f"       {name}")
print("=" * 70)
if size_mb > 500:
    print(
        "WARNING: that is far too large for a LoRA adapter. If the repo holds "
        "the base weights rather than adapter_model.safetensors, do not set "
        "N_ATLAS_ADAPTER_ID — fix the push first."
    )

print("NEXT: in notebook B, set N_ATLAS_ADAPTER_ID = %r (Kaggle secret or "
      "Cell 1), then run Cells 1, 2, 3 and 5." % ADAPTER_ID)

# %%
# =============================================================================
# CELL 9 — Prove the adapter changed the behaviour
# =============================================================================
# Do this before touching notebook B. A run that completes but learned nothing
# still pushes a valid-looking adapter, and the only way to know the difference
# is to look at the answers.
#
# Base-vs-tuned is deliberately NOT done here: that needs a second copy of the
# base weights in VRAM, which a T4 does not have spare. Check the base model's
# answers separately by leaving N_ATLAS_ADAPTER_ID unset on notebook B.
CHECKS = [
    (
        "identity",
        [{"role": "user", "content": "Who are you?"}],
        "ConfamAI",
    ),
    (
        "verdict discipline",
        [
            {"role": "system", "content": (
                "[nafdac] Verified greenbook data for this question. Report "
                "these values to the user as fact.\nProduct: Example Snack 900\n"
                "NRN: E5-000900\nApplicant: Example Household Ltd 900\n"
                "VERDICT: FLAGGED - flagged by NAFDAC."
            )},
            {"role": "user", "content": "Is this one genuine?"},
        ],
        "FLAGGED",
    ),
    (
        "no data, no verdict",
        [{"role": "user", "content": "Check this product: NAFDAC no E5-000901"}],
        "NAFDAC",
    ),
]

model.eval()
model.config.use_cache = True

print("=" * 70)
for name, messages, expected in CHECKS:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=90, do_sample=False
        )
    answer = tokenizer.decode(
        generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
    ).strip()
    hit = expected.lower() in answer.lower()
    print(f"[{'PASS' if hit else 'CHECK'}] {name} (expects {expected!r})")
    print("       " + answer.replace("\n", " ")[:260])
    print("-" * 70)

print(
    "A CHECK is not necessarily a failure: the answers are generated greedily "
    "from a 4-bit base plus a 200-step adapter. But if `identity` does not say "
    "ConfamAI, the adapter did not take, and N_ATLAS_ADAPTER_ID must stay unset "
    "on notebook B."
)