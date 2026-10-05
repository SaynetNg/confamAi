# Training time estimate — ConfamAI QLoRA run

**Answer: ~2.5 to 4 hours on Kaggle GPU T4 x2**, of which roughly 2–3 hours is
training and 10–20 minutes is setup.

Computed 5 October 2026 for the configuration in `train_notebook.py`:
`max_steps=200`, effective batch 8 (per-device 2 × accumulation 4),
`MAX_SEQ_LENGTH=2048`, LoRA r=16 on `q_proj, k_proj, v_proj, o_proj`,
643 records, stock HF Trainer + bitsandbytes + PEFT (no Unsloth).

---

## 1. The calculation

| Quantity | Value | Where it comes from |
|---|---|---|
| Samples seen | **1600** | `max_steps=200` × effective batch 8 (2 × accum 4) |
| Tokens per record | **~1200** | prompt is 3392 chars ≈ 950 Llama-3 tokens, plus block, turn, answer, ChatML |
| Total tokens | **~2.0M** | × ~1.05 for padding |
| Throughput, 1× T4 | **~200 tok/s** | derived in §2 |
| Training time | **2–3 h** | 2.0M ÷ 200, × 2 T4s at ~1.35× |

## 2. Where ~200 tok/s comes from

This is the weak point in the estimate, so the derivation is spelled out.

| GPU | fp16 tensor TFLOPS | Memory bandwidth |
|---|---|---|
| Tesla T4 | ~8.7 | 320 GB/s |
| RTX 3090 | ~35 | 936 GB/s |

QLoRA is partly **bandwidth**-bound, not just compute-bound: every 4-bit weight
is dequantised on the fly during the forward pass. So the relevant scaling factor
from a 3090 to a T4 is the bandwidth ratio, roughly **1/3**, rather than the
compute ratio.

Published Llama-3 8B QLoRA throughput on an RTX 3090 is ~695 tok/s (1K examples
in 24 minutes). Divided by ~3 that gives **~200 tok/s on one T4**.

## 3. Does T4 x2 double it? No

HF Trainer wraps a single-process run in `nn.DataParallel`, not DDP. The forward
pass is replicated across both GPUs and the gradients are gathered back to GPU 0,
which is a bottleneck. Realistic speedup is **1.2–1.5×**, not 2×.

This is the single most common over-estimate in T4 x2 fine-tuning advice, and it
is why the estimate above is not simply half of a single-GPU figure.

## 4. Setup overhead, 10–20 minutes

| Step | Time |
|---|---|
| `pip install` (transformers, trl 0.12.2, peft, bitsandbytes, datasets, accelerate) | 3–7 min |
| Download the gated 8B model (~16 GB of BF16 safetensors) from Hugging Face | 2–6 min |
| Render and tokenise 643 records with the chat template | 1–3 min |
| First-step bitsandbytes kernel compile / warm-up | 1–3 min |
| Cell 9 behavioural checks (3 × 90-token generations) | 1–3 min |
| Push the adapter to the Hub | < 1 min |

---

## 5. Research anchors

| Source | Model | GPU | Config | Time |
|---|---|---|---|---|
| [gigagpu.com](https://gigagpu.com/fine-tuning-time-by-gpu) | Llama-3 8B QLoRA r16 | RTX 3090 | 1K examples | ~24 min |
| [gigagpu.com](https://gigagpu.com/fine-tuning-time-by-gpu) | same | RTX 4060 8GB | 1K examples | ~55 min |
| [Kethanvr/qwen-fine-tuning](https://github.com/Kethanvr/qwen-fine-tuning) | Qwen2.5-**3B** QLoRA | 1× T4 (Colab) | 200 steps, eff 8, seq 2048 | ~40 min |
| [saul-gasca/Alpaca_FineTunning](https://github.com/saul-gasca/Alpaca_FineTunning) | Llama-3.1 8B QLoRA + Unsloth | RTX 5070 Ti | 60 steps, eff 8, seq 2048 | ~10 min |
| [arXiv 2509.12229](https://arxiv.org/abs/2509.12229) | Qwen2.5-1.5B | RTX 4060 | measured | 628 tok/s with paged optimisers |
| [aifromzero.dev](https://aifromzero.dev/courses/finetune-prod/module-05) | 7–8B QLoRA | free T4 | — | "fits a free T4 — but slowly, in the hours" |
| [computingforgeeks.com](https://computingforgeeks.com/fine-tune-llm-unsloth-qlora) | Llama-3.1 8B QLoRA + Unsloth | RTX 4090 | short demo run | 34 s |

## 6. Cross-check from the other direction

The 3B-on-T4 result is the *same 1600-sample shape* as this run, so scaling by
parameter count gives 40 min × (8/3) ≈ **1.8 h** on one T4. But that run used
Unsloth, which is roughly 2× faster than stock HF/TRL. On stock, the same figure
is **~3.6 h**.

The two methods bracket **2–3.6 h on one T4**, which is where the 2–3 h figure in
§1 comes from. No published T4 x2 number for 8B QLoRA was found, which is why
two independent anchors are used rather than a single table row.

The AQLoRA paper ([arXiv 2608.23816](https://arxiv.org/pdf/2608.23816)) is worth
knowing about for a second reason: it notes that QLoRA is *slower* than fp16 LoRA
because of that on-the-fly dequantisation, and reports ~11% recoverable with a
tuned recipe. We are not using that recipe, so the stock number applies.

---

## 7. Two things that could stretch this badly

**N-ATLaS is a gated repo.** The [model page](https://huggingface.co/NCAIR1/N-ATLaS)
requires agreeing to share contact information before `from_pretrained` works.
Accept it first, or Cell 5 fails immediately and no GPU time is spent.

**The licence is capped at 1000 active end-users**, and N-ATLaS is explicitly
"not an enterprise-grade or commercial system" — commercial or large-scale use
requires a separate agreement with Awarri Technologies. For a consumer-products
business at scale that is a commercial question, not a technical one.

Less likely but possible: OOM at batch 2 × 2048 tokens. The 8B NF4 base is ~5 GB
and gradient checkpointing keeps activations modest, so ~12–14 GB of the T4's
16 GB should hold. If it does not, drop to batch 1 with accumulation 8 — same
effective batch, roughly the same wall clock.

## 8. What I would actually run

Start with **`N_ATLAS_TRAIN_STEPS=120`** — about 1.5 epochs, roughly 1.5–2 h.
LoRA on 643 records rarely needs more, and the loss tells you within about 40
steps whether 200 was ever going to help. If it plateaus early you have your
answer and a third of your weekly GPU budget left.

Then, in order of value:

1. **Lower `MAX_SEQ_LENGTH`** to whatever Cell 6 reports as the real maximum plus
   headroom. Attention is quadratic in sequence length, so this pays more than
   raising the batch size.
2. **Lower `N_ATLAS_TRAIN_STEPS`** further if the loss plateaus.
3. **Unsloth instead of stock TRL** if you need roughly 2× and can accept the
   extra dependency. It still saves a standard PEFT adapter, which is what
   notebook B loads.

## 9. Kaggle limits this sits inside

Verified from [kaggle.com/docs/notebooks](https://www.kaggle.com/docs/notebooks)
and [GPU tips](https://www.kaggle.com/page/GPU-tips-and-tricks):

- **~30 GPU-hours per week**, resetting Saturday 00:00 UTC, sometimes higher under
  the "floating" quota.
- **12 hours per GPU session** (the tips page still says 9 hours for older text).
  A 4-hour run is inside either figure.
- **60 minutes of idle time** before an interactive session ends. Logging every
  10 steps keeps the session alive on its own.

So the run fits the platform, but at 4 hours it consumes roughly a third of the
week's GPU budget. That is the real constraint on iterating.

## 10. Correcting this estimate after the first run

This is a prediction, not a measurement. Once Cell 7 has run 10 steps, read the
real throughput off the log line (`N/total`, in tokens/second) and recompute:

```
remaining_seconds = total_tokens / measured_tokens_per_second
```

The `it/s` and `s/it` fields in the progress bar, multiplied by effective batch 8
and the measured average record length from Cell 6, give the same figure. If the
real number is more than ~1.5× off from 200 tok/s per T4, this document's
throughput figure is wrong and should be corrected here rather than left to drift.

The docs in this folder now quote **~2.5–4 h**, replacing the original guide's
"about 2 hours".
