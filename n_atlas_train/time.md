# Training time estimate — ConfamAI QLoRA run

**Answer: ~1.5 hours on Kaggle GPU T4 x2**, of which ~1.3 h is training and
10–20 minutes is setup. Only one of the two T4s is used (§3).

Computed 5 October 2026 for the configuration in `train_notebook.py`:
`max_steps=200`, effective batch 8 (per-device 1 × accumulation 8),
`MAX_SEQ_LENGTH=2048`, LoRA r=16 on `q_proj, k_proj, v_proj, o_proj`,
643 records, stock HF Trainer + bitsandbytes + PEFT (no Unsloth).

The first three sections of this document are the *prediction* as it stood
before the run, left in place because the reasoning is what was wrong. §2.1
records the measurement that replaced it.

---

## 1. The calculation, as predicted

| Quantity | Value | Where it comes from |
|---|---|---|
| Samples seen | **1600** | `max_steps=200` × effective batch 8 (1 × accum 8) |
| Tokens per record | **~1200** | prompt is 3392 chars ≈ 950 Llama-3 tokens, plus block, turn, answer, ChatML |
| Total tokens | **~2.0M** | × ~1.05 for padding |
| Throughput, 1× T4 | **~200 tok/s** | derived in §2 — **wrong by 2.2×, see §2.1** |
| Training time | **3.2 h** | predicted. **Measured 1.3 h** (§2.1) |

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

## 2.1 What the run actually measured

From the first run, 5 October 2026: **12 optimizer steps in 3 m 54 s**, i.e.
**19.5 s/step** at effective batch 8, which puts the full 200 steps at **1.3 h**.

| Quantity | Measured | How |
|---|---|---|
| Seconds per optimizer step | **19.5** | 234 s ÷ 12 steps, from the progress bar |
| Training time, 200 steps | **1.3 h** | 200 × 19.5 s |
| Throughput, counting padding | **~705 tok/s** | 1600 × 2048 padded positions ÷ 4650 s |
| Throughput, real tokens | **~430 tok/s** | the ~2.0M of §1 ÷ 4650 s |
| VRAM after the fix | **7.3 of 14.6 GiB** | Cell 7's own line, batch 1 |
| Training loss at step 10 | **2.27** | healthy for LoRA SFT on a base model |

**The prediction was 2.2× pessimistic**, and the bandwidth argument in §2 is what
failed. The 695 tok/s 3090 figure is for full-finetune-shaped work — all linear
layers adapted, dense 4-bit dequantisation on every matmul. This run adapts
`q_proj, k_proj, v_proj, o_proj` only, so most of the model is frozen and
skipped, and a T4's spare compute goes further than the 1/3 bandwidth ratio
assumes. Per-device batch 1, which cost ~15% on paper, turned out to cost
nothing measurable: the loss peak is halved but the forward pass is not the
bottleneck.

The first 12 steps also include bitsandbytes kernel warm-up, so the steady-state
rate is slightly better than 19.5 s/step.

**Use 430 tok/s per T4 for this configuration** — LoRA on attention projections
only, stock TRL — and not 200. For a heavier adaptation (`all-linear`, as
Unsloth's defaults do) the original figure is closer to right, which is why the
3090 anchor and this run disagree.

## 3. Does T4 x2 double it? No — it cannot be used at all

Corrected 5 October 2026, after a run that died on its first step.

HF Trainer wraps a single-process run in `nn.DataParallel`, not DDP. That was
assumed here to cost a little throughput (a 1.2–1.5× speedup) and to be worth
keeping. It is not: on this stack DataParallel does not run at all.
`Trainer.compute_loss` puts the scalar `num_items_in_batch` into the batch before
calling the model, and `DataParallel.scatter` cannot chunk a 0-dim tensor across
replicas, so step 1 fails with `RuntimeError: chunk expects at least a
1-dimensional tensor`
([transformers#37151](https://github.com/huggingface/transformers/issues/37151),
[trl#2338](https://github.com/huggingface/trl/issues/2338)). The same
`compute_loss` line has broken a lot of single-process multi-GPU runs.

So Cell 1 sets `CUDA_VISIBLE_DEVICES=0` and the second T4 idles. The supported
multi-GPU path is `accelerate launch` (DDP), which a Kaggle cell cannot reach —
and DDP with 4-bit QLoRA is its own experiment, not something to discover four
hours into a run. The 1.35× factor is therefore gone from §1, and the estimate is
a single-T4 number.

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

The two methods bracket **2.8–3.6 h on one T4**, which is where the ~2.8 h figure
in §1 comes from. No published T4 x2 number for 8B QLoRA was found, which is why
two independent anchors are used rather than a single table row.

Both are wrong by about the same factor, and §2.1 says why: the 3B anchor was
Unsloth with adapters on every linear layer, so scaling it by 8/3 and then
doubling for "stock is half as fast" compounded two penalties. Measured, the
1600-sample shape takes **1.3 h** on one T4. The third anchor — the only one that
counts — is the one in §2.1.

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

**This happened: OOM at batch 2 × 2048 tokens, on the first step.** The prediction
below was right that batch 2 was the risk and that batch 1 × accumulation 8 was
the fix; the reasoning about *where* the memory goes was wrong. The weights are
not the problem — 4-bit NF4 plus LoRA is ~5 GB, and gradient checkpointing keeps
activations under 1 GB. The peak is the **logits**: Llama-3's vocab is 128256, so
at 2048 padded positions one sequence is 501 MB in fp16, and `cross_entropy`
upcasts that to fp32 and keeps a softmax buffer for the backward pass. Call it
~2 GB per sequence, on top of the ~7 GB the quantized model and the fp32
embedding copy of `prepare_model_for_kbit_training` already hold — ~12 GB before
the last 1 GB allocation, against a T4's usable 14.56 GB.

Hence the two defaults that are now in the notebook: **per-device batch 1 with 8
accumulation steps** (effective batch unchanged at 8, so the step count, the
schedule and the 1600 samples all stay the same; 8 smaller forward passes per
optimizer step instead of 4 larger ones, which §2.1 measures as costing nothing),
and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which reclaims the 1.27 GB
that was reserved-but-unallocated when it failed. Batch 1 is the floor — the next
lever, if it is ever needed again, is `N_ATLAS_MAX_SEQ` lowered to Cell 6's
measured max, which saves VRAM linearly.

## 8. What I would actually run

Run the **default 200 steps** — at the measured 19.5 s/step that is 1.3 h, cheap
enough to iterate on, and 2.5 passes is the right amount for LoRA on 643 records.
The loss at step 10 (2.27) and its slope over the next 40 steps are the real
signal, not the step count; if it has plateaued by step 100, stop it there and
keep the rest of the week's budget.

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
  A ~1.5-hour run is well inside either figure.
- **60 minutes of idle time** before an interactive session ends. Logging every
  10 steps keeps the session alive on its own.

So the run is now a small part of the week's budget: ~1.5 h of the ~30 available,
which means iterating on the data and the prompt is cheap. That is a change from
when this was costed at 4 h and looked like a third of the quota.

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

**Done, 5 October 2026: it was 2.2× off, and §2.1 now carries the measured 430
tok/s.** The quoted figure in this folder is **~1.5 h** total, replacing the
original guide's "about 2 hours".
