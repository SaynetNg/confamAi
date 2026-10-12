# ConfamAI — QLoRA fine-tuning (Notebook A)

Trains the ConfamAI adapter with QLoRA and pushes it to Hugging Face. This folder
is the **training** half of the project and serves nothing.

| | This folder (Notebook A) | `n_atlas_server/` (Notebook B) |
|---|---|---|
| Job | trains the adapter, pushes it, ends | serves chat requests on port 8000 |
| Kaggle dataset attached | `confam-training` | `n-atlas-source` |
| Lifetime | one run, ~1.5 h, then stop it | as long as you leave it running |
| Your laptop client points at it | never | **only this one** |

The only thing that crosses between them is the adapter on the Hugging Face Hub.
Notebook B picks it up by setting `N_ATLAS_ADAPTER_ID`; nothing else changes, and
`served_model_name` stays `n-atlas` so no client is reconfigured.

## Hugging Face Hub

The artefacts this project produces and consumes, all under the `Saynetng` account:

- **Adapter v1.1** — the QLoRA adapter Notebook A pushes (Cell 8) and Notebook B serves (`N_ATLAS_ADAPTER_ID`):  
  <https://huggingface.co/Saynetng/confam-ai-adapter-v1.1>
- **Training dataset** — the 643-record JSONL that Cell 1 consumes:  
  <https://huggingface.co/datasets/Saynetng/confam_training_dataset>
- **Adapter (unversioned)** — the adapter repo without a version tag:  
  <https://huggingface.co/Saynetng/confam-ai-adapter>

## Files

| File | Role |
|---|---|
| `train_notebook.py` | **source of truth.** Plain sequential Python with `# %%` cell markers: openable in VS Code as an interactive notebook, or buildable to `.ipynb`. |
| `confam_train_kaggle.ipynb` | **build artefact** — upload this to Kaggle. Never hand-edit it. |
| `build_train_notebook.py` | `train_notebook.py` → `confam_train_kaggle.ipynb`. |
| `validate_train_notebook.py` | Compiles every cell, checks nbformat line encoding, and proves the notebook reassembles exactly into `train_notebook.py`. |
| `confam_training_dataset.jsonl` | The 643-record training set notebook A consumes. |
| `prepare_training_data.py` | Rebuilds that dataset from a raw source, or validates it in place when the raw file is gone. |
| `check_transcript_coverage.py` | Compares real saved conversations against what the dataset covers. |
| `qlora_finetuning_guide.md` | The original cell-by-cell walkthrough, kept for reference. **Superseded** by the built notebook and this README — where they differ, the notebook wins. |
| `time.md` | The training-time estimate: arithmetic, research anchors, Kaggle limits, and how to correct it after the first run. |

Mirrors `n_atlas_server/` on purpose: one editable source, one generated upload,
one validator. It reuses the server's `notebook_cells.py` rather than copying the
cell-splitting logic, because a drifting second copy of that logic is how "it
validates locally but breaks on Kaggle" happens.

The training dataset lives here, beside the notebook that consumes it, as
`confam_training_dataset.jsonl` (2.3 MB). Cell 1 looks for that filename wherever
Kaggle mounts it.

This folder also holds everything used to produce and check that file:
`prepare_training_data.py` rebuilds it from a raw source, and
`check_transcript_coverage.py` measures it against the real saved conversations.

## Running it

**Before the first run**

1. Upload `confam_training_dataset.jsonl` to Kaggle as a dataset named
   `confam-training`.
2. Create a Kaggle secret named `HF_TOKEN` with a Hugging Face token that has
   **write** permission. Cell 8 pushes the adapter and cannot work without it.
   The account on that token is the first half of `ADAPTER_ID`.
3. Accelerator: **GPU T4 x2**. The second T4 is deliberately left idle — see
   "Timing" below for why training on both of them does not work.
4. **Accept N-ATLaS's licence on Hugging Face first.** `NCAIR1/N-ATLaS` is gated:
   `from_pretrained` fails with a 401 until your account has agreed to share
   contact information. The server notebook needs the same access.
5. Budget **~1.5 hours** for the default 200 steps on one T4 — measured, see
   "Timing" below.

**Then**

1. `python build_train_notebook.py`
2. `python validate_train_notebook.py` — must print `VALID`
3. Upload `confam_train_kaggle.ipynb` to Kaggle, attach `confam-training`,
   attach the `HF_TOKEN` secret, run all cells in order.

## The nine cells

| Cell | Job | Fails loudly if |
|---|---|---|
| 1 | config, finds the dataset in `/kaggle/input/**` | the JSONL is not attached, listing what is mounted |
| 2 | installs libraries, `trl` pinned to 0.12.2 | no GPU is available, or more than one is visible (Cell 1 must run first) |
| 3 | Hugging Face login | `HF_TOKEN` is missing, before two hours are spent |
| 4 | **dataset integrity checks** | personas are mixed, a verdict has no block, or a record claims counterfeit / safety |
| 5 | base model in 4-bit NF4 + LoRA | the base model exposes no chat template |
| 6 | renders records with the model's own template, **measures token lengths** | any record exceeds `MAX_SEQ_LENGTH`, which would truncate the assistant turn |
| 7 | trains 200 steps | — |
| 8 | pushes the adapter to the Hub | the pushed repo is implausibly large, meaning base weights went up instead of the adapter |
| 9 | three behavioural checks | prints PASS/CHECK; a CHECK is not automatically a failure, but `identity` not saying ConfamAI means the adapter did not take |

Cell 4 runs **before** the 5-minute model load, so a bad upload fails in seconds.
Cell 6's length measurement is not decoration: all 643 records repeat the same
~850-token system prompt, so a too-small `max_seq_length` silently truncates the
assistant turn — the only part of a record that teaches anything.

## After it works

Set `N_ATLAS_ADAPTER_ID` on Notebook B (a Kaggle secret named
`N_ATLAS_ADAPTER_ID` is the clean way), then re-run Cells 1, 2, 3 and 5 of
`natlas_kaggle.ipynb`. Cell 2's banner must print:

```
Adapter:           Saynetng/confam-ai-adapter-v1.1
```

and `GET /v1/health` must report `"adapter_id"`. If either says `none`, the
variable did not reach the process. A configured adapter that fails to load stops
Notebook B rather than falling back to the base model, so the two can never
silently disagree.

## Successful training run example

A successful run of the training notebook produces output like this:

**Cell 6 — Dataset rendering:**
```
[data] rendering 516 records with the model's chat template…
[data] rendered in 1s
[data] tokens  p50=914  p90=989  max=1080
[data] max_seq_length=2048, 0 records over it, every batch padded to 2048
```

**Cell 7 — Training:**
```
[train] 516 records, 200 steps, effective batch 8 (1 per step x 8 accum)
[train] weights resident 7.3 GiB of 14.6 GiB; logits 501 MiB per sequence at 2048 padded positions, ~3x that in the loss

 [200/200 1:45:51, Epoch 3/4]
Step    Training Loss
10      2.153200
20      0.641900
30      0.160500
40      0.102200
50      0.094000
...
190     0.041100
200     0.043900

[train] done in 107 minutes
```

**Cell 8 — Push to Hub:**
```
[push] Saynetng/confam-ai-adapter-v1.1
[push] 4 files, 0.0 MB
       .gitattributes
       README.md
       adapter_config.json
       adapter_model.safetensors
```

**Cell 9 — Behavioral checks:**
```
[CHECK] identity (expects 'ConfamAI')
       I am a language model developed by OpenAI...

[PASS] verdict discipline (expects 'FLAGGED')
       I can't verify whether Example Snack 900 is genuine because NAFDAC flagged it...

[CHECK] no data, no verdict (expects 'NAFDAC')
       I'm not a product verification service...
```

**What the checks mean:**

- **Training loss trajectory**: Should drop from ~2.15 to ~0.04-0.06 over 200 steps. The
  loss curve shows the model learning: rapid drop in the first 30 steps (2.15 → 0.16),
  then gradual refinement. Final loss around 0.04 is typical for this dataset size.

- **PASS vs CHECK**: Cell 9 runs three behavioral tests. A CHECK is not automatically a
  failure — these are greedy generations from a 4-bit base plus a 200-step adapter.
  The critical signal is the **identity** check:
  - If it says `"ConfamAI"` or similar → adapter took effect
  - If it says `"OpenAI"` or generic LLM identity → **adapter did not take**, do not
    set `N_ATLAS_ADAPTER_ID` on Notebook B

- **Adapter size**: Should be tens of megabytes (~50-80 MB), not gigabytes. If Cell 8
  reports 500+ MB, the push uploaded base weights instead of just the adapter, and the
  `N_ATLAS_ADAPTER_ID` must not be used.

- **Training time**: 107 minutes (1h 47m) for 200 steps matches the ~1.5-1.8h budget.
  The measured rate is ~32 seconds/step at effective batch 8 with a T4.

If the identity check fails (shows OpenAI or generic LLM), the adapter did not learn
the ConfamAI persona. This can happen if:
- The training steps were too few (increase `N_ATLAS_TRAIN_STEPS`)
- The learning rate was too low (default 2e-4 is usually fine)
- The dataset had issues (Cell 4 validates this)
- The adapter was not saved correctly (check Cell 8 output)

## Timing

The default run is `max_steps=200` at effective batch 8, so **1600 samples** of
~1000 tokens each — about **1.6M tokens**. N-ATLaS is **Llama-3 8B**, and a T4 has
roughly a third of an RTX 3090's bandwidth.

**Only one GPU is used, on purpose, and it is Cell 1 that decides it.**
`device_map="auto"` does not replicate the model across the T4 x2, it shards the
layers, which leaves the logits on `cuda:1` while the loss is divided by a scalar
on `cuda:0` and training dies with *"Expected all tensors to be on the same
device"* — Cell 5 pins the model to `cuda:0` for that. But pinning the weights is
not enough on its own: with two GPUs *visible*, `Trainer` wraps the model in
`torch.nn.DataParallel`, and `DataParallel` cannot scatter the 0-dim
`num_items_in_batch` tensor that `compute_loss` adds to every batch, so step 1 dies
with *"chunk expects at least a 1-dimensional tensor"*
([transformers#37151](https://github.com/huggingface/transformers/issues/37151),
[trl#2338](https://github.com/huggingface/trl/issues/2338)). Cell 1 therefore sets
`CUDA_VISIBLE_DEVICES=0` before anything imports torch, so the count of visible
devices is 1 and no wrapper is created. A 4-bit 8B model with LoRA ranks fits one
T4, so this costs throughput rather than correctness — and the single-T4 estimate
below is the one to plan against. **Restart the kernel and re-run from Cell 1 if
you ever edit that line**: torch caches the device count on first use, so Cell 1
has to be the first cell that touches CUDA.

That gives **~1.3 h of training plus 10-20 min of setup**, so budget **~1.5 h**.

That number is measured, not predicted: the first run did 12 optimizer steps in
3 m 54 s, i.e. 19.5 s/step, which is 200 steps in 1.3 h. The 4-hour figure this
section used to quote came from scaling a published RTX 3090 number down by the
T4's bandwidth ratio, and it was 2.2× pessimistic — adapting `q_proj, k_proj,
v_proj, o_proj` only skips most of the model, so a T4's spare compute goes
further than the bandwidth ratio assumes. **time.md §2.1 has the arithmetic and
§10 has the correction.**

For the full arithmetic, the research anchors it is based on, the Kaggle limits it
sits inside, and how to correct the estimate from the real `it/s` after a run,
see **[time.md](time.md)**.

**Levers, in the order I would pull them:** lower `MAX_SEQ_LENGTH` to whatever
Cell 6 reports as the real maximum plus headroom (attention is quadratic in it,
and it is also the VRAM lever — see below), then `N_ATLAS_TRAIN_STEPS` if the
loss plateaus, then Unsloth in place of stock TRL if you want roughly 2× at the
cost of another dependency. At 1.3 h the default 200 steps is cheap enough to
just run.

**The VRAM floor is per-device batch 1.** The peak is not the weights (~5 GB for
4-bit NF4) but the logits: Llama-3's vocab is 128256, so one 2048-token sequence
is 501 MB in fp16 and ~2 GB once `cross_entropy` has upcast it to fp32 and kept a
softmax buffer for backward. Batch 2 does not fit a T4's 14.56 GB and dies in
`ForCausalLMLoss` on step 1; batch 1 holds at 7.3 GiB. Do not raise
`N_ATLAS_TRAIN_BATCH` — raise `N_ATLAS_TRAIN_ACCUM` alongside it if you need a
larger effective batch.

**Limits that are not negotiable:** Kaggle allows ~30 GPU-hours per week (reset
Saturday 00:00 UTC) and caps a single GPU session at 12 hours, so this run fits
both comfortably — at 1.5 h it is 5% of the week's budget, which leaves plenty of
room to iterate on the dataset and the prompt.

## What fine-tuning cannot do here

- **It cannot add consumer products to the database.** The greenbook export
  holds Drugs, Medical devices, Vaccines and Biologics, Herbals and
  Nutraceuticals and Veterinary — and zero food, beverage, cosmetic, household or
  textile rows. After training, the assistant still finds nothing for those,
  because that is a data gap in `nafdac_source`, not something the weights can
  invent. Fix it by loading a consumer dataset or adding a category-coverage
  guard. Note the distinction: the *training records* use real Nigerian
  consumer and medicine names as vocabulary, but every registration value in them
  is synthetic, so the model learns to copy the block and can never supply a
  number the database does not hold.
- **It does not replace the prompt.** `confam_prompt.txt` remains the served
  instruction and the guardrail layer; the adapter only shifts behaviour the
  prompt could not hold on its own.
- **There is no held-out set.** 643 records, ~2.5 passes, no evaluation split, so
  you cannot separate learning from memorisation. Cell 9 is the only signal you
  get. If you want a real measurement, hold out 10% before training and score
  both models on it.

## Training configuration

The values Cell 1, Cell 5 and Cell 7 actually run with:

| Setting | Value | Where |
|---|---|---|
| Quantization | 4-bit NF4 with double quantization (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`) | Cell 5 |
| Compute dtype | float16 (`bnb_4bit_compute_dtype=torch.float16`, `fp16=True`) — a T4 is Turing-class and has no bf16 support | Cell 5, Cell 7 |
| LoRA rank (r) | 16 | Cell 1 `LORA_RANK` |
| LoRA alpha | 32 | Cell 1 `LORA_ALPHA` |
| LoRA dropout | 0.05 | Cell 5 `LoraConfig` |
| LoRA targets | `q_proj, k_proj, v_proj, o_proj`, `bias="none"`, `task_type="CAUSAL_LM"` | Cell 1, Cell 5 |
| Learning rate | 2e-4, cosine schedule, 10 warmup steps | Cell 1 `N_ATLAS_TRAIN_LR`, Cell 7 |
| Epochs | `num_train_epochs=1`, but the run is step-limited: `max_steps=200` at effective batch 8, so ~2.5 passes over the 643 records | Cell 7 |
| Max sequence | 2048 (`N_ATLAS_MAX_SEQ`); Cell 6 measures every record in tokens and fails the run if any exceeds it | Cell 1, Cell 6 |
| Optimizer | `paged_adamw_8bit` | Cell 7 |
| Batch | per-device 1 × gradient accumulation 8 = effective batch 8; 1 is the VRAM floor (see "Timing") | Cell 1, Cell 7 |
| Other | gradient checkpointing with `use_reentrant=False`, `packing=False`, seed 42, `save_steps=50`, `save_total_limit=2`, `report_to="none"` | Cell 7 |

## Tuning knobs

Set as environment variables in Cell 1, or edit `train_notebook.py` and rebuild:

| Variable | Default | Note |
|---|---|---|
| `N_ATLAS_MODEL_ID` | `NCAIR1/N-ATLaS` | base model |
| `N_ATLAS_ADAPTER_ID` | `Saynetng/confam-ai-adapter-v1.1` | where this notebook pushes **and** what Notebook B reads |
| `N_ATLAS_TRAIN_STEPS` | `200` | effective batch 8, so ~2.5 passes over 643 records |
| `N_ATLAS_TRAIN_BATCH` / `N_ATLAS_TRAIN_ACCUM` | `1` / `8` | effective batch 8, so 200 steps is 1600 samples either way. `1` is the VRAM floor: at batch 2 the logits alone outgrow a T4. Do not raise it without re-reading the Cell 1 arithmetic |
| `N_ATLAS_TRAIN_LR` | `2e-4` | conventional LoRA rate; lower it if the answers drift from the prompt |
| `N_ATLAS_TRAIN_OUTPUT` | `/kaggle/working/confam-ai-adapter` | local checkpoints, not the push target |
| `N_ATLAS_MAX_SEQ` | `2048` | raise only if Cell 6 reports a record over the limit; lowering it truncates the assistant turn |

## Rebuilding the dataset

`confam_training_dataset.jsonl` is produced by `prepare_training_data.py` from a
raw source file. That raw file was never committed, so if it is gone the builder
validates the existing dataset in place instead of rebuilding, and says so:

```bash
python n_atlas_train/prepare_training_data.py --check-only
```

If you change `confam_prompt.txt` or the `PRODUCT_STEMS` / `APPLICANT_STEMS`
tables, the file on disk no longer matches them and `--check-only` will say so.
Move it forward in place rather than rebuilding, because the raw dataset is gone:

```bash
python n_atlas_train/prepare_training_data.py --migrate
```

It rewrites the system prompt and the product/applicant names in every record,
checks the result, and writes only if the checks pass. It is idempotent, so
running it twice changes nothing the second time. After changing
`confam_prompt.txt` also run `python n_atlas_server/check_prompt_drift.py`, which
fails when the three Python copies of the prompt fall out of step with the file.

Both scripts find `confam_prompt.txt` by walking up to the repository root, so
they work from the repository root or from inside this folder.
