# ConfamAI — QLoRA fine-tuning (Notebook A)

Trains the ConfamAI adapter with QLoRA and pushes it to Hugging Face. This folder
is the **training** half of the project and serves nothing.

| | This folder (Notebook A) | `n_atlas_server/` (Notebook B) |
|---|---|---|
| Job | trains the adapter, pushes it, ends | serves chat requests on port 8000 |
| Kaggle dataset attached | `confam-training` | `n-atlas-source` |
| Lifetime | one run, ~2.5–4 h, then stop it | as long as you leave it running |
| Your laptop client points at it | never | **only this one** |

The only thing that crosses between them is the adapter on the Hugging Face Hub.
Notebook B picks it up by setting `N_ATLAS_ADAPTER_ID`; nothing else changes, and
`served_model_name` stays `n-atlas` so no client is reconfigured.

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
3. Accelerator: **GPU T4 x2**.
4. **Accept N-ATLaS's licence on Hugging Face first.** `NCAIR1/N-ATLaS` is gated:
   `from_pretrained` fails with a 401 until your account has agreed to share
   contact information. The server notebook needs the same access.
5. Budget **~2.5–4 hours** for the default 200 steps on T4 x2 — the arithmetic is
   under "Timing" below. Start with a lower `N_ATLAS_TRAIN_STEPS` if this is your
   first run.

**Then**

1. `python build_train_notebook.py`
2. `python validate_train_notebook.py` — must print `VALID`
3. Upload `confam_train_kaggle.ipynb` to Kaggle, attach `confam-training`,
   attach the `HF_TOKEN` secret, run all cells in order.

## The nine cells

| Cell | Job | Fails loudly if |
|---|---|---|
| 1 | config, finds the dataset in `/kaggle/input/**` | the JSONL is not attached, listing what is mounted |
| 2 | installs libraries, `trl` pinned to 0.12.2 | no GPU is available |
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
Adapter:           Saynetng/confam-ai-adapter
```

and `GET /v1/health` must report `"adapter_id"`. If either says `none`, the
variable did not reach the process. A configured adapter that fails to load stops
Notebook B rather than falling back to the base model, so the two can never
silently disagree.

## Timing

The default run is `max_steps=200` at effective batch 8, so **1600 samples** of
~1200 tokens each — about **2.0M tokens**. N-ATLaS is **Llama-3 8B**, and a T4 has
roughly a third of an RTX 3090's bandwidth, so a stock run lands near 200 tok/s
on one T4; `nn.DataParallel` over two T4s adds 1.2–1.5×, not 2×.

That gives **~2–3 h of training plus 10–20 min of setup**, so budget **~2.5–4 h**.

For the full arithmetic, the research anchors it is based on, the Kaggle limits it
sits inside, and how to correct the estimate from the real `it/s` after the first
run, see **[time.md](time.md)**.

**Levers, in the order I would pull them:** `N_ATLAS_TRAIN_STEPS=120` (about 1.5
epochs, ~1.5–2 h — my recommendation for a first run), then lowering
`MAX_SEQ_LENGTH` to whatever Cell 6 reports as the real maximum plus headroom,
then Unsloth in place of stock TRL if you want roughly 2× at the cost of another
dependency.

**Limits that are not negotiable:** Kaggle allows ~30 GPU-hours per week (reset
Saturday 00:00 UTC) and caps a single GPU session at 12 hours, so this run fits
both — but a 4-hour session is a third of your weekly budget.

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

## Tuning knobs

Set as environment variables in Cell 1, or edit `train_notebook.py` and rebuild:

| Variable | Default | Note |
|---|---|---|
| `N_ATLAS_MODEL_ID` | `NCAIR1/N-ATLaS` | base model |
| `N_ATLAS_ADAPTER_ID` | `Saynetng/confam-ai-adapter` | where this notebook pushes **and** what Notebook B reads |
| `N_ATLAS_TRAIN_STEPS` | `200` | effective batch 8, so ~2.5 passes over 643 records |
| `N_ATLAS_TRAIN_BATCH` / `N_ATLAS_TRAIN_ACCUM` | `2` / `4` | lower the batch before raising steps if VRAM is tight |
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
