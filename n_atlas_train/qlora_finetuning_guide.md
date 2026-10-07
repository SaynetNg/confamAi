# Fine-Tuning Confam AI with QLoRA

> **Superseded.** This is the original cell-by-cell walkthrough, kept for
> reference. The maintained version is `confam_train_kaggle.ipynb`, built from
> `train_notebook.py` — see `README.md`. The notebook does the same work with
> input discovery, dataset integrity checks, real token-length measurement and a
> behavioural check before the push. **Where this guide and the notebook differ,
> the notebook wins.**

You now have the cleaned training dataset: **643 examples** in
`confam_training_dataset.jsonl`, rebuilt from the original 866 by
`prepare_training_data.py`. Time to bake this behaviour into the model's weights.

**This file is the one to upload — not the raw dataset it replaced.** The raw
866-record file asserted a verdict with no data attached in 73 examples (three of
them calling a product counterfeit), carried invented registry facts, claimed
"it is safe for use", and used a different persona from the one the server
serves. The cleaned file has none of that: every verdict sits next to a verified
block, every value in that block is synthetic, and every system message is
`confam_prompt.txt` verbatim. Prove it any time with
`python prepare_training_data.py --check-only`.

What fine-tuning still cannot fix: the loaded dataset has **no consumer
products** in it, so the assistant will still find nothing for food, drinks or
cosmetics. That is a data problem, not a weights problem.

We will do this on **Kaggle** using a technique called **QLoRA** (Quantized Low-Rank Adaptation). Instead of retraining the entire massive model (which requires supercomputers), QLoRA freezes the base model and trains tiny "adapter" layers on top of it. On Kaggle's free T4 x2, budget **~2.5–4 hours** for the default 200 steps — see the Timing section of `README.md` for the arithmetic.

---

## Step 1: Upload the Dataset to Kaggle
1. Go to [kaggle.com/datasets](https://kaggle.com/datasets) and click **New Dataset**.
2. Name it `confam-training`.
3. Upload the `confam_training_dataset.jsonl` file from your PC.
4. Click **Create**.

## Step 2: Prepare the Training Notebook
1. Create a **New Notebook** on Kaggle.
2. Under **Accelerator** on the right, select **GPU T4 x2**.
3. Click **Add Data** (top right), search for the `confam-training` dataset you just uploaded, and add it.
4. Go to **Add-ons -> Secrets** and ensure your `HF_TOKEN` (Hugging Face token) is attached.

## Step 3: The Training Code

Paste these cells into your Kaggle notebook and run them in order.

### Cell 1: Install Libraries
```python
!pip install -q -U transformers datasets accelerate bitsandbytes peft trl==0.12.2
```

`peft` is what serves the adapter later, and what trains it here.

**Why `trl` is pinned.** The `SFTTrainer` call in Cell 5 uses the 0.12 argument
names (`max_seq_length=`, `tokenizer=`). TRL renamed them to `max_length=` and
`processing_class=` in later releases, and the old names are then forwarded to
`TrainingArguments`, which rejects them with an unhelpful error about an unknown
argument. Pinning keeps this notebook working. Remove the pin only if you also
update Cell 5.

### Cell 2: Login to Hugging Face
```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)
```

### Cell 3: Load the Model and Tokenizer
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model_id = "NCAIR1/N-ATLaS"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model in 4-bit (This is the 'Q' in QLoRA)
# compute_dtype MUST be float16 on a T4. The T4 is Turing-class and has no
# bfloat16 support; bf16 here fails inside bitsandbytes or silently falls back,
# and Cell 5 also asks for fp16. The inference server uses float16 for the same
# reason, so training and serving agree.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto"
)

# Prepare for LoRA training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Define the LoRA Adapters (This is the 'LoRA' in QLoRA)
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```
*(You will see that only about 0.1% of the model parameters are actually being trained!)*

### Cell 4: Prepare the Dataset
Each record is rendered with the model's **own** chat template — N-ATLaS derives
from Llama 3.1, so this is that template and not ChatML. Records carrying a
verified block have two system messages in a row, which is exactly the shape the
server sends at inference.
```python
from datasets import load_dataset

# Load your uploaded dataset
# (Check the exact path in the 'Input' folder on the right side of Kaggle)
dataset = load_dataset("json", data_files="/kaggle/input/datasets/sabuwa/confam-training/confam_training_dataset.jsonl", split="train")

def format_chat_template(example):
    # This applies the ChatML template (e.g. <|im_start|>user...)
    example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return example

print("Formatting dataset...")
dataset = dataset.map(format_chat_template)
print(f"Ready to train on {len(dataset)} examples!")
```

### Cell 5: Start Training!
```python
from trl import SFTTrainer
from transformers import TrainingArguments

output_dir = "/kaggle/working/confam-ai-adapter"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=200, # Start with 200 steps to test it
    save_strategy="steps",
    save_steps=50,
    optim="paged_adamw_8bit",
    fp16=True,
    run_name="confam-ai-finetune"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    # 2048, not 1024. The system prompt alone is ~850 tokens, so the longest
    # records land within a few tokens of 1024 and a real tokenizer overflows
    # that. Truncation cuts the assistant turn off the end, which throws away
    # the only part of the record that teaches anything.
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
)

print("Starting training! This will take a while...")
trainer.train()

# Push the final adapter to Hugging Face Hub!
trainer.model.push_to_hub("Saynetng/confam-ai-adapter-v1.1", token=hf_token)
print(f"Training complete! Adapter uploaded to Hugging Face Hub successfully.")
```

Check it worked before moving on:

1. Open `https://huggingface.co/Saynetng/confam-ai-adapter-v1.1`. You should see
   `adapter_config.json` and `adapter_model.safetensors`. Change the account name
   if your push went somewhere else — the serving step needs the exact id that
   exists.
2. The adapter is small (tens of MB, not gigabytes). If the repo shows the full
   base model, something loaded the weights instead of the adapter.

---

## Step 4: Serve the Adapter from the Kaggle Notebook

The server already knows how to attach an adapter, so this is one environment
variable — no client change, no code edit. In **Cell 1** (or as a Kaggle secret),
set:

```python
os.environ["N_ATLAS_ADAPTER_ID"] = "Saynetng/confam-ai-adapter-v1.1"
```

Cell 5 then loads the base model as usual and attaches the adapter on top of it,
before the first generation. Cell 2's banner confirms it:

```
Model:             NCAIR1/N-ATLaS
Adapter:           Saynetng/confam-ai-adapter-v1.1
Served as:         n-atlas
```

and `/v1/health` reports `"adapter_id": "Saynetng/confam-ai-adapter-v1.1"`.

Three things worth knowing:

- **`served_model_name` stays `n-atlas`.** The laptop client, the WhatsApp bot and
  anything else pointing at this server keep working unchanged. Only the weights
  behind that name changed.
- **A broken adapter stops the server instead of quietly serving the base
  model.** If the repo name is wrong, private without `HF_TOKEN`, or was trained
  against a different base, `load_model()` raises rather than falling back —
  otherwise you would be reading answers from un-fine-tuned weights while the
  config claimed otherwise, and nothing would tell you.
- **`peft` must be installed.** Cell 3 installs it when an adapter is
  configured; without `N_ATLAS_ADAPTER_ID` it is not a dependency at all, so
  nothing changes for a deployment that does not fine-tune.

To go back to the base model, unset the variable and re-run the cells.
