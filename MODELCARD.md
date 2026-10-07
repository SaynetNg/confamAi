---
library_name: transformers
tags:
- n-atlas
- nitda
- saynetng
language:
- ha
- yo
- ig
- en
base_model:
- NCAIR1/N-ATLaS
---

# Confam AI Adapter v1.1

**Confam AI Adapter v1.1** is an improved version of the Confam AI fine-tuning adapter, built on top of [NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS), a multilingual African language model based on Llama 3 8B.

The adapter is developed for **Confam AI**, a Nigerian-focused AI platform designed to provide helpful, culturally relevant and multilingual AI experiences.

🌐 **Confam AI**: https://confamai.saynet.ng/

🤗 **Model v1.1**: https://huggingface.co/Saynetng/confam-ai-adapter-v1.1

🤗 **Model v1**: https://huggingface.co/Saynetng/confam-ai-adapter

🤗 **Base Model**: https://huggingface.co/NCAIR1/N-ATLaS

**Version**: v1.1  
**Status**: Improved release with refined training data

---

## Model Details

| Property | Details |
|----------|--------|
| **Model name** | Confam AI Adapter v1.1 |
| **Version** | v1.1 |
| **Organization** | Saynetng |
| **Model type** | Fine-tuning adapter |
| **Base model** | NCAIR1/N-ATLaS |
| **Base architecture** | Llama 3 8B |
| **Task** | Causal Language Modeling / Conversational AI |
| **Languages** | English, Hausa, Igbo, Yoruba |
| **Platform** | Confam AI |
| **Format** | Safetensors |
| **Framework** | Hugging Face Transformers / PEFT |
| **Release** | Version 1.1 / Improved release |

---

## About Confam AI

**Confam AI Adapter v1.1** is an improved version of the Confam AI adapter with more accurate and refined training data.

It extends the capabilities of the N-ATLaS multilingual African language model and is designed to provide a foundation for Confam AI's conversational AI experiences, with particular focus on NAFDAC product verification for Nigerian consumers.

### What is Confam AI?

**Confam AI** is the official product verification assistant for **CONFAM**, helping Nigerians verify whether a product is registered with **NAFDAC** (National Agency for Food and Drug Administration and Control) before they buy or use it.

The platform covers:
- 🍽️ Food Products (packaged foods, beverages, water, supplements)
- 💊 Drugs & Pharmaceuticals (medications, vaccines, herbal remedies)
- 🧴 Cosmetics & Personal Care
- 🏥 Medical Devices
- 🧪 Chemicals & Agro-chemicals
- 🐾 Veterinary Products
- ⚠️ Controlled Substances

**Confam AI is powered by [Saynet.ng](https://saynet.ng)** — Building innovative AI solutions for Africa.

Confam AI focuses on building useful AI experiences for Nigerian and African users, with support for English and major Nigerian languages (Hausa, Igbo, Yoruba).

🌐 **Visit Confam AI**: https://confamai.saynet.ng/

🌐 **Powered by Saynet**: https://saynet.ng

---

## Version History

### v1.1 — Improved Release

**Confam AI Adapter v1.1** is an improved version with refined and more accurate training data compared to v1.

**Key Improvements over v1:**

- **More accurate training dataset**: Refined 643-record dataset with improved data quality
- **Better factual accuracy**: Enhanced training data ensures more reliable responses
- **Improved instruction following**: Better adherence to system prompts and user instructions
- **Enhanced NAFDAC product verification**: More accurate handling of product verification queries
- **Refined conversation quality**: Improved response coherence and relevance

This release maintains the same training methodology as v1 but benefits from significantly improved training data quality, resulting in better overall model performance.

### v1 — Initial Release

**Confam AI Adapter v1** was the first publicly released version of the Confam AI adapter, establishing the initial model foundation.

---

## Base Model

This adapter is fine-tuned from:

**[NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS)**

N-ATLaS is a multilingual African language model based on Llama 3 8B, with support for English, Hausa, Igbo and Yoruba.

**N-ATLaS is an initiative of the Federal Ministry of Communications, Innovation and Digital Economy, and powered by Awarri Technologies.**

The Confam AI Adapter v1.1 builds on these capabilities for the Confam AI use case.

---

## Supported Languages

| Language | Support |
|----------|--------|
| 🇬🇧 English | Supported |
| 🇳🇬 Hausa | Supported |
| 🇳🇬 Igbo | Supported |
| 🇳🇬 Yoruba | Supported |

Performance may vary between languages and tasks.

---

## Intended Use

**Confam AI Adapter v1.1** is intended for:

- Conversational AI
- Nigerian-focused AI applications
- African language applications
- Multilingual assistants
- NAFDAC product verification
- Educational applications
- Language technology research
- Translation and language transformation
- AI-powered applications
- Digital inclusion
- English, Hausa, Igbo and Yoruba interactions

---

## Training & Quantization

**Confam AI Adapter v1.1** was fine-tuned from NCAIR1/N-ATLaS using parameter-efficient LoRA / PEFT fine-tuning with **4-bit NF4 quantization**.

### Training Configuration

| Parameter | Configuration |
|-----------|---------------|
| **Base model** | NCAIR1/N-ATLaS |
| **Fine-tuning method** | LoRA / PEFT |
| **Quantization** | 4-bit NF4 |
| **Double quantization** | Enabled |
| **4-bit loading** | `load_in_4bit=True` |
| **Quantization type** | `bnb_4bit_quant_type="nf4"` |
| **Double quantization** | `bnb_4bit_use_double_quant=True` |
| **Compute dtype** | `float16` |
| **4-bit compute dtype** | `bnb_4bit_compute_dtype=torch.float16` |
| **Training precision** | FP16 (`fp16=True`) |
| **LoRA rank (r)** | 16 |
| **LoRA alpha** | 32 |
| **LoRA dropout** | 0.05 |
| **Learning rate** | 2e-4 |
| **LR scheduler** | Cosine |
| **Warmup steps** | 10 |
| **Optimizer** | `paged_adamw_8bit` |
| **Configured epochs** | 1 |
| **Maximum training steps** | 200 |
| **Effective batch size** | 8 |
| **Maximum sequence length** | 2048 tokens |
| **Training records** | 643 |
| **Estimated dataset passes** | ~2.5 passes |
| **Training hardware** | NVIDIA T4 |
| **BF16** | Not used |

### Quantization

The base N-ATLaS model was loaded using **4-bit NormalFloat (NF4) quantization** with double quantization enabled.

```python
load_in_4bit=True
bnb_4bit_quant_type="nf4"
bnb_4bit_use_double_quant=True
bnb_4bit_compute_dtype=torch.float16
```

The training run used **float16** computation because the NVIDIA T4 hardware used for training does not provide the required BF16 support for this configuration.

### LoRA Configuration

The adapter uses **Low-Rank Adaptation (LoRA)** to fine-tune a relatively small number of trainable parameters while keeping the majority of the underlying N-ATLaS model frozen.

```python
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
```

| LoRA Parameter | Value |
|----------------|-------|
| **Rank (r)** | 16 |
| **Alpha** | 32 |
| **Dropout** | 0.05 |

### Optimization

The model was trained with a learning rate of **2 × 10⁻⁴** using a **cosine learning-rate scheduler** and **10 warmup steps**.

```
Learning rate:  2e-4
Scheduler:      cosine
Warmup steps:   10
Optimizer:      paged_adamw_8bit
```

The `paged_adamw_8bit` optimizer was used to reduce optimizer memory requirements during fine-tuning.

### Training Duration

Although `num_train_epochs=1` was configured, the training run was step-limited to **200 steps**.

With an effective batch size of 8 and 643 training records, the run corresponds to approximately **2.5 passes** over the training records.

| Training Parameter | Value |
|--------------------|-------|
| **Training records** | 643 |
| **Effective batch size** | 8 |
| **Maximum steps** | 200 |
| **Configured epochs** | 1 |
| **Approx. dataset passes** | ~2.5 |

### Sequence Length

The maximum sequence length was set to **2048 tokens**.

```python
N_ATLAS_MAX_SEQ = 2048
```

The training data was checked to ensure that no record exceeded the configured maximum sequence length.

### Hardware

Training was performed using an **NVIDIA T4 GPU**.

Because the T4 was used for the training run, **FP16** was used for computation rather than BF16.

```
Compute dtype:       float16
Training precision:  FP16
GPU:                 NVIDIA T4
```

The combination of 4-bit NF4 quantization, LoRA, and `paged_adamw_8bit` was used to make parameter-efficient fine-tuning feasible within the available GPU memory.

### Training Method Summary

```
N-ATLaS (Llama 3 8B)
        │
        ▼
4-bit NF4 Quantization
        │
        ├── Double Quantization
        └── FP16 Compute
        │
        ▼
LoRA Fine-Tuning
        │
        ├── Rank: 16
        ├── Alpha: 32
        └── Dropout: 0.05
        │
        ▼
Cosine LR Scheduler
        │
        ├── Learning Rate: 2e-4
        └── Warmup: 10 steps
        │
        ▼
paged_adamw_8bit
        │
        ▼
200 Training Steps
        │
        ▼
Confam AI Adapter v1.1
```

---

## Using the Adapter

Because this repository contains an **adapter**, it should be loaded together with the compatible N-ATLaS base model.

### Installation

```bash
pip install -U transformers peft accelerate torch
```

### Loading the Adapter

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "NCAIR1/N-ATLaS"
ADAPTER_MODEL = "Saynetng/confam-ai-adapter-v1.1"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load Confam AI adapter v1.1
model = PeftModel.from_pretrained(
    model,
    ADAPTER_MODEL
)

messages = [
    {
        "role": "system",
        "content": "You are Confam AI, a helpful Nigerian AI assistant."
    },
    {
        "role": "user",
        "content": "Hello, how are you?"
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(response)
```

---

## Model Architecture

```
Llama 3 8B
     │
     ▼
N-ATLaS
NCAIR1/N-ATLaS
     │
     │ Fine-tuning / LoRA Adapter
     ▼
Confam AI Adapter v1.1
Saynetng/confam-ai-adapter-v1.1
     │
     ▼
Confam AI
confamai.saynet.ng
```

The **Confam AI Adapter** is not a standalone base model. It is designed to work with the compatible N-ATLaS base model.

---

## Evaluation

Independent benchmark results for **Confam AI Adapter v1.1** are not currently published.

Users are encouraged to evaluate v1.1 across:

- English
- Hausa
- Igbo
- Yoruba
- Nigerian cultural context
- Instruction following
- Factual accuracy
- Helpfulness
- Safety
- Hallucination
- Conversation quality
- NAFDAC product verification accuracy

Future versions may include expanded benchmarks and evaluation results.

---

## Improvements over v1

**v1.1** provides the following improvements over the initial v1 release:

1. **Refined Training Data**: More accurate and carefully curated 643-record training dataset
2. **Better Factual Accuracy**: Improved data quality leads to more reliable factual responses
3. **Enhanced Product Verification**: Better handling of NAFDAC product verification queries
4. **Improved Instruction Following**: Better adherence to system prompts and user intent
5. **Higher Quality Responses**: More coherent and contextually appropriate responses

Users upgrading from v1 to v1.1 should observe improved performance across all supported use cases, particularly in product verification scenarios.

---

## Limitations

While v1.1 represents an improvement over v1, it should still be used with appropriate caution.

Potential limitations include:

- Hallucinations
- Incorrect factual information
- Uneven performance across languages
- Bias
- Cultural misunderstandings
- Limited domain-specific knowledge
- Sensitivity to prompting
- Inconsistent responses
- Performance limitations outside the training distribution

Users should independently verify important information generated by the model.

---

## Responsible Use

**Confam AI Adapter v1.1** should be used responsibly.

Developers should:

- Verify important information.
- Avoid using model outputs as a substitute for professional advice.
- Test the model extensively before production deployment.
- Evaluate performance across relevant Nigerian languages.
- Monitor for harmful, biased or culturally inappropriate outputs.
- Maintain human oversight for high-impact applications.
- Never rely solely on the model for critical medical or safety decisions.

---

## License and Attribution

**Confam AI Adapter v1.1** is a derivative work based on NCAIR1/N-ATLaS.

Users should review and comply with the licensing and attribution requirements of the N-ATLaS model before using, modifying or redistributing this adapter.

For the complete terms, please refer to:

**N-ATLaS**: https://huggingface.co/NCAIR1/N-ATLaS

---

## Links

### 🌐 Confam AI
https://confamai.saynet.ng/

### 🤗 Confam AI Adapter v1.1
https://huggingface.co/Saynetng/confam-ai-adapter-v1.1

### 🤗 Confam AI Adapter v1
https://huggingface.co/Saynetng/confam-ai-adapter

### 🤗 N-ATLaS Base Model
https://huggingface.co/NCAIR1/N-ATLaS

---

## Citation

If you use **Confam AI Adapter v1.1** in your project or research, please cite:

```bibtex
@misc{confamai_adapter_v1_1,
  title={Confam AI Adapter v1.1},
  author={Saynetng},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/Saynetng/confam-ai-adapter-v1.1},
  note={Version 1.1 fine-tuning adapter based on N-ATLaS with improved training data}
}
```

For the underlying N-ATLaS model, please also refer to its official model card and citation.

---

## Acknowledgements

Confam AI acknowledges the developers and contributors behind N-ATLaS and the broader Nigerian and African language AI ecosystem.

The Confam AI Adapter builds upon the work of the N-ATLaS project and its contributors.

**Special thanks to:**
- The Federal Ministry of Communications, Innovation and Digital Economy for the N-ATLaS initiative
- Awarri Technologies for powering N-ATLaS
- The N-ATLaS development team and contributors
- The broader African language AI community

---

## Model Card Authors

**Saynetng / Confam AI**

🌐 **Confam AI**: https://confamai.saynet.ng/

🌐 **Saynet.ng**: https://saynet.ng

🤗 **Hugging Face**: https://huggingface.co/Saynetng

---

**Confam AI Adapter — Version 1.1**