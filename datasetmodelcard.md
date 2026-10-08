# Confam Training Dataset

## Dataset Overview

The Confam Training Dataset is a curated collection of 516 conversational examples designed for training the Confam AI Adapter v1.1, enhancing the NAFDAC product verification capabilities of the Nigerian-focused conversational AI assistant.

## Dataset Purpose

This dataset serves as an improvement over previous iterations, providing more accurate and reliable training data for NAFDAC product verification. The examples focus on ConfamAI helping Nigerian consumers verify whether products are registered with NAFDAC (National Agency for Food and Drug Administration and Control) before purchase.

## Dataset Structure

Each training example in this JSONL dataset follows a conversation format:

- **system**: Defines ConfamAI's identity and rules for verification
- **user**: User queries about product authenticity or verification
- **assistant**: ConfamAI's response using verification data

Some entries include an additional system message with NAFDAC verification data in the format:
```
[nafdac] Verified greenbook data for this question. Report these values to the user as fact
Product: Product Name
NRN: Registration Number
Applicant: Company Name
VERDICT: AUTHENTIC/FLAGGED/NOT REGISTERED
```

## Content Details

**Number of Examples**: 516 total conversations

**Average Message Length**:
- System: 3,596 characters (prompt definitions)
- User: 2-12 characters (brief queries)
- Assistant: 98-106 characters (concise responses)

**Supported Languages**: Multilingual Nigerian contexts (English, Hausa, Igbo, Yoruba)

**Training Files**:
- `confam_training_dataset.jsonl`: Main training dataset

## Key Features

1. **Standardized Format**: Consistent message roles and response structure
2. **Factual Verification**: Based on real NAFDAC verification data
3. **Safety Focus**: Clear instructions to avoid making unverified claims
4. **Multi-language**: Covers Nigerian languages and contexts
5. **Quality Control**: Each example follows specific data rules for accuracy

## Example Format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are ConfamAI, the official product verification assistant for the CONFAM powered by saynet.ng. ... (full system prompt)"
    },
    {
      "role": "user",
      "content": "Is OMO Detergent 20 authentic?"
    },
    {
      "role": "assistant",
      "content": "FLAGGED — OMO Detergent 20 is flagged by NAFDAC. Do not buy or use this product. You can report it to NAFDAC."
    }
  ]
}
```

## Improvements over v1

This dataset represents an improvement over previous versions with:

- **More accurate training examples**: Refactored and quality-checked
- **Better NAFDAC verification data**: Enhanced factual accuracy
- **Improved instruction following**: Clearer adherence to system rules
- **Enhanced safety**: Stronger emphasis on verification requirements
- **Quality consistency**: Standardized response formats

## Usage

### For Training

This dataset is designed for fine-tuning the Confam AI Adapter v1.1 or similar conversational AI models. It's particularly suitable for:

1. **NAFDAC Product Verification**: Building verification assistants
2. **Nigerian Language Processing**: Multi-lingual conversational AI
3. **Safety-Focused AI**: Models that require strict factual verification
4. **Instruction Following**: Training on complex system prompt rules

### Loading the Dataset

```python
import pandas as pd

# Load the dataset
df = pd.read_json('confam_training_dataset.jsonl', lines=True)

# Each example has a 'messages' field with conversation turns
examples = df['messages'].tolist()
```

### Expected Tasks

- Causal language modeling
- Conversational AI training
- Product verification assistants
- NAFDAC compliance checking
- Multi-language AI systems

## Technical Specifications

- **Format**: JSONL (JSON Lines)
- **Files**: 1 training file
- **Total Size**: ~2.08 MB
- **Version**: Training dataset for Confam AI Adapter v1.1 improvements
- **Dependencies**: Transformers/PEFT framework for model training

## Intended Use

This dataset is intended for:

1. **Research and Development**: Training conversational AI for product verification
2. **Academic Study**: Analyzing NAFDAC verification patterns
3. **Commercial Applications**: Building Nigerian-specific AI assistants
4. **Safety-Focused Systems**: Models that require accurate factual information
5. **Multi-language AI**: Supporting Nigerian languages

## Limitations and Considerations

1. **Language Bias**: Primarily Nigerian English and languages
2. **Domain Specificity**: Focused specifically on NAFDAC verification
3. **Data Quality**: While improved, still requires verification
4. **Cultural Context**: Designed for Nigerian market context

## Citation

If you use this dataset in your research or project:

```bibtex
@misc{saynetng_confam_training_dataset_2026,
  title={Confam Training Dataset},
  author={Saynetng},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/Saynetng/confam_training_dataset},
  note={Training dataset for Confam AI Adapter v1.1 improvements}
}
```

## Contact and Links

- **Dataset Hub**: [Saynetng/confam_training_dataset](https://huggingface.co/datasets/Saynetng/confam_training_dataset)
- **Confam AI**: [https://confamai.saynet.ng/](https://confamai.saynet.ng/)
- **Saynet**: [https://saynet.ng](https://saynet.ng)

## Dataset Card Authors

**Saynetng / Confam AI**

For more information about the Confam AI Adapter v1.1, please refer to the model card at: [Saynetng/confam-ai-adapter-v1.1](https://huggingface.co/Saynetng/confam-ai-adapter-v1.1)

## Related Work

This dataset improves upon:

1. **Confam AI Adapter v1**: Previous adapter version with less accurate training data
2. **N-ATLaS**: Base multilingual African language model
3. **NAFDAC Database**: Product verification data source

## Changes from Previous Versions

This dataset represents an improvement over the initial training datasets, with:

1. **Enhanced Data Quality**: More carefully curated and verified examples
2. **Improved Verification**: Better NAFDAC data integration
3. **Clearer Instructions**: More specific system prompt rules
4. **Better Safety**: Stronger guidance on verification requirements

## Feedback and Contributions

We welcome feedback on:

1. Dataset improvements
2. Additional verification scenarios
3. Multi-language expansions
4. Quality assessments
5. Bug reports and issues

Please use the dataset discussions on Hugging Face for feedback and contributions.