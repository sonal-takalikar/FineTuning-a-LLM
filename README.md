# FFineTuning-a-LLM

Fine-tuning Flan-T5 for dialogue summarization using the DialogSum dataset.

## Overview

This project fine-tunes Google's Flan-T5 model to generate concise summaries of conversations. Achieved 27.5% improvement in ROUGE-1 scores over baseline through systematic hyperparameter optimization.

## Dataset

**DialogSum** - 13,460 real-world dialogue-summary pairs
- Training: 10,460 examples
- Validation: 500 examples  
- Test: 2,500 examples

## Model

**Flan-T5-base** (250M parameters)
- Encoder-decoder architecture
- Instruction-tuned for natural language tasks
- Input format: "Summarize this conversation:\n{dialogue}"

## Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|---------|---------|
| Baseline | 0.2882 | 0.0857 | 0.2154 |
| Fine-tuned | 0.3673 | 0.1194 | 0.2788 |
| **Improvement** | **+27.5%** | **+39.3%** | **+29.4%** |

## Setup

```bash
# Clone repository
git clone <<URL>>
cd FineTuning-a-LLM

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```python
python train.py --config config2
```

### Inference
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained('./fine_tuned_model')
tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')

dialogue = "Your conversation here..."
input_text = f"Summarize this conversation:\n{dialogue}"
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_length=128)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Hyperparameter Configurations

**Config 1 (Default)**
- Learning rate: 1e-3
- Batch size: 8
- Weight decay: 0.01

**Config 2 (Best)** ⭐
- Learning rate: 5e-4
- Batch size: 16
- Weight decay: 0.01

**Config 3**
- Learning rate: 2e-3
- Batch size: 8
- Weight decay: 0.001

## Error Analysis

Identified 6 error categories across 20 test examples:
- Missing key information (35%)
- Length issues (25%)
- Context misunderstanding (20%)
- Hallucination (10%)
- Entity confusion (5%)
- Focus problems (5%)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Datasets
- ROUGE-score
- CUDA-capable GPU (recommended)
