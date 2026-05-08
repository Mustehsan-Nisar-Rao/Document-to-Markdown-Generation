# VLM QLoRA Fine-Tuning — Document to Markdown Generation

**Course:** AI4009 — Generative AI | Spring 2026  
**Assignment:** 5  
**Institution:** National University of Computer and Emerging Sciences (FAST-NUCES)

---

## Overview

This project fine-tunes a Vision Language Model (VLM) using QLoRA (Quantized Low-Rank Adaptation) to convert document images into structured Markdown. The model takes a page image as input and outputs a faithful Markdown representation preserving headings, equations, tables, lists, and captions.

---

## Model

- **Base Model:** [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- **Fine-Tuning Method:** QLoRA (4-bit quantization + LoRA adapters)
- **LoRA Rank:** 16
- **LoRA Alpha:** 32
- **Target Modules:** q_proj, k_proj, v_proj, o_proj

---

## Dataset

- **Source:** [Nougat Training Dataset Example](https://www.kaggle.com/datasets/zphilip/nougat-training-dataset-example)
- **Subset Used:** 1500 samples (dataset allows subset selection)
- **Split:** 80% training (1200) / 20% validation (300)
- **Format:** ChatML — each sample contains image, instruction prompt, and target markdown

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 2 |
| Batch Size | 1 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 4 |
| Learning Rate | 1.5e-4 |
| Optimizer | AdamW |
| Scheduler | Cosine Annealing |
| Image Resolution | 512px |
| Platform | Kaggle T4 x2 |

---

## Results

### Training vs Validation Loss

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 1 | ~4.2 | ~4.8 |
| 2 | ~3.1 | ~3.9 |

### Zero-Shot vs Fine-Tuned

| Aspect | Zero-Shot | Fine-Tuned |
|--------|-----------|------------|
| Markdown headings | Missing | Correctly generated |
| Bold/italic formatting | Ignored | Preserved |
| Table structure | Broken | Better preserved |
| Overall structure | Weak | Significantly improved |

---

## Project Structure

```
├── app.py                  # Streamlit/Gradio deployment app
├── requirements.txt        # Dependencies
├── weights/                # Fine-tuned LoRA adapter weights
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── processor_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── chat_template.jinja
└── AI_ASS05_VLM_QLoRA.ipynb   # Full training notebook
```

---

## Setup & Installation

```bash
pip install transformers>=4.45.0 peft>=0.13.0 bitsandbytes>=0.43.0
pip install accelerate>=0.34.0 qwen-vl-utils gradio Pillow torch
```

---

## Running the App

```bash
streamlit run app.py
```

Upload any document image and the model will generate structured Markdown output.

---

## Key Findings

- Fine-tuning with only 1500 samples and 2 epochs produced visible improvement over zero-shot
- QLoRA reduced memory usage significantly — full fine-tuning would require 40GB+ VRAM, QLoRA runs on a single T4 (15GB)
- The model learned to generate markdown headings and formatting that the zero-shot baseline consistently missed
- Corrupt images in the dataset needed to be filtered before training

---

## Dependencies

```
transformers>=4.45.0
peft>=0.13.0
bitsandbytes>=0.43.0
accelerate>=0.34.0
torch
torchvision
Pillow
qwen-vl-utils
gradio
streamlit
matplotlib
tqdm
```

---

## References

- [Qwen2-VL Paper](https://arxiv.org/abs/2409.12191)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Nougat Paper](https://arxiv.org/abs/2308.13418)
- [PEFT Library](https://github.com/huggingface/peft)
