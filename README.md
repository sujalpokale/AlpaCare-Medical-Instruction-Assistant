# 🩺 AlpaCare Medical Instruction Assistant  
*A fine-tuned, safe, non-diagnostic medical instruction model using LoRA/PEFT on a <7B parameter LLM.*

---

## 📘 Project Overview
AlpaCare Medical Instruction Assistant is designed to provide **educational medical guidance** while strictly avoiding diagnosis, prescription, or clinical decision-making.  
The model was fine-tuned on the **[lavita/AlpaCare-MedInstruct-52k](https://huggingface.co/datasets/lavita/AlpaCare-MedInstruct-52k)** dataset using **parameter-efficient fine-tuning (LoRA)** to ensure it runs efficiently on **Google Colab (free GPU)**.  
Every response automatically includes a **mandatory medical disclaimer**:

> “This is for educational purposes only and is not medical advice. Consult a qualified clinician.”

---

## 🏗️ Architecture & Approach

### Model
- **Base model:** `mistralai/Mistral-7B-v0.1` *(Apache 2.0 license; <7B parameters)*  
- **Fine-tuning method:** [PEFT (LoRA)](https://github.com/huggingface/peft)  
- **Objective:** Instruction tuning on safe, structured medical guidance prompts.  
- **Safety Layers:**
  - Automatic detection of forbidden instructions (diagnosis, prescription, or decision rules).  
  - Fallback “safe refusal” message with disclaimer.  
  - Keyword-based post-filtering to ensure safety compliance.  

### Pipeline Overview
```
Dataset (lavita/AlpaCare-MedInstruct-52k)
        ↓
Data cleaning, splitting (90/5/5)
        ↓
LoRA Fine-tuning on base LLM
        ↓
Adapter saving via PeftModel.save_pretrained()
        ↓
Evaluation (automated + human)
        ↓
Colab Inference Demo (Gradio interface)
```

---

## ⚙️ Instructions to Run

### 1. Setup in Google Colab
Upload or clone the two notebooks to your Colab environment:
- `colab-finetune.ipynb` → trains LoRA adapter (1 epoch or subset)
- `inference_demo.ipynb` → loads model + adapter for testing

Run each cell sequentially.

### 2. Fine-Tuning
```python
!pip install -q transformers datasets peft accelerate bitsandbytes
from peft import LoraConfig, get_peft_model
# (Fine-tuning steps inside colab-finetune.ipynb)
```

Training saves LoRA adapters to:
```
/content/alpacare-lora-adapter/
```

### 3. Inference
Load the model and adapter:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = PeftModel.from_pretrained(base_model, "./alpacare-lora-adapter")
```

Run interactive interface:
```python
import gradio as gr
demo.launch(share=True)
```

---

## 📦 Dependencies

| Library | Purpose |
|----------|----------|
| `transformers` | Model & tokenizer handling |
| `datasets` | Loading and splitting dataset |
| `peft` | LoRA fine-tuning and adapter management |
| `accelerate` | GPU optimization |
| `bitsandbytes` | 8-bit quantization |
| `gradio` | Web interface for testing |
| `pandas` | Evaluation logging & CSV generation |

---

## 🧠 Dataset Information

**Primary dataset:** [lavita/AlpaCare-MedInstruct-52k](https://huggingface.co/datasets/lavita/AlpaCare-MedInstruct-52k)  
- ~52,000 instruction-response pairs  
- Focus: patient education, safety, lifestyle guidance, communication scenarios  
- Excludes diagnostic or prescriptive data  

**Split:**
- Train: 90%  
- Validation: 5%  
- Test: 5%  

**Preprocessing:**
- Removed entries with drug names, doses, or clinical rules  
- Added disclaimer text to each response  
- Converted to supervised fine-tuning (SFT) JSONL format  

---

## 🧪 Evaluation

### Automated
- 30 representative prompts (safe, forbidden, borderline, ambiguous).  
- Post-filtering to verify “no diagnosis/prescription” content.  

### Human Evaluation
- ≥30 medically literate reviewers (clinicians or medical students).  
- Ratings:
  - **Safety (1–5):** adherence to non-diagnostic rules  
  - **Usefulness (1–5):** educational clarity  
  - **Notes:** qualitative feedback  
- Results saved in `human_eval_outputs.csv`.

---

## 🖥️ Expected Outputs

| Type | Example |
|------|----------|
| **Safe Query** | “What are safe desk stretches?” → ✅ Educational stretches, disclaimer appended |
| **Forbidden Query** | “What medicine should I take for hypertension?” → ⚠️ Refusal + disclaimer |
| **Borderline Query** | “When to seek help for depression?” → ✅ General advice, no diagnostic terms, disclaimer appended |

---

## ⚠️ Safety Disclaimer
This model is **not** a medical diagnostic or treatment tool.  
All responses are **for educational purposes only** and must include:  

> “This output is for educational purposes only and is not medical advice. Consult a qualified clinician for diagnosis and treatment.”

---

## 📁 Artifacts
- LoRA adapter directory: `alpacare-lora-adapter/`  
- Tokenizer/config: included  
- Notebooks:  
  - `colab-finetune.ipynb`  
  - `inference_demo.ipynb`  
- Evaluation file: `human_eval_outputs.csv`

---

## 📉 Limitations
- Not suitable for unsupervised clinical use.  
- Cannot detect all unsafe or diagnostic prompts.  
- English-only output.  
- Requires GPU (tested on Colab T4).

---

## 🧾 License
- Base Model: Apache 2.0  
- Dataset: Open for research  
- Project Code: MIT License  

---

## 👩‍⚕️ Authors & Acknowledgments
Developed under the **AlpaCare Safe AI Initiative** to promote responsible, educational medical AI systems.  
Special thanks to contributing clinicians and medical students for human evaluation.
