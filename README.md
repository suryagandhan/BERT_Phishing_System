<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

## Starting From Scratch — Complete Workflow When No Pre-Trained Model Is Provided

Follow this checklist in order. When you finish every step, you will have (1) a fine-uned BERT phishing model saved in `models/`, (2) the Flask web app running locally, and (3) a working REST/JSON API.

### 1  Clone the Source Code

```bash
git clone https://github.com/suryagandhan/BERT_Phishing_System.git
cd BERT_Phishing_System
```


### 2  Create and Activate a Virtual Environment

```bash
python -m venv .venv
# Mac / Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```


### 3  Install Core Dependencies

```bash
pip install --upgrade pip wheel
pip install -r requirements.txt
```

`requirements.txt` already contains:

- Flask (web server)
- Transformers + Torch (BERT)
- PyTorch Lightning (training loop)
- Pandas / scikit-learn (data wrangling \& metrics)


### 4  Prepare a Training Dataset

1. Create a folder `data/`.
2. Place two CSV files inside:
| File | Purpose | Minimum columns |
| :-- | :-- | :-- |
| `data/train.csv` | model learns here | `text`,`label` |
| `data/valid.csv` | early-stopping \& calibration | `text`,`label` |

Example row:

```csv
text,label
http://secure-paypa1.com/verify,1
https://www.microsoft.com,0
Email body: Dear user, …,1
```

*Label 0 = legitimate — Label 1 = phishing.*

### 5  Run the Training Script

```bash
python train_model.py \
    --train data/train.csv \
    --valid data/valid.csv \
    --epochs 3 \
    --batch-size 32 \
    --model-out models/phishing-bert.pt
```

What happens:

1. Tokeniser = `bert-base-uncased` (can change with `--pretrained`).
2. URLs/e-mails are truncated/padded to 128 tokens.
3. Script prints loss/F1 every epoch and saves:

```
models/
  ├─ config.json
  ├─ pytorch_model.bin
  └─ tokenizer/
```


### 6  Point the Inference Code to Your New Weights

Open `predict.py` and set:

```python
MODEL_PATH = "models/phishing-bert.pt"      # or export MODEL_PATH env-var
```

No other change required.

### 7  Launch the Flask Dev Server

```bash
python app.py
# Visit http://127.0.0.1:5000
```

Paste any URL or e-mail text—your freshly-trained model now powers the prediction.

### 8  (Optional) Docker Build Once Model Exists

1. Keep the `models/` folder locally (do **not** push big files to Git).
2. Build container that copies weights:
```bash
docker build -t phishing-detector .
docker run -p 8000:5000 phishing-detector
```


### 9  Version Control Best Practice

1. Add large checkpoints to `.gitignore`.
2. If you want to share the model:
    * push to Hugging Face Hub **or**
    * store in an S3 bucket and write `scripts/download_model.sh`.

### 10  Speed Tips for Future Retraining

| Need | Flag / Setting |
| :-- | :-- |
| Freeze BERT \& train only classifier | `--freeze-bert` |
| Larger batch via gradient-accum | `--accum 4` |
| Load balanced sampler | `--balance` |
| Calibrate probabilities | pass `--calibrate` |

### One-Line Summary

**Clone → create venv → install deps → add CSV → `train_model.py` → update `MODEL_PATH` → `python app.py`.**

