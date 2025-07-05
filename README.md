# Whisper Fine-Tuning for Arabic Speech with Tashkeel

Fine-tuning [OpenAI's Whisper](https://github.com/openai/whisper) model on Arabic speech datasets that include **Tashkeel (diacritics)**, enabling accurate speech-to-text transcription with full vocalization.

## 🎯 Project Highlights

- ✅ Fine-tunes Whisper on Arabic audio with diacritics
- 🎤 Converts Arabic speech into text **with Tashkeel**
- 🔎 Evaluation includes comparison between ground truth and predictions
- ⚙️ Built using Hugging Face's `transformers` and `torch`

---

## 🗂️ Files Overview

| File                     | Description                                  |
|--------------------------|----------------------------------------------|
| `dataset_Building.ipynb` | Preprocesses and prepares Arabic audio data  |
| `training.ipynb`         | Fine-tunes the Whisper model                 |
| `Testing.ipynb`          | Evaluates and visualizes results             |

---

## 🔗 Hugging Face Links

- 🤗 **Fine-tuned model:**  
  [🔗 View on Hugging Face](https://huggingface.co/your-username/your-model-name)

- 📚 **Arabic Dataset with Tashkeel:**  
  [🔗 View on Hugging Face Datasets](https://huggingface.co/datasets/your-username/your-dataset-name)

---

## 📷 Sample Results

> Whisper successfully transcribes Arabic audio **with diacritics**:

| 🎧 Audio | ✅ Ground Truth | 🤖 Whisper Prediction |
|---------|----------------|------------------------|
| example.wav | الرَّحْمَٰنُ الرَّحِيمُ | الرَّحْمَٰنُ الرَّحِيمُ |

![Results Screenshot](./results/sample_output.png)

---

## 🧠 How It Works

We used the Hugging Face `pipeline` for automatic speech recognition (ASR), and fine-tuned Whisper using the following tools:

```python
from transformers import pipeline
import torch
