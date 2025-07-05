# Whisper Fine-Tuning for Arabic Speech with Tashkeel

Fine-tuning [OpenAI's Whisper-small](https://huggingface.co/openai/whisper-small) model on Arabic speech with **Tashkeel (diacritics)**. This project enables automatic transcription of Arabic audio with full vocalization, which is critical for applications like Quranic transcription, language education, and accessibility tools.

---

## 🎯 Project Highlights

- 🎤 Converts Arabic audio into **diacritized text**
- 📚 Trained on a high-quality Arabic dataset with tashkeel
- 🔬 Evaluated on Word Error Rate (WER) and loss
- 🤗 Built using Hugging Face Transformers and Whisper

---

## 🗂️ Files Overview

| File                     | Description                                  |
|--------------------------|----------------------------------------------|
| `dataset_Building.ipynb` | Prepares and preprocesses the Arabic dataset |
| `training.ipynb`         | Fine-tunes the Whisper-small model           |
| `Testing.ipynb`          | Evaluates model and visualizes predictions   |

---

## 📊 Training Results

This model is a fine-tuned version of `openai/whisper-small` on the `updated_Rev3_9643_2021` dataset.

| Metric | Value     |
|--------|-----------|
| Loss   | 0.2857    |
| WER    | 45.35%    |

> 📈 WER is calculated on a validation set containing Arabic audio with diacritics.

---

## 🔗 Hugging Face Links

- 🤗 **Model:**  
  [🔗 View the fine-tuned model on Hugging Face](https://huggingface.co/SamaYousef/whisper-small-Arabic-finetund)  
  ✅ *You can directly use this model via Hugging Face Transformers using the pipeline API:*

  ```python
  from transformers import pipeline

  pipe = pipeline("automatic-speech-recognition", model="SamaYousef/whisper-small-Arabic-finetund")
  prediction = pipe("path/to/arabic_audio.wav")
  print(prediction["text"])


- 📚 **Dataset (`updated_Rev3_9643_2021`):**  
  [🔗 View the dataset on Hugging Face](https://huggingface.co/datasets/SamaYousef/updated_Rev3_9643_2021)  
  ✅ *You can directly load and use this dataset via the 🤗 `datasets` library:*

  ```python
  from datasets import load_dataset, DatasetDict
  
  # Load the train split
  dataset = load_dataset("SamaYousef/updated_Rev3_9643_2021", split="train")
  
  # Split the dataset into train and test subsets (e.g., 80% train, 20% test)
  train_test_split = dataset.train_test_split(test_size=0.2)
  
  # Create a DatasetDict to hold the splits
  common_voice = DatasetDict({
      "train": train_test_split["train"],
      "test": train_test_split["test"]
  })
  
  print(common_voice)
---

## 📷 Sample Results

| 🎧 Audio Sample | ✅ Ground Truth | 🤖 Whisper Prediction |
|-----------------|----------------|------------------------|
| audio_001.wav   | الرَّحْمَٰنُ الرَّحِيمُ | الرَّحْمَٰنُ الرَّحِيمُ |


---

