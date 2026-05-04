# MonReader 📄🤖

MonReader is an AI-powered mobile document digitization system designed to deliver a fast, fully automatic, and high-quality scanning experience. Built for accessibility and efficiency, it is especially useful for visually impaired users, researchers, and anyone needing to digitize large volumes of documents with minimal effort.

## 🌟 Overview

MonReader leverages Artificial Intelligence and Computer Vision to automate the document scanning pipeline. Users simply flip pages while the system handles the rest:

- Detects page flips using low-resolution camera preview  
- Identifies document boundaries and performs automatic cropping  
- Applies dewarping to obtain a clean, bird’s-eye view  
- Enhances contrast for better readability  
- Extracts text while preserving formatting  
- Translates extracted text into English  
- Converts text to speech for audio playback  

This project focuses specifically on the **page-flip detection component**, a critical step in enabling bulk document digitization.

---

## 🧠 Problem Statement

Given a single image frame extracted from a video, the task is to:

**Predict whether a page is being flipped or not.**

This is a binary classification problem:
- `1` → Page flipping  
- `0` → Not flipping  

---

## 🎯 Goals

- Build a robust model to detect page flipping from image frames  
- Provide end-to-end document understanding (OCR, translation, audio)  
- Improve accessibility for visually impaired users  
- Streamline bulk document digitization workflows  

---

## 📊 Model Performance

Model performance is evaluated using:

- **F1 Score (primary metric)**  
  - Balances precision and recall  
  - Particularly effective for imbalanced datasets  

### ✅ Results

The model achieves outstanding performance across all key metrics:

- **Precision:** > 0.99  
- **Recall:** > 0.99  
- **F1 Score:** > 0.99  

---

## 📁 Dataset Description

The dataset consists of smartphone-recorded videos of document pages being flipped.

### Data Collection

- Videos captured using mobile devices  
- Manually labeled as:
  - **Flipping**
  - **Not flipping**

## Repository Structure

```
MonReader/
│
├── src/                            
│   ├── eda.py
│   ├── image_preprocessor.py
│   ├── text_extractor.py
│   ├── model_trainer.py
│   ├── model_evaluator.py
|   └── main.py
|
├── requirements.txt
|
├── data/                           # Dataset for training/testing models
│   └── raw/
├── results/                        # Saved best model
│
├── figures/                        # Visualizations and plots
│                             
├── LICENSE                         # MIT License
├── README.md                       # Project documentation
└── .gitignore                      # Ignored files (temporary files, OS artifacts, Python cache)
```


