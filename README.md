# MonReader 📄🤖

MonReader is an AI-powered mobile document digitization system designed to deliver a fast, fully automatic, and high-quality scanning experience. Built for accessibility and efficiency, it is especially useful for visually impaired users, researchers, and anyone needing to digitize large volumes of documents with minimal effort.

## 🌟 Overview

MonReader leverages Artificial Intelligence, Deep Learning, and Computer Vision to automate the document scanning pipeline. Users simply flip pages while the system handles the rest automatically:

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

* Build a robust CNN-based model to detect page flipping from image frames
* Provide end-to-end document understanding (OCR, translation, audio)
* Improve accessibility for visually impaired users
* Streamline bulk document digitization workflows
* Automate image-to-text and text-to-speech conversion

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
 
---

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
---
## ✨ Key Features

### 📖 Intelligent Page Flip Detection

* Detects whether a page is currently flipping or stationary
* Enables automated document capture without manual interaction

### 🧾 Image-to-Text Conversion (OCR)

* Uses **PyTesseract OCR** to extract text from scanned document images
* Preserves document readability for downstream processing

### 🌍 Language Translation

* Integrates **Google Translator** to translate extracted text into English

### 🔊 Text-to-Speech Conversion

* Converts translated text into speech output
* Improves accessibility for visually impaired users

### 🤖 AI-Powered Automation

* Fully automatic document digitization workflow
* Reduces manual scanning effort and improves efficiency

---

## 🧠 Deep Learning with CNN

MonReader uses a **Convolutional Neural Network (CNN)** for the page-flip detection task.

CNNs are highly effective for image classification because they automatically learn important visual patterns such as:

* Motion blur during page flipping
* Hand movement patterns
* Page orientation changes
* Texture and edge variations

### CNN Architecture Benefits

* Automatic feature extraction from image frames
* High accuracy in binary image classification
* Robust performance under varying lighting and camera conditions
* Efficient learning from large image datasets

The CNN model was trained on labeled video frames and achieved excellent classification performance for detecting page-flipping events.

---
