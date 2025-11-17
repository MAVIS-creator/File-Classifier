# File Type Classifier Using Hash Features

## Objective
Detect malicious files using MD5 hash-related metadata.

## Description
This project generates hash features from files, labels them as benign/malicious, and trains a Naive Bayes classifier to detect malware.

## Algorithm
- **Naive Bayes Classifier**

## Features
- MD5 hash-based feature extraction
- Binary classification (benign/malicious)
- Comprehensive accuracy reporting
- Performance metrics visualization

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python file_classifier.py
```

## Expected Output
Accuracy report showing the model's malware classification performance including:
- Accuracy score
- Precision, Recall, F1-score
- Confusion Matrix
- Classification Report
