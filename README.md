# File Type Classifier Using Hash Features

Lightweight MD5 hash metadata based malware vs benign file classifier.

![Classification Results (generated after first run)](classification_results.png)

> If the image is missing, run `python file_classifier.py` to generate `classification_results.png`.

---

## Objective

Detect potentially malicious files using features engineered from each file's MD5 hash and basic metadata.

## Overview

This educational project shows how simple numeric features from a file's MD5 hash can feed a Gaussian Naive Bayes model to classify files as `benign` (0) or `malicious` (1). The dataset is synthetic and not suitable for real threat detection.

## Architecture

| Stage             | Module               | Summary                                                                |
|-------------------|----------------------|------------------------------------------------------------------------|
| Feature Extraction | `hash_features.py`   | Computes MD5, first 16 hash bytes, entropy, averages, dispersion       |
| Data Preparation   | `data_handler.py`    | Generates sample files, labels them, builds feature matrix             |
| Modeling           | `file_classifier.py` | Trains Gaussian NB, evaluates, reports & saves plots                   |

## Extracted Features (20)

1. `file_size`: Raw size in bytes
2. `hash_byte_0` .. `hash_byte_15`: First 16 bytes of MD5 digest (hex → int)
3. `hash_entropy`: Approximate Shannon entropy of MD5 bytes
4. `hash_avg_byte`: Mean MD5 byte value
5. `hash_std`: Standard deviation of MD5 bytes

These are coarse, illustrative indicators only.

## Metrics Reported

- Accuracy
- Precision / Recall / F1-Score
- Confusion Matrix (text + heatmap)
- Classification report

## Quick Start

**GUI Mode (Recommended for Beginners):**

```bash
pip install -r requirements.txt
python gui_classifier.py
```

**Command-Line Mode:**

```bash
pip install -r requirements.txt
python file_classifier.py
```

## GUI Usage

Launch the graphical interface for an easy-to-use experience:

```bash
python gui_classifier.py
```

The GUI provides:

1. **Step 1: Train Model** - Click "Train Classifier" to build the model on sample data
2. **Step 2: Select Files** - Browse and select any files you want to classify
3. **Step 3: Classify** - Run classification and see results instantly
4. **View Report** - Display the accuracy visualization with confusion matrix and metrics

Features:

- 📁 File browser to select files from anywhere
- 🔍 Real-time classification results
- 📊 Visual accuracy report viewer
- 🎨 Clean, intuitive interface

## CLI Usage

Train with synthetic samples (default):

```bash
python file_classifier.py
```

Use your own dataset directories:

```bash
python file_classifier.py --benign-dir path/to/benign --malicious-dir path/to/malicious
```

Scan arbitrary files after training (any mix of paths):

```bash
python file_classifier.py --scan suspicious.exe notes.txt image.png
```

Save a text metrics report:

```bash
python file_classifier.py --report metrics.txt
```

Prevent auto-generation if directories missing:

```bash
python file_classifier.py --benign-dir b_dir --malicious-dir m_dir --no-generate
```

Show plots interactively:

```bash
python file_classifier.py --show-plots
```

## Project Structure

```text
File Classifier/
├── file_classifier.py        # Entry point & evaluation
├── hash_features.py          # MD5-based feature engineering
├── data_handler.py           # Sample data generation & splits
├── requirements.txt          # Dependencies
├── README.md                 # Documentation
├── LICENSE                   # MIT license
└── .gitignore
```

## Sample Output (Truncated)

```text
======================================================================
FILE TYPE CLASSIFIER USING HASH FEATURES
======================================================================
Training Naive Bayes Classifier...
Training completed!
...
ACCURACY: 0.8750  PRECISION: 0.8571  RECALL: 0.8571  F1: 0.8571
```

## Visualization

`classification_results.png` contains:

- Confusion Matrix heatmap
- Bar chart of metrics

## Limitations & Disclaimer

- MD5 is broken cryptographically; used here only for deterministic hashing.
- Real malware detection requires richer static + dynamic analysis and modern ML.
- Synthetic data → not indicative of production performance.

## Possible Extensions

- Add SHA256 / PE header / magic number features
- Byte n‑gram or entropy window statistics
- Alternate models (RandomForest, XGBoost, LightGBM)
- Persist model with `joblib`
- CLI args for data paths and split sizes

## Configuration Ideas (Future)

Potential flags: `--sample-dir`, `--test-size`, `--no-visualize`.

Current implemented flags: `--benign-dir`, `--malicious-dir`, `--test-size`, `--no-generate`, `--scan`, `--show-plots`, `--report`

## Contribution Guidelines

1. Fork & branch (`feature/your-idea`)
2. Keep changes focused & documented
3. Open PR describing motivation & sample output

## License

MIT License (see `LICENSE`).

## Support / Questions

Open an issue or adapt freely for learning.

---

_Educational demonstration of hash feature engineering._

