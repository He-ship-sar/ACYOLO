
# 📦 AC-YOLO — Lightweight SAR Ship Detection Model

Images Based on YOLO11

## 📑 Project Description

This repository contains the full implementation of the model proposed in the paper "AC-YOLO: A Lightweight Ship Detection Model for SAR". The model is a lightweight architecture based on YOLO, enhanced with attention mechanisms for efficient and accurate ship detection in high-resolution Synthetic Aperture Radar (SAR) imagery.

The codebase includes scripts for training, validation, testing, configuration files, example data, and comprehensive documentation for reproducing the results presented in the paper.

## 🧠 Key Features

- Lightweight YOLO-based architecture
- Supports detection in SAR images
- Training (`train.py`), validation (`val.py`), and testing (`test.py`) scripts included
- Modular structure for easy extension
- Docker environment and unit tests provided

## 🛠️ Installation Requirements

Recommended Python version: 3.12 or above.

To install dependencies:

```bash
pip install -r requirements.txt
```

Or use a Conda environment (optional):

```bash
conda create -n acyolo python=3.12
conda activate acyolo
pip install -r requirements.txt
```

## 📂 Directory Structure

```bash
.
├── docker/              # Docker deployment files
├── docs/                # Project documentation (for mkdocs)
├── examples/            # Sample data and usage examples
├── tests/               # Unit test scripts
├── ultralytics/         # YOLO model implementation
├── train.py             # Training script
├── val.py               # Validation script
├── test.py              # Testing script
├── LICENSE              # Open-source license
├── CONTRIBUTING.md      # Contributing guidelines
├── CITATION.cff         # Citation file
├── mkdocs.yml           # MkDocs configuration
├── pyproject.toml       # Project dependency config (Poetry compatible)
└── README.md            # Project overview (this file)
```

## 🚀 Quick Start

1. Clone the repository:

```bash
git clone https://github.com/He-ship-sar/ACYOLO.git
cd ACYOLO
```

2. Train the model:

```bash
python train.py 
```

3. Validate the model:

```bash
python val.py 
```

4. Test the model:

```bash
python test.py 
```

## 🧪 Example Data

Sample SAR images and corresponding annotations can be found in the `examples/` directory to test model functionality.

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).


