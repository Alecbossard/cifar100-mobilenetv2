# CIFAR-100 MobileNetV2 (Two-Stage Fine-Tuning)

A robust, production-oriented fine-tuning pipeline for **CIFAR-100** using **MobileNetV2**.  
This project implements a **two-stage training strategy** (transfer learning + fine-tuning) to maximize accuracy on a resource-constrained architecture, achieving **> 71% accuracy** on the test set.

---

## 🚀 Features

- **Two-Stage Training Strategy**
  1. **Stage 1 – Head Training**  
     Train only the classification head while the backbone is frozen to stabilize weights.
  2. **Stage 2 – Fine-Tuning**  
     Unfreeze the top layers of MobileNetV2 (`FINE_TUNE_AT`) with a lower learning rate for better adaptation.

- **Advanced Regularization**
  - **Label smoothing** to reduce overfitting by softening hard one-hot targets.
  - **Data augmentation** with random flip, rotation, translation, and zoom.
  - **Cosine decay** learning rate scheduling for smoother convergence.

- **Performance Optimization**
  - **Mixed precision** (`mixed_float16`) automatically enabled on compatible GPUs (e.g., NVIDIA T4) for faster training and lower memory usage.
  - **Efficient data pipeline** using `tf.data.AUTOTUNE` for parallel mapping and prefetching.

---

## 🛠️ Requirements

- Python 3.9+
- TensorFlow 2.10+
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```text
.
├── data/               # Dataset cache (created on first run)
├── models/             # Saved model artifacts (.h5)
├── data_loader.py      # Utilities for downloading/loading CIFAR-100
├── train.py            # Main training script (2-stage pipeline)
└── requirements.txt    # Project dependencies
```

---

## ⚡ Usage

Run the training script:

```bash
python train.py
```

On the first run, the script will automatically download and cache the CIFAR-100 dataset in the `./data` folder.

---

## ⚙️ Configuration

You can customize hyperparameters via environment variables or by editing `train.py` directly.

| Parameter       | Default | Description                               |
|-----------------|---------|-------------------------------------------|
| `IMG_SIZE`      | 128     | Input image size (128×128)                |
| `BATCH`         | 64      | Batch size                                |
| `EPOCHS_STAGE1` | 15      | Epochs for head training (frozen backbone)|
| `EPOCHS_STAGE2` | 10      | Epochs for fine-tuning (unfrozen backbone)|

Example with custom settings:

```bash
BATCH=128 EPOCHS_STAGE2=20 python train.py
```

(You can also use your shell's syntax for setting environment variables on Windows / PowerShell.)

---

## 📊 Results

The model is evaluated on:
- A fixed **1000-image test subset** (proxy for the challenge target).
- The **full CIFAR-100 test set**.

- **Target accuracy**: > 70% on CIFAR-100.
- **Inference speed**: ~15 ms per image (on T4 GPU / edge-equivalent hardware).

---

## 📎 Notes

- This repository is designed as a clean baseline for CIFAR-100 transfer learning with MobileNetV2.
- You can plug this into experiment tracking tools (e.g., Weights & Biases, TensorBoard) by adding callbacks in `train.py`.
