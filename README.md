# MobileNetV2 on CIFAR-100 (TensorFlow 2.10 + DirectML)

Fine-tune a pretrained **MobileNetV2** on **CIFAR-100** with a **minimal head** (GAP → Dense(100)).  
**Goal:** simple to run, few added parameters, good accuracy on a 1,000-image test subset.

---

## Requirements

- Python 3.9 or 3.10  
- `numpy==1.23.5`  
- `tensorflow-cpu==2.10`  
- `tensorflow-directml-plugin` (optional; for AMD/Windows GPU via DirectML)

Install:
```bash
pip install numpy==1.23.5 tensorflow-cpu==2.10 tensorflow-directml-plugin
```

## Project Structure : 

cifar100_mobilenetv2/

├─ data/                # local cache (ignored by git) — created at first run

├─ models/              # saved models (ignored by git)

├─ data_loader.py       # download/cache CIFAR-100 once (npz) and load arrays

├─ train.py             # build model, train, evaluate, save

## Dataset (Not Included) :

- The dataset is not versioned.
- On first run, data_loader.py downloads CIFAR-100 via Keras and caches it as ./data/cifar100.npz.
- Subsequent runs reuse this local file (no re-download).

## How to Run : 

From the project root:
```bash
python train.py
```
What the script does:
1. Creates ./data/cifar100.npz if missing.
2. Resizes images to 224×224 and applies preprocess_input (MobileNetV2, [-1,1]).
3. Trains only the head (backbone frozen) for a few epochs.
4. Evaluates on:
  - a 1,000-image test subset (challenge target metric)
  - the full 10,000-image test set (informational)
5. Saves the model to ./models/mobilenetv2_cifar100.h5.


## Troubleshooting :

- Push rejected (>100 MB): ensure data/ and models/ are ignored .
- Import errors: confirm numpy==1.23.5 and tensorflow-cpu==2.10.
- GPU not used: install tensorflow-directml-plugin and run on Windows with a compatible GPU; otherwise training will run on CPU.


## Acknowledgments :

- TensorFlow/Keras MobileNetV2 pretrained on ImageNet
- CIFAR-100 dataset 
