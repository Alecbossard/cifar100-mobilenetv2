import os
import numpy as np
import tensorflow as tf

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def prepare_cifar100_npz(data_dir="./data"):
    _ensure_dir(data_dir)
    npz_path = os.path.join(data_dir, "cifar100.npz")
    if os.path.exists(npz_path):
        return npz_path
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    np.savez_compressed(
        npz_path,
        x_train=x_train.astype(np.uint8),
        y_train=y_train.astype(np.int64),
        x_test=x_test.astype(np.uint8),
        y_test=y_test.astype(np.int64),
    )
    return npz_path

def load_cifar100_arrays(data_dir="./data"):
    path = prepare_cifar100_npz(data_dir)
    data = np.load(path)
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]
