import os
import numpy as np
import tensorflow as tf


def _ensure_dir(path):
    """Creates directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def prepare_cifar100_npz(data_dir="./data"):
    """
    Downloads CIFAR-100 to a local .npz file if not present.
    """
    _ensure_dir(data_dir)
    npz_path = os.path.join(data_dir, "cifar100.npz")

    if os.path.exists(npz_path):
        return npz_path

    print(f"Downloading CIFAR-100 to {data_dir}...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")

    np.savez_compressed(
        npz_path,
        x_train=x_train.astype(np.uint8),
        y_train=y_train.astype(np.int64),
        x_test=x_test.astype(np.uint8),
        y_test=y_test.astype(np.int64)
    )
    return npz_path


def load_cifar100_arrays(data_dir="./data"):
    """
    Loads CIFAR-100 arrays from the local .npz cache.
    Returns: (x_train, y_train, x_test, y_test)
    """
    path = prepare_cifar100_npz(data_dir)
    data = np.load(path)
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"]