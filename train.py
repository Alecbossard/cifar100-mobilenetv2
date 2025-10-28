import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from data_loader import load_cifar100_arrays

DATA_DIR = "./data"
MODELS_DIR = "./models"
BATCH = 64
EPOCHS = 8
IMG_SIZE = (224, 224)
TEST_SUBSET = 1000
FINE_TUNE = False
UNFREEZE_AT = -30

os.makedirs(MODELS_DIR, exist_ok=True)
print("Devices:", tf.config.list_physical_devices())

def make_datasets():
    x_train, y_train, x_test, y_test = load_cifar100_arrays(DATA_DIR)
    n_val = int(0.1 * x_train.shape[0])
    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_tr, y_tr = x_train[n_val:], y_train[n_val:]
    AUTOTUNE = tf.data.AUTOTUNE

    def _map(x, y):
        x = tf.image.resize(x, IMG_SIZE)
        x = tf.cast(x, tf.float32)
        x = preprocess_input(x)
        y = tf.cast(tf.squeeze(y, axis=-1), tf.int32)
        return x, y

    ds_train = (tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
                .shuffle(10000).map(_map, num_parallel_calls=AUTOTUNE)
                .batch(BATCH).prefetch(AUTOTUNE))
    ds_val = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
              .map(_map, num_parallel_calls=AUTOTUNE)
              .batch(BATCH).prefetch(AUTOTUNE))
    x_sub, y_sub = x_test[:TEST_SUBSET], y_test[:TEST_SUBSET]
    ds_test_subset = (tf.data.Dataset.from_tensor_slices((x_sub, y_sub))
                      .map(_map, num_parallel_calls=AUTOTUNE)
                      .batch(BATCH).prefetch(AUTOTUNE))
    ds_test_full = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                    .map(_map, num_parallel_calls=AUTOTUNE)
                    .batch(BATCH).prefetch(AUTOTUNE))
    return ds_train, ds_val, ds_test_subset, ds_test_full

def build_model(base_trainable=False):
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                       include_top=False, weights="imagenet")
    base.trainable = base_trainable
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(100, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model, base


def main():
    ds_train, ds_val, ds_test_subset, ds_test_full = make_datasets()
    model, base = build_model(base_trainable=False)
    model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, verbose=1)
    print("\nÉvaluation sur 1000 images (objectif du challenge) :")
    loss_s, acc_s = model.evaluate(ds_test_subset, verbose=0)
    print(f"Subset test {TEST_SUBSET} — loss={loss_s:.4f}  acc={acc_s*100:.2f}%")
    print("Évaluation sur tout le test (10k images) :")
    loss_f, acc_f = model.evaluate(ds_test_full, verbose=0)
    print(f"Full test — loss={loss_f:.4f}  acc={acc_f*100:.2f}%")
    out_path = os.path.join(MODELS_DIR, "mobilenetv2_cifar100.h5")
    model.save(out_path)
    print(f"Modèle sauvegardé dans: {out_path}")

if __name__ == "__main__":
    main()
