import os
import math
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data_loader import load_cifar100_arrays, _ensure_dir

# ==========================================
# Configuration & Setup
# ==========================================
DATA_DIR = "./data"
MODELS_DIR = "./models"
_ensure_dir(MODELS_DIR)

# Hyperparameters
IMG_SIZE = (128, 128)  # 128px for MobileNetV2
BATCH_SIZE = int(os.environ.get("BATCH", 64))
EPOCHS_STAGE1 = 15  # Head training
EPOCHS_STAGE2 = 10  # Fine-tuning
LR = 1e-3
FINE_TUNE_AT = 120  # Unfreeze from layer 120 onwards
TEST_SUBSET_SIZE = 1000
NUM_CLASSES = 100

# 1. GPU Memory Growth (Prevent OOM)
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPUs Detected: {len(gpus)}")
except Exception as e:
    print(f"GPU Setup Error (Ignored): {e}")

# 2. Mixed Precision Policy
# Attempts to use mixed_float16, falls back gracefully if not supported
try:
    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed Precision: Enabled (mixed_float16)")
except Exception:
    print("Mixed Precision: Disabled (using float32)")

# ==========================================
# Data Pipeline
# ==========================================
print("Loading data...")
x_train, y_train, x_test, y_test = load_cifar100_arrays(DATA_DIR)

# Validation Split (first 5000 images)
x_val, y_val = x_train[:5000], y_train[:5000]
x_train, y_train = x_train[5000:], y_train[5000:]

AUTOTUNE = tf.data.AUTOTUNE


def _resize_cast(x):
    """Resizes image to target size and casts to float32."""
    x = tf.image.resize(x, IMG_SIZE, method="bilinear")
    return tf.cast(x, tf.float32)


def make_ds(x, y, training=False):
    """Creates an optimized tf.data.Dataset."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(20000, reshuffle_each_iteration=True)

    # Map: Resize -> Cast -> Batch -> Prefetch
    ds = ds.map(lambda xi, yi: (_resize_cast(xi), tf.cast(yi, tf.int32)),
                num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=training).prefetch(AUTOTUNE)
    return ds


train_ds = make_ds(x_train, y_train, training=True)
val_ds = make_ds(x_val, y_val, training=False)
test_ds = make_ds(x_test[:TEST_SUBSET_SIZE], y_test[:TEST_SUBSET_SIZE], training=False)


# ==========================================
# Custom Loss & Utilities
# ==========================================
def make_sparse_cce_with_smoothing(num_classes=NUM_CLASSES, eps=0.1):
    """
    Creates Categorical Crossentropy loss with Label Smoothing.
    Converts sparse inputs (integers) to one-hot internally.
    """
    cce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=eps)

    def loss(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
        y_true = tf.one_hot(y_true, num_classes)
        return cce(y_true, y_pred)

    return loss


# ==========================================
# Model Builder
# ==========================================
def build_model():
    # Data Augmentation Layers
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1)
    ], name="data_augmentation")

    # Base Model (MobileNetV2)
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = False  # Frozen initially

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = layers.Lambda(preprocess_input, name="mobilenet_preproc")(x)  # Inputs to [-1, 1]

    x = base(x, training=False)  # Keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)

    # Output: float32 is strictly required for stability in Mixed Precision
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

    return models.Model(inputs, outputs, name="MobileNetV2_CIFAR100"), base


model, base_model = build_model()
model.summary(expand_nested=False)

# ==========================================
# Training: Stage 1 (Head Only)
# ==========================================
# Scheduler
steps_per_epoch = math.ceil(x_train.shape[0] / BATCH_SIZE)
cosine1 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR,
    decay_steps=max(1, steps_per_epoch * EPOCHS_STAGE1),
    alpha=0.1
)
optimizer_stage1 = tf.keras.optimizers.Adam(learning_rate=cosine1)

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    ModelCheckpoint(os.path.join(MODELS_DIR, "mobilenetv2_cifar100_best.h5"),
                    monitor="val_accuracy", save_best_only=True, verbose=1)
]

print(f"\n=== STAGE 1: Training Head ({EPOCHS_STAGE1} Epochs) ===")
model.compile(
    optimizer=optimizer_stage1,
    loss=make_sparse_cce_with_smoothing(eps=0.10),
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE1, callbacks=callbacks)

# ==========================================
# Training: Stage 2 (Fine-Tuning)
# ==========================================
print(f"\n=== STAGE 2: Fine-Tuning Backbone ({EPOCHS_STAGE2} Epochs) ===")

# Unfreeze the base model from layer FINE_TUNE_AT
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

print(f"Backbone unfrozen. Training from layer {FINE_TUNE_AT} onwards.")

# Slower learning rate for fine-tuning
cosine2 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR / 10,
    decay_steps=max(1, steps_per_epoch * EPOCHS_STAGE2),
    alpha=0.2
)
optimizer_stage2 = tf.keras.optimizers.Adam(learning_rate=cosine2)

# Recompile with lower label smoothing (eps=0.05)
model.compile(
    optimizer=optimizer_stage2,
    loss=make_sparse_cce_with_smoothing(eps=0.05),
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE2, callbacks=callbacks)

# ==========================================
# Final Evaluation
# ==========================================
print(f"\n=== Final Evaluation on {TEST_SUBSET_SIZE} Test Images ===")
test_loss, test_acc = model.evaluate(test_ds, verbose=2)

print(f"✅ Test Accuracy: {test_acc:.2%}")

final_path = os.path.join(MODELS_DIR, "mobilenetv2_cifar100_final.h5")
model.save(final_path)
print(f"Model saved to: {final_path}")