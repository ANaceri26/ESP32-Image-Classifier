"""Fine-tune MobileNetV2 on Cats vs Dogs, quantize to int8, export model.tflite and model_data.h.

Usage: python train.py [--limit N]
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow's info spam

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

HERE = Path(__file__).resolve().parent
SKETCH_DIRS = [HERE.parent / "esp32" / "cat_dog_cam", HERE.parent / "esp32" / "cat_dog_usb"]  # model goes to both
LABELS = ["cat", "dog"]  # tfds label 0 = cat, 1 = dog

def square_resize(image, size):
    """Centre-crop to a square, then resize. The ESP32 does exactly the same to camera frames,
    so training and deployment see images prepared the same way."""
    shape = tf.shape(image)
    side = tf.minimum(shape[0], shape[1])
    image = tf.image.resize_with_crop_or_pad(image, side, side)
    return tf.image.resize(image, (size, size))  # float32, values still 0..255

def fix_tfds_windows_paths():
    """Work around a tensorflow-datasets bug on Windows.

    Its cats_vs_dogs builder normalises paths with os.path.normpath, which on Windows turns
    'PetImages/Cat/0.jpg' into 'PetImages\\Cat\\0.jpg'. Paths inside a zip file always use '/',
    so the lookup fails with: KeyError: "There is no item named 'PetImages\\Cat\\0.jpg'".
    Here we give that one module a normpath that keeps '/' (what it already does on Mac/Linux).
    """
    if os.sep == "/":
        return
    import posixpath
    from tensorflow_datasets.image_classification import cats_vs_dogs

    class PathProxy:
        def __getattr__(self, name):
            return getattr(os.path, name)

        @staticmethod
        def normpath(p):
            return posixpath.normpath(str(p).replace("\\", "/"))

    class OsProxy:
        path = PathProxy()

        def __getattr__(self, name):
            return getattr(os, name)

    cats_vs_dogs.os = OsProxy()

def load_data(size, batch, limit):
    print("[1/7] Loading Cats vs Dogs (first run downloads ~800 MB)...")
    fix_tfds_windows_paths()
    splits = ["train[:80%]", "train[80%:90%]", "train[90%:]"]
    train, val, test = tfds.load("cats_vs_dogs", split=splits, as_supervised=True)
    if limit:
        train, val, test = train.take(limit), val.take(limit // 8), test.take(limit // 8)

    # brightness/contrast jitter helps with the ESP camera's duller colours
    augment = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.08),
        keras.layers.RandomZoom(0.15),
        keras.layers.RandomBrightness(0.25, value_range=(0, 255)),
        keras.layers.RandomContrast(0.25),
    ], name="augment")

    def prep(img, label):
        return square_resize(img, size), label

    def to_model_range(img, label):
        return keras.applications.mobilenet_v2.preprocess_input(img), label

    AUTO = tf.data.AUTOTUNE
    train = (train.map(prep, AUTO).shuffle(2000).batch(batch)
             .map(lambda x, y: (augment(x, training=True), y), AUTO)
             .map(lambda x, y: (tf.clip_by_value(x, 0, 255), y), AUTO)
             .map(to_model_range, AUTO).prefetch(AUTO))
    val = val.map(prep, AUTO).batch(batch).map(to_model_range, AUTO).cache().prefetch(AUTO)
    test = test.map(prep, AUTO).batch(batch).map(to_model_range, AUTO).cache().prefetch(AUTO)
    return train, val, test

def build_model(size, alpha):
    print(f"[2/7] Building MobileNetV2 (alpha={alpha}, input {size}x{size}x3) with ImageNet weights...")
    base = keras.applications.MobileNetV2(input_shape=(size, size, 3), alpha=alpha,
                                          include_top=False, weights="imagenet")
    base.trainable = False  # freeze: keep the pretrained pattern detectors as they are
    inputs = keras.Input((size, size, 3))
    x = base(inputs, training=False)          # BatchNorm stays in inference mode
    x = keras.layers.GlobalAveragePooling2D()(x)  # each feature map -> one number
    x = keras.layers.Dropout(0.2)(x)          # randomly drop 20% during training (regularisation)
    outputs = keras.layers.Dense(len(LABELS), activation="softmax")(x)  # 2 probabilities that sum to 1
    model = keras.Model(inputs, outputs)
    print(f"      total weights: {model.count_params():,} | trainable now: "
          f"{sum(int(np.prod(w.shape)) for w in model.trainable_weights):,}")
    return model, base

def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",  # "how wrong" for integer class labels
                  metrics=["accuracy"])

def representative_images(dataset, n=300):
    """A few hundred real inputs so the converter can measure each layer's value range."""
    count = 0
    for images, _ in dataset:
        for img in images:
            yield [tf.expand_dims(img, 0)]
            count += 1
            if count >= n:
                return

def quantize(model, rep_ds, out_dir):
    print("[6/7] Quantizing to int8...")
    if not hasattr(tf, "lite"):
        raise SystemExit("This TensorFlow has no tf.lite. Install a version below 2.21: "
                         'pip install "tensorflow>=2.16,<2.21"')
    saved = out_dir / "saved_model"
    model.export(str(saved))  # SavedModel format, the converter's most reliable input
    conv = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: representative_images(rep_ds)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]  # integer-only ops
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()

def evaluate_tflite(tflite_bytes, dataset):
    """Run the int8 model exactly like the ESP32 will: quantize input, run, dequantize output."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    correct = total = 0
    for images, labels in dataset:
        for img, label in zip(images.numpy(), labels.numpy()):
            q = np.clip(np.round(img / in_scale + in_zp), -128, 127).astype(np.int8)
            interp.set_tensor(inp["index"], q[None])
            interp.invoke()
            pred = int(np.argmax(interp.get_tensor(out["index"])[0]))
            correct += pred == label
            total += 1
    return correct / total, (in_scale, in_zp)

def write_c_header(tflite_bytes, size, in_q, path):
    """A .tflite file can't be opened on a microcontroller (no file system), so the model
    becomes a C array compiled into the firmware and stored in flash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [", ".join(f"0x{b:02x}" for b in tflite_bytes[i:i + 16]) for i in range(0, len(tflite_bytes), 16)]
    path.write_text(
        "// Generated by train.py - do not edit by hand.\n"
        "#pragma once\n#include <stdint.h>\n\n"
        f"#define MODEL_INPUT_SIZE {size}\n"
        f"// input quantization (for reference; the sketch reads it from the model): "
        f"scale={in_q[0]:.8f} zero_point={in_q[1]}\n"
        f'static const char *MODEL_LABELS[] = {{{", ".join(chr(34) + l + chr(34) for l in LABELS)}}};\n'
        f"static const int MODEL_NUM_LABELS = {len(LABELS)};\n\n"
        f"alignas(16) const unsigned char g_model[] = {{\n  " + ",\n  ".join(lines) + "\n};\n"
        f"const unsigned int g_model_len = {len(tflite_bytes)};\n")

def plot_history(histories, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history["accuracy"]; val_acc += h.history["val_accuracy"]
        loss += h.history["loss"]; val_loss += h.history["val_loss"]
    ep = range(1, len(acc) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, acc, label="train"); a1.plot(ep, val_acc, label="validation"); a1.set_title("Accuracy")
    a2.plot(ep, loss, label="train"); a2.plot(ep, val_loss, label="validation"); a2.set_title("Loss")
    for a in (a1, a2):
        a.axvline(len(histories[0].history["loss"]) + 0.5, ls="--", c="gray")  # where fine-tuning starts
        a.set_xlabel("epoch"); a.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"      training curves saved to {path}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=96, help="input image size (96 fits the ESP32 well)")
    ap.add_argument("--alpha", type=float, default=0.35, help="MobileNetV2 width: 0.35 smallest ... 1.0 full")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--head-epochs", type=int, default=6)
    ap.add_argument("--fine-epochs", type=int, default=6)
    ap.add_argument("--unfreeze", type=int, default=40, help="how many top layers to fine-tune")
    ap.add_argument("--limit", type=int, default=0, help="use only this many training images (quick test)")
    args = ap.parse_args()

    out = HERE / "output"
    out.mkdir(exist_ok=True)
    tf.keras.utils.set_random_seed(42)  # same results each run

    train, val, test = load_data(args.size, args.batch, args.limit)
    model, base = build_model(args.size, args.alpha)
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    print("[3/7] Transfer learning: training only the new head (learning rate 1e-3)...")
    compile_model(model, 1e-3)
    h1 = model.fit(train, validation_data=val, epochs=args.head_epochs, callbacks=[stop])

    print(f"[4/7] Fine-tuning the top {args.unfreeze} MobileNet layers (learning rate 1e-5)...")
    base.trainable = True
    for layer in base.layers[:-args.unfreeze]:
        layer.trainable = False
    for layer in base.layers:  # BatchNorm layers stay frozen: their statistics come from ImageNet
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 1e-5)  # tiny steps so we don't wreck the pretrained weights
    h2 = model.fit(train, validation_data=val, epochs=args.fine_epochs, callbacks=[stop])
    plot_history([h1, h2], out / "training_curves.png")

    print("[5/7] Testing on photos the model has never seen...")
    _, test_acc = model.evaluate(test, verbose=0)
    y_true = np.concatenate([y.numpy() for _, y in test])
    y_pred = np.argmax(model.predict(test, verbose=0), axis=1)
    cm = np.zeros((2, 2), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    print(f"      float32 test accuracy: {test_acc:.2%}")
    print(f"      confusion matrix (rows = truth, cols = prediction):")
    print(f"                 pred cat  pred dog\n      true cat   {cm[0,0]:8d}  {cm[0,1]:8d}\n"
          f"      true dog   {cm[1,0]:8d}  {cm[1,1]:8d}")
    model.save(out / "cat_dog_float.keras")

    tflite = quantize(model, val, out)  # validation images = realistic, un-augmented inputs
    (out / "model.tflite").write_bytes(tflite)
    int8_acc, in_q = evaluate_tflite(tflite, test)
    float_kb = model.count_params() * 4 / 1024
    print(f"      int8 test accuracy:    {int8_acc:.2%}  (change {100 * (int8_acc - test_acc):+.2f} points)")
    print(f"      size: ~{float_kb:,.0f} KB as float32  ->  {len(tflite) / 1024:,.0f} KB as int8 .tflite")

    print("[7/7] Writing model_data.h for the ESP32 sketch...")
    for d in SKETCH_DIRS:
        write_c_header(tflite, args.size, in_q, d / "model_data.h")
        print(f"      {d / 'model_data.h'}")
    print("Done. Open esp32/cat_dog_usb/cat_dog_usb.ino (USB version) in Arduino IDE and upload.")

if __name__ == "__main__":
    main()
