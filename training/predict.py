"""Run the int8 model on local images: python predict.py img1.jpg img2.png"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from train import LABELS, square_resize

HERE = Path(__file__).resolve().parent

def main(paths):
    interp = tf.lite.Interpreter(model_path=str(HERE / "output" / "model.tflite"))
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    size = inp["shape"][1]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    for p in paths:
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = square_resize(tf.cast(img, tf.float32), size)
        x = img.numpy() / 127.5 - 1.0                               # same scaling as training
        q = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q[None])
        interp.invoke()
        probs = (interp.get_tensor(out["index"])[0].astype(np.float32) - out_zp) * out_scale
        best = int(np.argmax(probs))
        verdict = LABELS[best] if probs[best] >= 0.7 else "not sure"
        print(f"{p}: {verdict}  (" + ", ".join(f"{l} {v:.0%}" for l, v in zip(LABELS, probs)) + ")")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
