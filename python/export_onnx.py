"""Export the trained scikit-learn model to ONNX (model.onnx).

ONNX is a language-neutral format: the resulting model.onnx can run in Node.js
(e.g. a Next.js API route via onnxruntime-node) with no Python at runtime.

Run:  .venv/bin/python export_onnx.py

Tensor names the Node.js side needs:
  input:  "float_input"  shape [1, 7], float32, in this column order:
          Team, Technology, Market, Value Proposition,
          Competitive Advantage, Socially Impactful, funding
  outputs: "label"          -> int64  [1]      (0 = pass, 1 = invest)
           "probabilities"  -> float32 [1, 2]  ([:,1] = invest probability)
"""

import os
import numpy as np
import joblib
import onnx.helper as _oh
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

# Compatibility shim: skl2onnx emits some tree attributes (e.g.
# nodes_missing_value_tracks_true) as booleans, but modern onnx's make_attribute
# rejects bools where ints are expected. Coerce bools -> ints for INT attributes.
_orig_make_attribute = _oh.make_attribute


def _make_attribute(key, value, *args, **kwargs):
    if isinstance(value, (list, tuple, np.ndarray)):
        coerced = [int(v) if isinstance(v, (bool, np.bool_)) else v for v in value]
        if any(isinstance(v, (bool, np.bool_)) for v in value):
            value = coerced
    elif isinstance(value, (bool, np.bool_)):
        value = int(value)
    return _orig_make_attribute(key, value, *args, **kwargs)


_oh.make_attribute = _make_attribute

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(SCRIPT_DIR, "../models/model.joblib")
ONNX_FILE = os.path.join(SCRIPT_DIR, "../models/model.onnx")


def main():
    bundle = joblib.load(MODEL_FILE)
    model, features = bundle["model"], bundle["features"]
    n = len(features)

    onnx_model = to_onnx(
        model,
        initial_types=[("float_input", FloatTensorType([None, n]))],
        # zipmap=False -> probabilities come back as a plain [N, 2] tensor
        # instead of a list-of-dicts, which is far easier to read from JS.
        options={"zipmap": False},
        target_opset=17,
    )
    with open(ONNX_FILE, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Wrote {ONNX_FILE}  (input 'float_input' shape [None, {n}])")
    print(f"Feature order: {features}")


if __name__ == "__main__":
    main()
