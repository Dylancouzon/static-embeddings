"""Candidate A subprocess entrypoint — runs in a clean process that imports only
numpy/tokenizers/huggingface_hub (via static_numpy), then asserts the import guard.
Used by correctness_gate.py and speed.py so A's measured process provably excludes
torch/safetensors/model2vec.

Usage: python embed_a.py <model_id> <texts.json> <out.npy>
"""
import json
import sys

import numpy as np

from static_numpy import StaticNumpy, assert_no_forbidden_imports

model_id, texts_path, out_path = sys.argv[1:4]
texts = json.loads(open(texts_path).read())
vecs = StaticNumpy(model_id).embed(texts)
assert_no_forbidden_imports()
np.save(out_path, vecs)
