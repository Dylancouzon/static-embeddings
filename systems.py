"""Embedder registry. Every dense system exposes embed(texts, batch_size) -> np.ndarray
(float32). BM25 exposes embed_sparse(texts) -> list[(indices, values)]. Hybrid (H) is
an index/query strategy over A's dense + BM25 sparse, handled in 03/04, not here.

kinds: bm25 | fastembed_dense | model2vec | static_numpy | static_onnx
"""
from __future__ import annotations

import numpy as np


class FastembedDense:
    def __init__(self, model: str):
        from fastembed import TextEmbedding
        self.m = TextEmbedding(model)

    def embed(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        return np.asarray(list(self.m.embed(texts, batch_size=batch_size)), dtype=np.float32)


class Model2Vec:  # candidate B — correctness oracle + speed baseline for A
    def __init__(self, model: str):
        from model2vec import StaticModel
        self.m = StaticModel.from_pretrained(model)

    def embed(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        # max_length=None: full-document encoding (potion seq_length=1e6), matching
        # candidate A's no-truncation rule so B is a faithful oracle. multiprocessing
        # off for determinism + apples-to-apples single-process throughput vs A.
        return self.m.encode(texts, max_length=None, batch_size=batch_size,
                             use_multiprocessing=False, show_progress_bar=False).astype(np.float32)


class StaticOnnx:  # candidate C — potion onnx/model.onnx (EmbeddingBag: flat ids + offsets)
    FILES = ("config.json", "tokenizer.json", "onnx/model.onnx")

    @staticmethod
    def resolve(model: str) -> dict[str, str]:
        from huggingface_hub import hf_hub_download
        return {f: hf_hub_download(model, f) for f in StaticOnnx.FILES}

    def __init__(self, model: str, paths: dict[str, str] | None = None):
        import json

        import onnxruntime as ort
        from tokenizers import Tokenizer
        paths = paths or self.resolve(model)
        cfg = json.load(open(paths["config.json"]))
        self.tokenizer = Tokenizer.from_file(paths["tokenizer.json"])
        self.normalize = bool(cfg.get("normalize", False))
        vocab = self.tokenizer.get_vocab()
        unk = getattr(self.tokenizer.model, "unk_token", None)
        self.unk_token_id = vocab.get(unk) if unk is not None else None
        so = ort.SessionOptions()
        so.intra_op_num_threads = _ort_threads()
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            paths["onnx/model.onnx"], so, providers=["CPUExecutionProvider"])
        self.dim = self.session.get_outputs()[0].shape[1]

    def _ids(self, texts):
        encs = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        unk = self.unk_token_id
        return [[t for t in e.ids if t != unk] if unk is not None else e.ids for e in encs]

    def embed(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            chunk = self._ids(texts[start:start + batch_size])
            flat: list[int] = []
            offsets: list[int] = []
            for ids in chunk:
                offsets.append(len(flat))
                flat.extend(ids)
            if not flat:  # every bag empty -> zeros row (EmbeddingBag can't take empty input)
                continue
            res = self.session.run(None, {
                "input_ids": np.asarray(flat, dtype=np.int64),
                "offsets": np.asarray(offsets, dtype=np.int64),
            })[0]
            out[start:start + len(chunk)] = res
        if self.normalize:
            out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-32)
        return out.astype(np.float32)


class BM25:
    def __init__(self):
        from fastembed import SparseTextEmbedding
        self.m = SparseTextEmbedding("Qdrant/bm25")

    def embed_query(self, texts: list[str]):
        return list(self.m.query_embed(texts))

    def embed_sparse(self, texts: list[str], batch_size: int = 256):
        return list(self.m.embed(texts, batch_size=batch_size))


def _ort_threads() -> int:
    import os
    return int(os.environ.get("ORT_INTRA_OP_THREADS", "1"))


def build(kind: str, model: str | None = None):
    if kind == "bm25":
        return BM25()
    if kind == "fastembed_dense":
        return FastembedDense(model)
    if kind == "model2vec":
        return Model2Vec(model)
    if kind == "static_onnx":
        return StaticOnnx(model)
    if kind == "static_numpy":
        from static_numpy import StaticNumpy
        return StaticNumpy(model)
    raise ValueError(f"unknown system kind: {kind}")
