"""Candidate A — native numpy port of model2vec static inference.

IMPORTS ONLY numpy, tokenizers, huggingface_hub (candidate A rule). No torch,
safetensors, or model2vec — assert_no_forbidden_imports() proves it in the
measured process. Replicates model2vec._encode_batch exactly so A==B within
atol=1e-5 (see correctness_gate.py). Verified against the real potion files:
tensor name 'embeddings' F32, unk_token_id resolved from tokenizer, mean-pool,
L2-normalize (+1e-32) iff config.normalize.
"""
from __future__ import annotations

import json
import struct
import sys

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

FORBIDDEN = {"torch", "safetensors", "model2vec"}


def assert_no_forbidden_imports() -> None:
    """Fail if A's process ever loaded a dependency it isn't allowed to ship with."""
    loaded = FORBIDDEN & set(sys.modules)
    assert not loaded, f"candidate A loaded forbidden modules: {loaded}"


def _parse_safetensors(path: str) -> tuple[np.ndarray, str]:
    """np.memmap the embedding matrix straight from the safetensors buffer.

    Layout: 8-byte LE u64 header length, JSON header (per-tensor dtype/shape/
    data_offsets), then the flat tensor buffer. Matrix lives at 8 + header_len +
    tensor_offset.
    """
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    name = "embeddings" if "embeddings" in header else next(
        k for k, v in header.items()
        if k != "__metadata__" and len(v["shape"]) == 2)
    meta = header[name]
    dtype = {"F32": np.float32, "F16": np.float16, "F64": np.float64}[meta["dtype"]]
    offset = 8 + header_len + meta["data_offsets"][0]
    matrix = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=tuple(meta["shape"]))
    return matrix, name


class StaticNumpy:
    FILES = ("model.safetensors", "config.json", "tokenizer.json")

    @staticmethod
    def resolve(model_id: str) -> dict[str, str]:
        """Cache-check / download phase — kept separate from load so V4 can time
        mmap+tokenizer alone (CLAUDE.md phase split)."""
        return {f: hf_hub_download(model_id, f) for f in StaticNumpy.FILES}

    def __init__(self, model_id: str, paths: dict[str, str] | None = None):
        paths = paths or self.resolve(model_id)
        st = paths["model.safetensors"]
        cfg = json.load(open(paths["config.json"]))
        self.tokenizer = Tokenizer.from_file(paths["tokenizer.json"])
        self.embedding, _ = _parse_safetensors(st)
        self.dim = self.embedding.shape[1]
        self.normalize = bool(cfg.get("normalize", False))
        # model2vec resolves unk via tokenizer.model.unk_token, then vocab lookup.
        vocab = self.tokenizer.get_vocab()
        unk = getattr(self.tokenizer.model, "unk_token", None)
        self.unk_token_id = vocab.get(unk) if unk is not None else None

    def _token_ids(self, texts: list[str]) -> list[list[int]]:
        encs = self.tokenizer.encode_batch(texts, add_special_tokens=False)
        if self.unk_token_id is None:
            return [e.ids for e in encs]
        unk = self.unk_token_id
        return [[t for t in e.ids if t != unk] for e in encs]

    def embed(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        # batch_size only bounds tokenizer memory; math is per-document.
        out: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            for ids in self._token_ids(texts[start:start + batch_size]):
                if ids:
                    out.append(self.embedding[ids].mean(axis=0))
                else:
                    out.append(np.zeros(self.dim))
        arr = np.stack(out) if out else np.zeros((0, self.dim))
        if self.normalize:
            arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-32)
        return arr.astype(np.float32)


if __name__ == "__main__":
    # smoke check against the reference captured from model2vec on potion-retrieval-32M
    m = StaticNumpy("minishlab/potion-retrieval-32M")
    v = m.embed(["hello world"])[0]
    ref = np.array([0.019733, -0.01530093, -0.08678473, 0.02290591, 0.04700558])
    assert np.allclose(v[:5], ref, atol=1e-5), f"drift vs model2vec: {v[:5]}"
    assert_no_forbidden_imports()
    print("static_numpy OK — matches model2vec reference, no forbidden imports")
