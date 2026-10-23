"""Shared helpers for the Core ML feasibility spike (not product code)."""
import os
import pathlib
import time

import numpy as np
import torch

ART_DIR = pathlib.Path(os.environ.get("NANO_SPIKE_DIR", os.path.expanduser("~/nano-spike/coreml")))

LINES = [
    "Which number, times 7, makes 56?",
    "12 divided by which number makes 3?",
    "Make both sides worth the same. What does the missing plank need to be?",
]

MAX_CTX = 1024
MAX_TEXT_LEN = 512
MAX_SPEECH_TOKENS = 600
SPEECH_TOKEN_BUCKETS = (64, 128, 192)
N_CFM_TIMESTEPS = 2
HIFT_UPSAMPLE = 480  # prod(upsample_rates) * istft hop_len = 8*5*3*4


def artifacts_dir() -> pathlib.Path:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    return ART_DIR


def load_nano(device="cpu"):
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    torch.set_num_threads(2)
    t0 = time.time()
    model = ChatterboxTurboTTS.from_pretrained(device=device, nano=True)
    model.t3.eval()
    model.s3gen.eval()
    print(f"[common] loaded nano in {time.time() - t0:.1f}s")
    return model


def max_abs_err(a, b) -> float:
    a = a.detach().cpu().float().numpy() if torch.is_tensor(a) else np.asarray(a, dtype=np.float64)
    b = b.detach().cpu().float().numpy() if torch.is_tensor(b) else np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"cannot compare shapes {a.shape} and {b.shape}")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def causal_mask(q_len: int, kv_len: int, dtype=torch.float32, neg=-1e4) -> torch.Tensor:
    """Additive mask [1, 1, q_len, kv_len]; the query at offset i may attend to keys <= past+i."""
    past = kv_len - q_len
    idx_q = torch.arange(q_len)[:, None] + past
    idx_k = torch.arange(kv_len)[None, :]
    allowed = idx_k <= idx_q
    mask = torch.where(allowed, torch.zeros((), dtype=dtype), torch.full((), neg, dtype=dtype))
    return mask[None, None]


def dir_size(path) -> int:
    path = pathlib.Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def human(nbytes: int) -> str:
    return f"{nbytes / (1024 * 1024):.1f} MiB"
