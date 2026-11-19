"""Configuration and shared experiment utilities."""
import hashlib, json, os, random, time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data import features as core, cache as storage

PROTOCOL = "position_matched_text_missingness"
RATES = (0.2, 0.5)
N_MASKS = 8
MODELS = ("B0_TEXT", "B1_FUSION", "B2_RATE", "ALMT_FEATURE")

@dataclass(frozen=True)
class Config:
    models: tuple[str, ...] = MODELS
    input_root: str = "inputs"
    output_root: str = "runs"
    dataset_paths: dict = field(default_factory=dict)
    bert_paths: dict = field(default_factory=dict)
    workers: int = 0
    batch_size: int = 32
    eval_batch_size: int = 64
    epochs: int = 40
    min_epochs: int = 12
    patience: int = 8
    warmup_epochs: int = 4
    learning_rate: float = 1e-3
    almt_learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    use_amp: bool = True
    gpu_hour_limit: float = 8.0
    allocated_gpu_count: int = 0  # 0: charge all visible allocated GPUs; >0 must match visibility
    seed: int = 2026
    mask_seed: int = 2026
    bert_batch_size: int = 32
    clean_probability: float = 0.5


class BudgetReached(RuntimeError):
    def __init__(self, used_gpu_hours: float, limit_gpu_hours: float):
        self.used_gpu_hours = float(used_gpu_hours)
        self.limit_gpu_hours = float(limit_gpu_hours)
        super().__init__(f"allocated GPU-hour budget reached ({used_gpu_hours:.3f}/{limit_gpu_hours:.3f})")


def _budget_check(cfg: Config, started: float, prior_gpu_hours: float = 0.0,
                  ledger_path: Path | None = None, run_fingerprint: str | None = None,
                  allocated_gpu_count: int | None = None):
    if cfg.gpu_hour_limit is None or cfg.gpu_hour_limit <= 0:
        return float(prior_gpu_hours)
    count = max(0, int(allocated_gpu_count if allocated_gpu_count is not None
                        else cfg.allocated_gpu_count))
    used = float(prior_gpu_hours) + (time.perf_counter() - started) * count / 3600
    if ledger_path is not None:
        _json(ledger_path, {"run_fingerprint": run_fingerprint,
                            "allocated_gpu_hours": used,
                            "allocated_gpu_count": count,
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    if used >= cfg.gpu_hour_limit:
        raise BudgetReached(used, cfg.gpu_hour_limit)
    return used


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(core.json_ready(value), ensure_ascii=False, indent=2,
                              default=str, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_csv(path: Path, rows, columns=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame = pd.DataFrame.from_records(rows, columns=columns)
    storage.reserve_file(path.parent, 3*int(frame.memory_usage(index=True, deep=True).sum()) + (1 << 20), "CSV export")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha_array(arr) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def _fingerprint(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _atomic_torch_save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    storage.reserve_file(path.parent, int(storage.estimate_tree_bytes(value)*1.3) + (8 << 20), "Checkpoint export")
    torch.save(value, tmp)
    os.replace(tmp, path)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g
