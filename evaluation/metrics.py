"""Evaluate the matched conditions and select checkpoints on validation MAE."""
import numpy as np
import torch
from common import RATES, N_MASKS
from data import features as core
from data.batches import MaskedSentimentDataset

def _metrics(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    out = core.binary_metrics(y, pred)
    out.update(mae=float(np.mean(np.abs(y - pred))), mse=float(np.mean((y - pred) ** 2)))
    return out


def _eval_specs(full: bool):
    specs = [{"condition": "clean", "encoding": "POST", "rate": 0.0, "repeat": 0}]
    reps = range(N_MASKS)
    for encoding in ("PRE", "POST"):
        for rate in RATES:
            for repeat in reps:
                specs.append({"condition": f"{encoding}_r{int(rate*100):02d}_m{repeat:02d}",
                              "encoding": encoding, "rate": rate, "repeat": repeat})
    return specs


def _evaluate(model, model_name, arm, part, ds, post_cache, delete_banks, pre_cache,
              cfg, device, full: bool, prefix: str, budget_check=None):
    results, pred_rows = [], []
    y = part.y
    for spec in _eval_specs(full):
        if budget_check is not None:
            budget_check()
        dataset = MaskedSentimentDataset(part, ds, model_name, arm, post_cache, delete_banks, pre_cache,
                             training=False, spec=spec)
        preds = _predict_dataset(model, dataset, cfg)
        if len(preds) != len(y) or not np.array_equal(np.asarray([r["id"] for r in preds]), np.asarray(part.ids, dtype=str)):
            # DataLoader is sequential; any mismatch is an identity error, not a row to drop.
            raise ValueError(f"{ds}/{model_name}/{arm}/{spec['condition']}: prediction IDs changed order")
        p = np.asarray([r["prediction"] for r in preds], dtype=np.float64)
        metrics = _metrics(y, p)
        metrics.update({"dataset": ds, "model": model_name, "arm": arm,
                        "checkpoint": prefix, "condition": spec["condition"],
                        "encoding": spec["encoding"], "rate": spec["rate"], "repeat": spec["repeat"]})
        results.append(metrics)
        if full or prefix in ("epoch_10", "epoch_20", "epoch_40"):
            for row in preds:
                pred_rows.append({"dataset": ds, "model": model_name, "arm": arm,
                                  "checkpoint": prefix, "condition": spec["condition"], **row})
    return results, pred_rows


def _composite_metrics(rows):
    expected = {x["condition"] for x in _eval_specs(True)}
    by = {r["condition"]: r for r in rows}
    if len(rows) != 33 or set(by) != expected:
        raise ValueError("Selection requires exactly the complete 33 conditions")
    clean = float(by["clean"]["mae"])
    pre = float(np.mean([r["mae"] for c, r in by.items() if c.startswith("PRE_")]))
    post = float(np.mean([r["mae"] for c, r in by.items() if c.startswith("POST_")]))
    metrics = {"clean_mae": clean, "pre_mae": pre, "post_mae": post,
               "selection_score": (clean + pre + post) / 3}
    for key, alias in (("pearson_r", "corr"), ("acc2_including_zero", "acc2"),
                       ("weighted_f1_including_zero", "f1")):
        if key in by["clean"]:
            metrics[f"clean_{alias}"] = by["clean"][key] if by["clean"][key] is not None else float('nan')
    return metrics


def _selection_score(rows):
    return _composite_metrics(rows)["selection_score"]


@torch.inference_mode()
def _predict_dataset(model, dataset, cfg):
    """Read resident/mmap arrays directly: no per-condition worker lifecycle."""
    model.eval()
    records = []
    for start in range(0, len(dataset), cfg.eval_batch_size):
        batch = core.collate_batch([dataset[i] for i in range(start, min(start + cfg.eval_batch_size, len(dataset)))])
        b = core.move_batch(batch)
        pred = model(b).float().cpu().numpy()
        if not np.isfinite(pred).all():
            raise FloatingPointError("nonfinite FP32 validation prediction")
        for sid, src, y, p in zip(batch["id"], batch["source_id"], batch["target"].numpy(), pred):
            records.append({"id": str(sid), "source_id": str(src), "target": float(y),
                            "prediction": float(p), "abs_error": abs(float(y) - float(p))})
    return records
