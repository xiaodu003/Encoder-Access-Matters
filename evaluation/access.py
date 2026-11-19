"""Change deleted token IDs and measure changes at fixed downstream visibility."""
import numpy as np

from common import PROTOCOL, _atomic_csv
from data.batches import MaskedSentimentDataset
from data.encoding import _bert_forward
from data.features import stable_seed
from evaluation.controls import transform_inputs
from evaluation.metrics import _predict_dataset


def replace_deleted(ids, deleted, vocabulary, seed):
    rng = np.random.default_rng(seed)
    changed = ids.copy()
    values = rng.choice(vocabulary, size=int(deleted.sum()))
    same = values == ids[deleted]
    indices = np.searchsorted(vocabulary, values[same])
    values[same] = vocabulary[(indices + 1) % len(vocabulary)]
    changed[deleted] = values
    return changed


def retained_rms(a, b, keep):
    delta = np.where(keep[..., None], a.astype(np.float64) - b.astype(np.float64), 0.)
    return np.sqrt((delta ** 2).sum((1, 2)) / (keep.sum(1) * a.shape[2]).clip(1))


def audit_access(bert, tokenizer, part, raw_post, banks, scaler, models, out, cfg, device, check, draws):
    special_ids = set(tokenizer.all_special_ids)
    vocabulary = np.array([i for i in range(len(tokenizer)) if i not in special_ids], dtype=np.int64)
    if len(vocabulary) < 2 or draws < 1:
        raise ValueError("Deleted-ID comparison needs two ordinary IDs and at least one draw")
    content = part.vpos["text"] & part.available["text"][:, None]

    def encode(ids, attention, types):
        return _bert_forward(bert, ids, attention, types, device,
                             batch_size=cfg.bert_batch_size, budget_check=check)

    def predict(model, record, hidden, keep, rate, rep):
        text = np.where(keep[..., None], (hidden - scaler["mean"]) / scaler["std"], 0.)
        if np.max(np.abs(text), initial=0.) > np.finfo(np.float16).max or not np.isfinite(text).all():
            raise FloatingPointError("Diagnostic features cannot be stored as float16")
        dataset = MaskedSentimentDataset(
            part, record["dataset"], record["model"], record["arm"], text.astype(np.float16), banks, {},
            spec={"condition": "diagnostic", "encoding": "POST", "rate": rate, "repeat": rep},
        )
        return np.array([r["prediction"] for r in _predict_dataset(model, dataset, cfg)])

    for rate, bank in banks.items():
        for rep, deleted in enumerate(bank):
            keep = content & ~deleted
            eligible = np.flatnonzero(deleted.any(1) & keep.any(1))
            if not len(eligible):
                continue
            pre_ids, blocked, types = transform_inputs(part, deleted, "PRE", tokenizer.mask_token_id)
            pre = encode(pre_ids, blocked, types)
            for draw in range(draws):
                replaced = replace_deleted(part.token_ids, deleted, vocabulary,
                                           stable_seed(PROTOCOL, "deleted_id", cfg.mask_seed, rate, rep, draw))
                changed_pre = encode(replaced, blocked, types)
                changed_post = encode(replaced, part.token_attention, part.token_types)
                if np.abs(pre - changed_pre)[keep].max(initial=0.) > 1e-5:
                    raise AssertionError("Blocked IDs changed retained PRE states")
                pre_rms = retained_rms(pre, changed_pre, keep)
                post_rms = retained_rms(raw_post, changed_post, keep)
                rows = []
                for model, record in models():
                    p_pre = predict(model, record, pre, keep, rate, rep)
                    p_post = predict(model, record, raw_post, keep, rate, rep)
                    delta_pre = np.abs(p_pre - predict(model, record, changed_pre, keep, rate, rep))
                    delta_post = np.abs(p_post - predict(model, record, changed_post, keep, rate, rep))
                    if delta_pre[eligible].max(initial=0.) > 2e-4:
                        raise AssertionError("Blocked IDs changed the downstream PRE prediction")
                    rows.extend({**record, "id": str(part.ids[i]), "source_id": str(part.sources[i]),
                                 "rate": rate, "repeat": rep, "draw": draw,
                                 "deleted_tokens": int(deleted[i].sum()), "retained_tokens": int(keep[i].sum()),
                                 "pre_retained_rms": pre_rms[i], "post_retained_rms": post_rms[i],
                                 "pre_prediction_abs_change": delta_pre[i], "post_prediction_abs_change": delta_post[i],
                                 "original_pre_post_error_gap": abs(p_pre[i] - part.y[i]) - abs(p_post[i] - part.y[i])}
                                for i in eligible)
                _atomic_csv(out / f"r{int(rate * 100)}_{rep}_{draw}.csv", rows)
