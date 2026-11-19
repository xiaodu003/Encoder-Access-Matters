"""Build matched model inputs from encoded features and deletion masks."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from common import Config, PROTOCOL, RATES, N_MASKS, _make_generator
from data import features as core

def _generate_train_schedule(part, ds: str, cfg: Config):
    n = len(part.y)
    clean = np.zeros((cfg.epochs, n), dtype=bool)
    rate_index = np.zeros((cfg.epochs, n), dtype=np.int8)
    source_pre = np.zeros((cfg.epochs, n), dtype=bool)
    for ep in range(cfg.epochs):
        for i, sample_id in enumerate(part.ids):
            rng = core.rng_for(PROTOCOL, cfg.seed, ds, "train_schedule", ep, str(sample_id))
            clean[ep, i] = bool(rng.random() < cfg.clean_probability)
            rate_index[ep, i] = int(rng.integers(0, len(RATES)))
            source_pre[ep, i] = bool(rng.random() < 0.5)
    return {"clean": clean, "rate_index": rate_index, "source_pre": source_pre}


class MaskedSentimentDataset(Dataset):
    def __init__(self, part, ds: str, model_name: str, arm: str, post_cache,
                 delete_banks, pre_cache, training=False, epoch=0, schedule=None,
                 spec=None):
        self.part, self.ds, self.model_name, self.arm = part, ds, model_name, arm
        self.post_cache = post_cache
        self.delete_banks = delete_banks
        self.pre_cache = pre_cache
        self.training, self.epoch, self.schedule, self.spec = training, epoch, schedule, spec
        self.mods = ("text",) if model_name == "B0_TEXT" else core.MODALITIES

    def __len__(self):
        return len(self.part.y)

    def _text(self, i):
        p = self.part
        content = p.vpos["text"][i] & bool(p.available["text"][i])
        if self.training:
            clean = self.arm == "CLEAN" or bool(self.schedule["clean"][self.epoch, i])
            if clean:
                return np.asarray(self.post_cache[i], dtype=np.float32).copy(), content.copy()
            ri = int(self.schedule["rate_index"][self.epoch, i])
            rate = RATES[ri]
            mask_index = self.epoch % N_MASKS
            deleted = self.delete_banks[rate][mask_index, i]
            use_pre = self.arm == "PRE" or (self.arm == "MIX" and bool(self.schedule["source_pre"][self.epoch, i]))
            keep = content & ~deleted
            if use_pre:
                x = np.asarray(self.pre_cache[rate][mask_index, i], dtype=np.float32)
            else:
                x = np.asarray(self.post_cache[i], dtype=np.float32).copy()
                x[~keep] = 0.0
            return x, keep
        spec = self.spec
        if spec["condition"] == "clean":
            return np.asarray(self.post_cache[i], dtype=np.float32).copy(), content.copy()
        rate, mask_index = spec["rate"], int(spec["repeat"])
        deleted = self.delete_banks[rate][mask_index, i]
        keep = content & ~deleted
        if spec["encoding"] == "PRE":
            x = np.asarray(self.pre_cache[rate][mask_index, i], dtype=np.float32)
        else:
            x = np.asarray(self.post_cache[i], dtype=np.float32).copy()
            x[~keep] = 0.0
        return x, keep

    def __getitem__(self, i):
        p = self.part
        xt, kt = self._text(int(i))
        row = {"target": torch.tensor(float(p.y[i]), dtype=torch.float32),
               "id": str(p.ids[i]), "source_id": str(p.sources[i]),
               "row_index": torch.tensor(int(i), dtype=torch.long)}
        for m in self.mods:
            if m == "text":
                x, keep = xt, kt
            else:
                x = np.asarray(p.feat[m][i], dtype=np.float32).copy()
                keep = p.vpos[m][i] & bool(p.available[m][i])
                x[~keep] = 0.0
            row["x_" + m] = torch.from_numpy(np.ascontiguousarray(x))
            row["keep_" + m] = torch.from_numpy(keep.astype(np.float32))
            row["vpos_" + m] = torch.from_numpy(p.vpos[m][i].astype(np.float32))
            row["plen_" + m] = torch.tensor(max(1, int(p.plen[m][i])), dtype=torch.long)
            row["axis_width_" + m] = torch.tensor(int(x.shape[0]), dtype=torch.long)
            row["vlen_" + m] = torch.tensor(float(p.vpos[m][i].sum()), dtype=torch.float32)
            row["klen_" + m] = torch.tensor(int(keep.sum()), dtype=torch.long)
            row["available_" + m] = torch.tensor(bool(p.available[m][i]))
            row["stats_" + m] = torch.zeros(3, dtype=torch.float32)
        return row


def _loader(dataset, cfg: Config, shuffle=False, seed=0, batch_size=None):
    kwargs = {"dataset": dataset, "batch_size": batch_size or cfg.batch_size,
              "shuffle": shuffle, "num_workers": cfg.workers,
              "pin_memory": torch.cuda.is_available(), "collate_fn": core.collate_batch,
              "generator": _make_generator(seed)}
    if cfg.workers > 0:
        kwargs.update(prefetch_factor=2, persistent_workers=False)
    return DataLoader(**kwargs)
