"""Small synthetic check of masking, checkpoint I/O, and paired evaluation."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from common import Config, MODELS, RATES, _sha_file
from data import cache, encoding, features
from data.batches import MaskedSentimentDataset
from evaluation.controls import custom_banks
from models.fusion import build_model


class ToyEncoder(torch.nn.Module):
    def forward(self, input_ids, attention_mask, token_type_ids, return_dict):
        tokens = input_ids.float() / 101.
        attended = (tokens * attention_mask).sum(1, keepdim=True)
        context = tokens + attended
        return SimpleNamespace(last_hidden_state=torch.stack([context, tokens, context.square(), tokens.square()], -1))


class ToyTokenizer:
    all_special_ids = (0, 1, 101, 102)
    mask_token_id = 1

    def __len__(self):
        return 110


def fixture(name):
    n = 5
    part = features.SplitData(
        name=name, feat={m: np.ones((n, 6, 4), dtype=np.float32) for m in features.MODALITIES},
        vpos={m: np.ones((n, 6), dtype=bool) for m in features.MODALITIES},
        plen={m: np.full(n, 6) for m in features.MODALITIES},
        available={m: np.ones(n, dtype=bool) for m in features.MODALITIES},
        y=np.linspace(-1, 1, n, dtype=np.float32), ids=np.array([f"s{i}" for i in range(n)]),
        sources=np.array(["a", "a", "b", "b", "c"]), vision_zero=np.zeros(n, dtype=bool), meta={},
    )
    part.token_ids = np.repeat([[101, 2, 3, 4, 102, 0]], n, axis=0)
    part.token_ids[:, 2] += np.arange(n)
    part.token_attention = (part.token_ids != 0).astype(int)
    part.token_types = np.zeros_like(part.token_ids)
    part.vpos["text"][:, [0, 4, 5]] = False
    part.content_groups = [[np.array([i]) for i in (1, 2, 3)] for _ in range(n)]
    return part


def selfcheck():
    from run import analyze, evaluate, matrix, train
    from common import _make_generator
    from data.batches import _generate_train_schedule
    from evaluation.metrics import _eval_specs

    torch.set_num_threads(1)
    cfg = Config(batch_size=3, eval_batch_size=3, epochs=1, min_epochs=1, warmup_epochs=0, use_amp=False)
    runs = list(matrix({"B1_FUSION": {"arms": ["CLEAN"], "seeds": [2026]}}))
    settings = {"split": "valid", "allow_test": False, "suites": ["standard", "operators", "new_masks", "blocks"],
                "new_mask_seed": 4311, "replacement_draws": 1, "bootstrap_replicates": 16}
    ds = "MOSI"
    device = features.DEVICE
    part = fixture("valid")
    bank = custom_banks(part, ds, "valid", cfg.mask_seed)
    rate = RATES[0]
    pre = {rate: np.full((*bank[rate].shape, 4), 2., dtype=np.float16)}
    post = np.full((len(part.y), 6, 4), 3., dtype=np.float16)
    items = {}
    for model in MODELS:
        for encoding_name in ("PRE", "POST"):
            dataset = MaskedSentimentDataset(
                part, ds, model, "PRE", post, bank, pre,
                spec={"condition": "test_pair", "encoding": encoding_name, "rate": rate, "repeat": 0},
            )
            items[encoding_name] = dataset[0]
        assert torch.equal(items["PRE"]["keep_text"], items["POST"]["keep_text"])
        model = build_model(model, {m: 4 for m in features.MODALITIES}, {m: 6 for m in features.MODALITIES}, 0.)
        prediction = model(features.collate_batch([items["POST"]]))
        assert prediction.shape == (1,) and torch.isfinite(prediction).all()
    assert len(_eval_specs(True)) == 33
    schedule = _generate_train_schedule(part, ds, cfg)
    clean = MaskedSentimentDataset(part, ds, "B1_FUSION", "CLEAN", post, bank, pre, training=True, schedule=schedule)
    assert np.array_equal(clean[0]["keep_text"].numpy(), part.vpos["text"][0])
    assert torch.equal(torch.rand(3, generator=_make_generator(1)), torch.rand(3, generator=_make_generator(1)))

    with TemporaryDirectory() as scratch:
        root = Path(scratch)
        cache.configure(reserve_gib=0., result_headroom_gib=0., working_limit_gib=1., quota_root=root)
        labels = root / "labels.csv"
        labels.write_text("dummy", encoding="utf-8")

        def load_splits(cfg, ds, root, splits=("train", "valid")):
            parts = {s: fixture(s) for s in splits}
            raw = {s: {"raw_text": ["placeholder"] * 5} for s in splits}
            return parts, raw, labels, labels, _sha_file(labels), {s: "toy" for s in splits}

        def load_hf(*args):
            return ToyEncoder().to(device).eval(), ToyTokenizer(), {"local_files_sha256": {"toy": "toy"}}

        def tokenizer_audit(*args):
            return {"status": "synthetic"}

        def check(force=False):
            return 0.

        with patch.multiple(encoding, load_splits=load_splits, _load_hf=load_hf, _tokenizer_audit=tokenizer_audit), \
             patch("run.load_splits", load_splits), patch("run._load_hf", load_hf), \
             patch("run._tokenizer_audit", tokenizer_audit):
            train(cfg, root, ds, runs, device, check)
            train(cfg, root, ds, runs, device, check)
            evaluate(cfg, root, ds, runs, device, settings, check)
            evaluate(cfg, root, ds, runs, device, settings, check, access_only=True)
            analyze(root, [ds], settings, runs)
        standard = root / "predictions" / "valid" / ds / "standard"
        pre_predictions = pd.read_csv(standard / "B1_FUSION_CLEAN_2026_PRE.csv")
        post_predictions = pd.read_csv(standard / "B1_FUSION_CLEAN_2026_POST.csv")
        assert len(pre_predictions) == len(post_predictions) == 80
        access = pd.read_csv(next((root / "access" / "valid" / ds).glob("*.csv")))
        assert access.pre_retained_rms.max() <= 1e-5 and access.post_retained_rms.max() > 0.
    print("PASS: four model families, matched PRE/POST, CLEAN, packed caches, training, evaluation, and paired analysis")
