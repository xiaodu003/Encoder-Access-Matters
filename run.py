"""Run paper training, fixed-checkpoint evaluation, or paired analysis."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd
import torch

from analysis.paired import summarize
from common import BudgetReached, Config, MODELS, N_MASKS, PROTOCOL, RATES, _atomic_csv, _budget_check, _json, _sha_file
from data import cache
from data.encoding import (
    _bert_forward, _cache_dataset, _load_hf, _normalise_data, _tokenizer_audit, load_splits,
)
from evaluation.access import audit_access
from evaluation.controls import _scalers, custom_banks, encode_variant, evaluate_condition
from models.fusion import build_model
from training.engine import _amp_gate, _train_arm

ROOT = Path(__file__).resolve().parent
DATASETS = ("MOSI", "SIMS_V1", "MOSEI")
SUITES = ("standard", "operators", "new_masks", "blocks")


def matrix(selection):
    for model, entry in selection.items():
        if model not in MODELS:
            raise ValueError(f"Unknown model {model}")
        for arm in entry["arms"]:
            if arm not in ("CLEAN", "POST", "PRE", "MIX"):
                raise ValueError(f"Unknown training condition {arm}")
            for seed in entry["seeds"]:
                yield model, arm, int(seed)


def training_root(root, seed):
    return root / "training" / f"seed_{seed}"


def arm_directory(root, dataset, model, arm, seed):
    return training_root(root, seed) / "configs" / dataset / f"{model}_{arm}"


def code_hashes():
    paths = [ROOT / "run.py", ROOT / "common.py"]
    for directory in ("data", "models", "training", "evaluation"):
        paths.extend(sorted((ROOT / directory).glob("*.py")))
    return {p.relative_to(ROOT).as_posix(): _sha_file(p) for p in paths}


def train(cfg, root, ds, runs, device, check):
    parts, post, banks, pre_files, identity, _ = _cache_dataset(
        cfg, ds, root / "data" / ds, device, budget_check=check
    )
    cfg = replace(cfg, models=tuple(dict.fromkeys(m for m, _, _ in runs)))
    amp, reason = _amp_gate(ds, parts["train"], post, banks, pre_files, cfg, device)
    _json(root / "data" / ds / "precision.json", {"enabled": amp, "reason": reason})
    options = asdict(cfg)
    for key in ("input_root", "output_root", "dataset_paths", "bert_paths", "models", "seed",
                "workers", "allocated_gpu_count", "gpu_hour_limit"):
        options.pop(key)
    recorded = {"dataset": ds, "pkl_sha256": identity["pkl_sha256"], "csv_sha256": identity["csv_sha256"],
                "scalers_sha256": identity["scalers_sha256"],
                "bert_files": identity["bert"]["local_files_sha256"], "cache_key": identity["cache_key"],
                "amp": amp, "training": options, "code": code_hashes(),
                "train_label_median": float(np.median(parts["train"].y))}
    for model, arm, seed in runs:
        check()
        arm_dir = arm_directory(root, ds, model, arm, seed)
        run_identity = {**recorded, "model": model, "arm": arm, "seed": seed}
        marker = arm_dir / "complete.json"
        if marker.is_file():
            state = json.loads(marker.read_text(encoding="utf-8"))
            checkpoint = arm_dir / "best.pt"
            if (state.get("identity") != run_identity or not checkpoint.is_file() or
                    state.get("best_sha256") != _sha_file(checkpoint)):
                raise ValueError(f"Completed run identity or checkpoint mismatch at {arm_dir}")
            continue
        result = _train_arm(replace(cfg, seed=seed), ds, model, arm, parts, post, banks, pre_files,
                            device, training_root(root, seed), run_identity, amp, check)
        if result["status"] == "budget_limited":
            raise BudgetReached(result["allocated_gpu_hours"], cfg.gpu_hour_limit)


def evaluate(cfg, root, ds, runs, device, settings, check, access_only=False):
    split = settings["split"]
    if split not in ("valid", "test") or (split == "test" and not settings.get("allow_test", False)):
        raise ValueError("Evaluation requires split=valid or explicit allow_test=true for Test")
    if access_only and split != "valid":
        raise ValueError("Deleted-ID sensitivity is a validation diagnostic")
    out = root / ("access" if access_only else "predictions") / split / ds
    suites = tuple(settings["suites"])
    if not access_only and (not suites or not set(suites) <= set(SUITES) or len(set(suites)) != len(suites)):
        raise ValueError(f"Select distinct evaluation suites from {SUITES}")
    parts, raw, _, csv_path, pkl_hash, _ = load_splits(cfg, ds, out, (split,))
    part = parts[split]
    bert, tokenizer, meta = _load_hf(cfg, ds, device)
    _tokenizer_audit(tokenizer, raw[split], part, ds, split)
    del raw
    scaler_path = root / "data" / ds / "train_scalers.npz"
    scaler = _scalers(scaler_path)
    expected = {"pkl_sha256": pkl_hash, "csv_sha256": _sha_file(csv_path), "scalers_sha256": _sha_file(scaler_path),
                "bert_files": meta["local_files_sha256"], "code": code_hashes(),
                "dataset": ds}
    checkpoints = {}
    for model, arm, seed in runs:
        parent = arm_directory(root, ds, model, arm, seed)
        path = parent / "best.pt"
        marker = json.loads((parent / "complete.json").read_text(encoding="utf-8"))
        identity = marker["identity"]
        if (any(identity.get(k) != v for k, v in expected.items()) or
                (identity["model"], identity["arm"], identity["seed"]) != (model, arm, seed) or
                identity["training"]["mask_seed"] != cfg.mask_seed or _sha_file(path) != marker["best_sha256"]):
            raise ValueError(f"Checkpoint, input, or training identity mismatch: {path}")
        checkpoints[model, arm, seed] = path, marker
    _normalise_data(parts, scaler, skip_text=True)
    raw_post = _bert_forward(bert, part.token_ids, part.token_attention, part.token_types, device,
                             batch_size=cfg.bert_batch_size, budget_check=check)
    if raw_post.shape != part.feat["text"].shape or not np.isfinite(raw_post).all():
        raise ValueError(f"{ds}/{split}: invalid BERT output")
    content = part.vpos["text"] & part.available["text"][:, None]
    post = (raw_post - scaler["text"]["mean"]) / scaler["text"]["std"]
    post[~content] = 0
    post = post.astype(np.float16)
    if not np.isfinite(post).all():
        raise ValueError("Normalized POST features exceed float16 range")
    if not access_only:
        del raw_post
    random_bank = custom_banks(part, ds, split, cfg.mask_seed)
    dims = {m: x.shape[2] for m, x in part.feat.items()}
    lengths = {m: x.shape[1] for m, x in part.feat.items()}

    def models():
        for (name, arm, seed), (path, marker) in checkpoints.items():
            check()
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state.get("protocol") != PROTOCOL or state.get("identity") != marker["identity"]:
                raise ValueError(f"Wrong selected checkpoint: {path}")
            model = build_model(name, dims, lengths, marker["identity"]["train_label_median"])
            model.load_state_dict(state["model"])
            yield model.to(device).eval(), {"dataset": ds, "model": name, "arm": arm, "seed": seed}

    manifest = {"split": split, "mask_seed": cfg.mask_seed, "new_mask_seed": settings["new_mask_seed"],
                "replacement_draws": settings["replacement_draws"], "bert_batch_size": cfg.bert_batch_size,
                "checkpoints": {f"{m}/{a}/{s}": x[1]["best_sha256"] for (m, a, s), x in checkpoints.items()}}
    out.mkdir(parents=True, exist_ok=True)
    path = out / "evaluation_identity.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError(f"Evaluation settings changed. Use a separate results directory from {out}")
    _json(path, manifest)
    if access_only:
        audit_access(bert, tokenizer, part, raw_post, random_bank, scaler["text"], models,
                     out, cfg, device, check, settings["replacement_draws"])
        return
    for suite in suites:
        bank = random_bank
        operators = {"PRE": "PRE"}
        repeats_exact = None
        if suite == "operators":
            operators.update(MASK_OFF="MASK_OFF", MASK_ON="MASK_ON")
        elif suite == "new_masks":
            bank = custom_banks(part, ds, split, settings["new_mask_seed"])
        elif suite == "blocks":
            bank, budgets = custom_banks(part, ds, split, settings["new_mask_seed"], "block", random_bank)
            _atomic_csv(out / suite / "token_budgets.csv", budgets)
            exact = pd.DataFrame(budgets).groupby("id").token_budget_exact.all()
            repeats_exact = set(exact[exact].index)
        # Release each temporary encoding after all checkpoints use it.
        evaluations = (["clean", "POST"] if suite == "standard" else ["POST"]) + list(operators)
        for name in evaluations:
            with TemporaryDirectory(dir=out) as scratch:
                files = encode_variant(bert, tokenizer, part, bank, scaler["text"], Path(scratch),
                                       operators[name], device, cfg, check) if name in operators else {}
                for model, record in models():
                    stem = f"{record['model']}_{record['arm']}_{record['seed']}"
                    rows = evaluate_condition(model, record, part, post, bank, files, cfg, device, name, check)
                    _atomic_csv(out / suite / f"{stem}_{name}.csv", rows)
                    if repeats_exact is not None:
                        exact_rows = [r for r in rows if r["id"] in repeats_exact]
                        if exact_rows:
                            _atomic_csv(out / suite / "exact_budget" / f"{stem}_{name}.csv", exact_rows)


def analyze(root, datasets, settings, runs):
    identifiers = [f"{model}_{arm}_{seed}" for model, arm, seed in runs]
    for ds in datasets:
        prediction_dir = root / "predictions" / settings["split"] / ds
        for suite in settings["suites"]:
            for subset in (("", "exact_budget") if suite == "blocks" else ("",)):
                inputs = prediction_dir / suite / subset
                if subset and not inputs.is_dir():
                    continue
                names = ["clean", "POST", "PRE"] if suite == "standard" else ["POST", "PRE"]
                if suite == "operators":
                    names += ["MASK_OFF", "MASK_ON"]
                files = [inputs / f"{stem}_{name}.csv" for stem in identifiers for name in names]
                frame = pd.concat((pd.read_csv(p, dtype={"id": str, "source_id": str}) for p in files), ignore_index=True)
                target = root / "analysis" / settings["split"] / ds / suite / subset
                _json(target / "summary.json", summarize(frame, target, bootstrap=settings["bootstrap_replicates"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate", "audit-access", "analyze", "selfcheck"))
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    if args.action == "selfcheck":
        from checks.smoke import selfcheck
        selfcheck()
        return
    raw_config = args.config.read_text(encoding="utf-8")
    try:
        settings = json.loads(raw_config)
    except json.JSONDecodeError:
        import yaml
        settings = yaml.safe_load(raw_config)
    protocol = settings["protocol"]
    if tuple(protocol["delete_rates"]) != RATES or protocol["mask_repeats"] != N_MASKS:
        raise ValueError("The paper protocol uses 20% and 50% deletion, with eight masks per rate")
    cfg = Config(**settings["runtime"])
    if not (0 <= cfg.clean_probability <= 1 and 0 <= cfg.warmup_epochs <= cfg.epochs
            and 1 <= cfg.min_epochs <= cfg.epochs and cfg.patience >= 1
            and min(cfg.batch_size, cfg.eval_batch_size, cfg.bert_batch_size) >= 1 and cfg.workers >= 0):
        raise ValueError("Invalid training schedule, batch sizes, worker count, or clean probability")
    datasets, runs = settings["datasets"], list(matrix(settings["experiments"]))
    if (not datasets or not set(datasets) <= set(DATASETS) or len(set(datasets)) != len(datasets)
            or not runs or len(set(runs)) != len(runs)):
        raise ValueError("Select known datasets and distinct model/condition/seed runs")
    if (settings["evaluation"]["new_mask_seed"] == cfg.mask_seed
            or settings["evaluation"]["bootstrap_replicates"] < 1):
        raise ValueError("Use a distinct seed for new masks and a positive bootstrap count")
    root = Path(cfg.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache.configure(**settings["storage"], quota_root=root)
    if args.action == "analyze":
        analyze(root, datasets, settings["evaluation"], runs)
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    count = torch.cuda.device_count()
    if cfg.allocated_gpu_count not in (0, count):
        raise ValueError(f"Configured GPU allocation disagrees with {count} visible GPU(s)")
    started = time.perf_counter()

    def check(force=False):
        return _budget_check(cfg, started, allocated_gpu_count=count)

    for ds in datasets:
        if args.action == "train":
            train(cfg, root, ds, runs, device, check)
        else:
            evaluate(cfg, root, ds, runs, device, settings["evaluation"], check,
                     access_only=args.action == "audit-access")


if __name__ == "__main__":
    main()
