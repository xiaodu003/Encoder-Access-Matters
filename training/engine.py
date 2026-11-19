"""Train the downstream predictor and resume from validated checkpoints."""
import json, math, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from common import Config, PROTOCOL, RATES, BudgetReached, _seed, _json, _atomic_csv, _atomic_torch_save, _sha_array, _sha_file
from data import features as core, cache as storage
from data.batches import MaskedSentimentDataset, _loader, _generate_train_schedule
from evaluation.metrics import _evaluate, _composite_metrics, _selection_score
from models.fusion import build_model

def _sync_time(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _lr_lambda(step: int, total_steps: int, warmup_steps: int):
    if step < warmup_steps:
        return max((step + 1) / max(1, warmup_steps), 1e-8)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return max(0.5 * (1.0 + math.cos(math.pi * progress)), 0.0)


def _checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_score, stale,
                        history, init_seed, identity):
    return {"protocol": PROTOCOL, "identity": identity, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "epoch": int(epoch), "best_score": float(best_score),
            "stale": int(stale), "history": history, "init_seed": int(init_seed),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_checkpoint(path, model, optimizer, scheduler, scaler, identity, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("protocol") != PROTOCOL or state.get("identity") != identity:
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    return state


def _save_predictions(out: Path, key: str, rows):
    _atomic_csv(out / f"predictions_{key}.csv", rows,
                columns=["dataset", "model", "arm", "checkpoint", "condition", "id", "source_id", "target", "prediction", "abs_error"])


def _train_arm(cfg: Config, ds: str, model_name: str, arm: str, parts, post, banks,
               pre_files, device, output: Path, identity, amp_enabled, budget_check):
    train, valid = parts["train"], parts["valid"]
    train_schedule = _generate_train_schedule(train, ds, cfg)
    arm_dir = output / "configs" / ds / f"{model_name}_{arm}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    pre_cache = {r: storage.load(pre_files["train"][r]) for r in RATES}
    valid_pre = {r: storage.load(pre_files["valid"][r]) for r in RATES}
    train_delete = banks["train"]
    valid_delete = banks["valid"]
    model_seed = core.stable_seed(PROTOCOL, cfg.seed, ds, model_name)
    _seed(model_seed)
    model = build_model(model_name, {m: int(train.feat[m].shape[-1]) for m in core.MODALITIES},
                        {m: int(train.feat[m].shape[1]) for m in core.MODALITIES}, float(np.median(train.y))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=(cfg.almt_learning_rate if model_name == "ALMT_FEATURE" else cfg.learning_rate), weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train.y) / cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _lr_lambda(step, total_steps, warmup_steps))
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, growth_interval=1000,
                                  enabled=amp_enabled)
    last_path = arm_dir / "last.pt"
    best_path = arm_dir / "best.pt"
    history_path = arm_dir / "train_history.csv"
    history = []
    start_epoch, best_score, stale = 0, float("inf"), 0
    if last_path.exists():
        state = _restore_checkpoint(last_path, model, optimizer, scheduler, scaler, identity, device)
        start_epoch, best_score, stale, history = state["epoch"] + 1, state["best_score"], state["stale"], state["history"]
        if start_epoch >= cfg.min_epochs and stale >= cfg.patience:
            start_epoch = cfg.epochs
        if best_path.exists():
            best_state = torch.load(best_path, map_location=device, weights_only=False)
            if best_state.get("identity") != identity:
                raise ValueError(f"Best checkpoint identity mismatch: {best_path}")
    else:
        _atomic_torch_save(arm_dir / "initial.pt", {"identity": identity, "model": model.state_dict(), "seed": model_seed})
        initial_rows, initial_preds = _evaluate(model, model_name, arm, valid, ds, post["valid"], valid_delete,
                                                valid_pre, cfg, device, full=True, prefix="initial",
                                                budget_check=budget_check)
        _atomic_csv(arm_dir / "valid_metrics_initial.csv", initial_rows)
        _save_predictions(arm_dir, "initial", initial_preds)
    _json(arm_dir / "input_schedule_identity.json", {
        "schedule": {k: _sha_array(v) for k, v in train_schedule.items()},
        "sample_order_seeds": [core.stable_seed(PROTOCOL, cfg.seed, ds, "order", ep) for ep in range(cfg.epochs)],
        "train_ids_sha": _sha_array(np.asarray(train.ids, dtype='U'))})
    amp_mode = "fp16_dynamic_scaler" if amp_enabled else "fp32"
    started = time.perf_counter()
    device_count = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 0
    end_epoch = start_epoch if start_epoch >= cfg.min_epochs and stale >= cfg.patience else cfg.epochs
    for epoch in range(start_epoch, end_epoch):
        epoch_started = _sync_time(device)
        model.train()
        dataset = MaskedSentimentDataset(train, ds, model_name, arm, post["train"], train_delete,
                             pre_cache, training=True, epoch=epoch, schedule=train_schedule)
        loader = _loader(dataset, cfg, shuffle=True,
                         seed=core.stable_seed(PROTOCOL, cfg.seed, ds, "order", epoch))
        loss_sum, seen, amp_overflow_batches = 0.0, 0, 0
        pbar = tqdm(loader, desc=f"{ds} {model_name}/{arm} {epoch+1}/{cfg.epochs}", leave=False, dynamic_ncols=True)
        for batch in pbar:
            b = core.move_batch(batch)
            target = b["target"].float()
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(b).float()
                    loss = F.l1_loss(pred, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{ds}/{model_name}/{arm}: nonfinite training loss")
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradients_finite = all(p.grad is None or torch.isfinite(p.grad).all()
                                       for p in model.parameters())
                if gradients_finite:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                else:
                    amp_overflow_batches += 1
                    # GradScaler records the overflow during unscale_ and skips this update.
                scaler.step(optimizer)
                scaler.update()
                stepped = gradients_finite and scaler.get_scale() >= scale_before
            else:
                pred = model(b).float()
                loss = F.l1_loss(pred, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{ds}/{model_name}/{arm}: nonfinite training loss")
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError(f"{ds}/{model_name}/{arm}: nonfinite FP32 gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scale_before = None
                optimizer.step()
                stepped = True
            if stepped:
                scheduler.step()
            count = int(len(target))
            loss_sum += float(loss.detach().cpu()) * count
            seen += count
            pbar.set_postfix(loss=f"{loss_sum/max(1,seen):.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                             amp_skip=amp_overflow_batches)
        model.eval()
        training_seconds = _sync_time(device) - epoch_started
        selection_rows, selection_predictions = _evaluate(model, model_name, arm, valid, ds, post["valid"], valid_delete,
                                      valid_pre, cfg, device, full=True, prefix=f"epoch_{epoch+1}",
                                      budget_check=None)
        _atomic_csv(arm_dir / f"metrics_epoch_{epoch+1:02d}.csv", selection_rows)
        if epoch + 1 in (10, 20, 40):
            _save_predictions(arm_dir, f"epoch_{epoch+1}", selection_predictions)
        composite = _composite_metrics(selection_rows)
        score = composite["selection_score"]
        if not math.isfinite(score):
            raise FloatingPointError(f"{ds}/{model_name}/{arm}: nonfinite validation selection score")
        improved = score < best_score - 1e-12
        if improved:
            best_score, stale = score, 0
            _atomic_torch_save(best_path, {"protocol": PROTOCOL, "identity": identity,
                                           "model": model.state_dict(), "epoch": epoch + 1,
                                           "selection_score": score, "metrics": composite})
        else:
            stale += 1
        row = {"dataset": ds, "model": model_name, "arm": arm, "epoch": epoch + 1,
               "train_l1": loss_sum / max(1, seen), **composite,
               "lr": optimizer.param_groups[0]["lr"], "improved": improved,
               "amp": amp_mode, "amp_overflow_batches": amp_overflow_batches,
               "amp_scale": float(scaler.get_scale()) if amp_enabled else None,
               "epoch_seconds": _sync_time(device) - epoch_started,
               "training_seconds": training_seconds,
               "validation_seconds": _sync_time(device) - epoch_started - training_seconds,
               "cumulative_seconds": time.perf_counter() - started,
               "successful_optimizer_steps": int(scheduler.last_epoch),
               "early_stop_stale": stale}
        history.append(row)
        _atomic_csv(history_path, history)
        if epoch + 1 in (10, 20, 40):
            _atomic_torch_save(arm_dir / f"epoch_{epoch+1}.pt",
                               {"protocol": PROTOCOL, "identity": identity,
                                "model": model.state_dict(), "epoch": epoch + 1})
        _atomic_torch_save(last_path, _checkpoint_payload(model, optimizer, scheduler, scaler,
                                                           epoch, best_score, stale, history,
                                                           model_seed, identity))
        remaining = max(0, cfg.epochs - epoch - 1)
        avg_epoch = (time.perf_counter() - started) / max(1, epoch - start_epoch + 1)
        eta = remaining * avg_epoch
        amp_status = (f"amp_skip={amp_overflow_batches} scale={scaler.get_scale():.0f}"
                      if amp_enabled else "amp_skip=0 scale=off")
        print(f"[{ds} {model_name}/{arm}] {epoch+1}/{cfg.epochs} trainL1={row['train_l1']:.4f} "
              f"clean={row['clean_mae']:.4f} PRE={row['pre_mae']:.4f} POST={row['post_mae']:.4f} "
              f"score={score:.4f} Corr={row.get('clean_corr', float('nan')):.4f} "
              f"Acc2={row.get('clean_acc2', float('nan')):.3f} F1={row.get('clean_f1', float('nan')):.3f} "
              f"{amp_status} epoch={row['epoch_seconds']:.1f}s ETA={eta/60:.1f}m")
        try:
            allocated = budget_check(force=True)
        except BudgetReached as exc:
            allocated = exc.used_gpu_hours
            _json(arm_dir / "status.json", {"status": "budget_limited", "last_epoch": epoch + 1,
                                             "allocated_gpu_hours": allocated, "resume_from": str(last_path)})
            return {"status": "budget_limited", "last_epoch": epoch + 1, "allocated_gpu_hours": allocated}
        if epoch + 1 >= cfg.min_epochs and stale >= cfg.patience:
            break
    # Export last before loading best; both are true same-state FP32 evaluations.
    last_rows, last_preds = _evaluate(model, model_name, arm, valid, ds, post["valid"], valid_delete,
                                      valid_pre, cfg, device, full=True, prefix="last", budget_check=budget_check)
    _atomic_csv(arm_dir / "valid_metrics_last.csv", last_rows)
    _save_predictions(arm_dir, "last", last_preds)
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    full_rows, full_predictions = _evaluate(model, model_name, arm, valid, ds, post["valid"], valid_delete,
                                            valid_pre, cfg, device, full=True, prefix="best",
                                            budget_check=budget_check)
    _atomic_csv(arm_dir / "valid_metrics_best.csv", full_rows)
    _save_predictions(arm_dir, "best", full_predictions)
    checkpoint_epochs = {10, 20, 40}
    epochs_to_report = sorted(e for e in checkpoint_epochs if e <= len(history))
    all_epoch_rows = []
    for ep in epochs_to_report:
        state_path = arm_dir / f"epoch_{ep}.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing required fixed-epoch checkpoint: {state_path}")
        snap = torch.load(state_path, map_location=device, weights_only=False)
        if snap.get("identity") != identity:
            raise ValueError(f"Fixed-epoch checkpoint identity mismatch: {state_path}")
        model.load_state_dict(snap["model"])
        rows, _ = _evaluate(model, model_name, arm, valid, ds, post["valid"], valid_delete,
                            valid_pre, cfg, device, full=False, prefix=f"epoch_{ep}",
                            budget_check=budget_check)
        all_epoch_rows.extend(rows)
    model.load_state_dict(best["model"])
    _atomic_csv(arm_dir / "checkpoint_epoch_metrics.csv", all_epoch_rows)
    _json(arm_dir / "complete.json", {"status": "completed_exploratory", "identity": identity,
                                       "best_sha256": _sha_file(best_path),
                                       "best_epoch": int(best["epoch"]), "best_selection_score": float(best["selection_score"]),
                                       "last_epoch": int(history[-1]["epoch"]), "amp": amp_mode,
                                       "amp_overflow_batches_total": int(sum(
                                           int(row.get("amp_overflow_batches", 0)) for row in history)),
                                       "test_inference": False, "full_valid_conditions": len(full_rows),
                                       "elapsed_seconds": time.perf_counter() - started,
                                       "fixed_epochs_not_reached": [e for e in (10, 20, 40) if e > len(history)]})
    return {"status": "completed_exploratory", "best_epoch": int(best["epoch"]),
            "best_score": float(best["selection_score"]), "last_epoch": int(history[-1]["epoch"]),
            "amp": amp_mode}


def _amp_gate(ds, train, post, banks, pre_files, cfg, device):
    if not cfg.use_amp or device.type != "cuda":
        return False, "disabled_by_config_or_no_cuda"
    saved = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    try:
        dataset = MaskedSentimentDataset(train, ds, "B1_FUSION", "POST", post["train"], banks["train"],
                             {r: storage.load(pre_files["train"][r]) for r in RATES},
                             training=True, epoch=0, schedule=_generate_train_schedule(train, ds, cfg))
        b = core.move_batch(core.collate_batch([dataset[i] for i in range(min(4, len(dataset)))]))
        for name in cfg.models:
            model = build_model(name, {m: train.feat[m].shape[-1] for m in core.MODALITIES},
                                {m: train.feat[m].shape[1] for m in core.MODALITIES}, float(np.median(train.y))).to(device).train()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.l1_loss(model(b).float(), b["target"].float())
            (loss * 1024).backward()
            if not bool(torch.isfinite(loss)) or not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError(f"{name}: scaled backward failed")
            del model
        return True, "all_selected_models_passed_fp16_scaled_backward"
    except Exception as exc:
        return False, f"whole_dataset_fallback:{type(exc).__name__}:{exc}"
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])
        torch.cuda.set_rng_state_all(saved[3])
        torch.cuda.empty_cache()
