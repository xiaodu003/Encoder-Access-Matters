"""Position-matched PRE/POST encoding with frozen BERT and train-only scalers."""
import json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from common import Config, PROTOCOL, RATES, N_MASKS, _json, _atomic_csv, _sha_file, _sha_array, _fingerprint
from data import features as core, cache as storage
from models.fusion import build_model

def _normalise_mode(value: str) -> str:
    x = str(value).strip().lower()
    return {"val": "valid", "dev": "valid", "validation": "valid"}.get(x, x)


def _text_token_arrays(raw_split, n: int, length: int):
    packed, key = core.dict_get_ci(raw_split, ("text_bert", "text_tokens", "bert_inputs"))
    if packed is None:
        raise KeyError("PRE encoding requires text_bert token IDs from the same dataset rows")
    a = np.asarray(packed)
    if a.ndim != 3 or a.shape[0] != n or a.shape[1] not in (2, 3) or a.shape[2] != length:
        raise ValueError(f"{key}: expected (N,2/3,L)=({n},2/3,{length}), got {a.shape}")
    if not np.isfinite(a).all() or not np.equal(a, np.rint(a)).all():
        raise ValueError(f"{key}: token IDs, attention mask and token types must be finite integers")
    ids = a[:, 0, :].astype(np.int64)
    attention = a[:, 1, :].astype(np.int64)
    types = a[:, 2, :].astype(np.int64) if a.shape[1] == 3 else np.zeros_like(ids)
    if not np.isin(attention, (0, 1)).all():
        raise ValueError(f"{key}: attention mask is not binary")
    return ids, attention, types, key


def _raw_texts(raw_split, n: int):
    texts, key = core.dict_get_ci(raw_split, ("raw_text", "text_raw", "transcript", "utterance"))
    if texts is None or len(texts) != n:
        raise ValueError("Could not align raw_text with text_bert rows; refusing to guess tokenizer compatibility")
    out = []
    for value in texts:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, (list, tuple, np.ndarray)):
            value = " ".join(str(x) for x in np.asarray(value, dtype=object).reshape(-1))
        if not isinstance(value, str):
            raise ValueError(f"{key} contains a non-text value ({type(value).__name__})")
        out.append(value)
    return out


def _token_position_groups(valid_positions: np.ndarray) -> list[np.ndarray]:
    """Chinese SIMS masks use the dataset's existing BERT token positions as units."""
    return [np.asarray([pos], dtype=np.int64) for pos in np.flatnonzero(valid_positions)]


def _tokenizer_audit(tokenizer, raw_split, part, ds: str, split_name: str) -> dict:
    texts = _raw_texts(raw_split, len(part.y))
    width = part.feat["text"].shape[1]
    token_ids = part.token_ids
    attn = part.token_attention
    seg = part.token_types
    if token_ids.shape != (len(texts), width) or attn.shape != token_ids.shape:
        raise ValueError(f"{ds}/{split_name}: text_bert and cached text feature positions disagree")
    if np.max(token_ids, initial=0) >= len(tokenizer) or np.min(token_ids, initial=0) < 0:
        raise ValueError(f"{ds}/{split_name}: token IDs exceed tokenizer vocabulary")
    if tokenizer.cls_token_id != 101 or tokenizer.sep_token_id != 102 or tokenizer.pad_token_id != 0:
        raise ValueError(f"{ds}: expected verified BERT [CLS]=101 [SEP]=102 [PAD]=0 token IDs")
    if not np.array_equal(part.vpos["text"], attn.astype(bool) & (token_ids != 101) & (token_ids != 102)):
        raise ValueError(f"{ds}/{split_name}: text content positions do not match token ID/attention audit")
    encoded = tokenizer(
        texts,
        max_length=width,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_offsets_mapping=True,
    )
    got_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    got_attn = np.asarray(encoded["attention_mask"], dtype=np.int64)
    got_types = np.asarray(encoded.get("token_type_ids", np.zeros_like(got_ids)), dtype=np.int64)
    attention_mismatch = np.flatnonzero(np.any(got_attn != attn, axis=1))
    type_mismatch = np.flatnonzero(np.any(got_types != seg, axis=1))
    id_mismatch = np.flatnonzero(np.any(got_ids != token_ids, axis=1))
    structure_rows = sorted(set(attention_mismatch.tolist()) | set(type_mismatch.tolist()))
    structure_tolerated = []
    # core.get_text_positions already guarantees the stored attention mask is a contiguous
    # prefix with no attended PAD, and the fresh mask is contiguous by construction. So for a
    # single-sentence BERT input the only way an attention difference can exist is that the two
    # sides cover a different number of valid positions, i.e. a tail length shift. That is the
    # upstream-preprocessing artefact seen on SIMS_V1/train row 208 (stored 37 valid vs round
    # trip 38, with the stored [SEP] sitting where the round trip still has content).
    stored_len = attn.sum(axis=1).astype(np.int64)
    fresh_len = got_attn.sum(axis=1).astype(np.int64)
    length_shift = stored_len != fresh_len
    # Positions where the round trip disagrees with the stored IDs outside the stored content
    # region, i.e. at a special token or at padding.
    id_diff_outside_content = (got_ids != token_ids) & ~(part.vpos["text"] & attn.astype(bool))
    if ds != "SIMS_V1":
        if structure_rows:
            i = structure_rows[0]
            raise ValueError(
                f"{ds}/{split_name}: tokenizer sequence structure mismatch at row {i}; "
                f"token_id_diffs={int(np.sum(got_ids[i] != token_ids[i]))}, "
                f"attention_diffs={int(np.sum(got_attn[i] != attn[i]))}, "
                f"type_diffs={int(np.sum(got_types[i] != seg[i]))}. "
                "The stored text_bert positions cannot be aligned safely."
            )
        if np.any(id_diff_outside_content):
            row, pos = np.argwhere(id_diff_outside_content)[0]
            raise ValueError(
                f"{ds}/{split_name}: token-ID mismatch touches a special/padding position "
                f"(row={int(row)}, position={int(pos)}); refusing to continue."
            )
    else:
        # SIMS keeps the stored text_bert arrays as the canonical model input: never rewritten,
        # never remapped. A row is tolerated only when its valid length differs from the round
        # trip, which proves the mismatch is a tail boundary artefact and not an interior shift.
        same_length_structure = [int(r) for r in structure_rows if not bool(length_shift[r])]
        if same_length_structure:
            row = same_length_structure[0]
            raise ValueError(
                f"{ds}/{split_name}: structure mismatch at row {row} ({part.ids[row]}) with "
                f"identical valid length {int(stored_len[row])}; a contiguous mask cannot differ "
                f"without an interior shift. attention_diffs="
                f"{int(np.sum(got_attn[row] != attn[row]))}, "
                f"type_diffs={int(np.sum(got_types[row] != seg[row]))}. Refusing to realign."
            )
        outside_rows = sorted({int(r) for r, _ in np.argwhere(id_diff_outside_content)})
        same_length_outside = [r for r in outside_rows if not bool(length_shift[r])]
        if same_length_outside:
            row = same_length_outside[0]
            pos = int(np.flatnonzero(id_diff_outside_content[row])[0])
            raise ValueError(
                f"{ds}/{split_name}: token-ID mismatch touches a special position at row {row}, "
                f"position {pos} (stored={int(token_ids[row, pos])}, "
                f"roundtrip={int(got_ids[row, pos])}) while both sequences have the same valid "
                f"length {int(stored_len[row])}; refusing to continue."
            )
        shifted = int(np.count_nonzero(length_shift))
        # Corpus-level guard: a wrong checkpoint shifts every row, a few boundary rows do not.
        if shifted > max(1, int(0.25 * len(texts))):
            raise ValueError(
                f"{ds}/{split_name}: {shifted}/{len(texts)} rows show a stored-vs-roundtrip valid "
                "length shift; that is too many for isolated upstream preprocessing artefacts and "
                "points at the wrong tokenizer/checkpoint. Refusing to continue."
            )
        for row in structure_rows:
            row = int(row)
            diff = np.flatnonzero((got_attn[row] != attn[row]) | (got_types[row] != seg[row]))
            structure_tolerated.append({
                "row": row, "sample_id": str(part.ids[row]), "reason": "valid_length_shift",
                "positions": diff.astype(int).tolist(),
                "stored_valid_len": int(stored_len[row]),
                "roundtrip_valid_len": int(fresh_len[row]),
                "stored_ids": token_ids[row, diff].astype(int).tolist(),
                "roundtrip_ids": got_ids[row, diff].astype(int).tolist(),
            })
    if len(id_mismatch) and ds != "SIMS_V1":
        i = int(id_mismatch[0])
        raise ValueError(
            f"{ds}/{split_name}: tokenizer token-ID mismatch at row {i}; "
            f"token_id_diffs={int(np.sum(got_ids[i] != token_ids[i]))}. "
            "MOSI requires exact round-trip token IDs."
        )
    if ds != "SIMS_V1" and not getattr(tokenizer, "is_fast", False):
        raise ValueError(f"{ds}: a fast tokenizer is required for whole-word deletion")
    # The PKL's text_bert IDs are the actual model inputs. Chinese raw-text re-tokenization
    # can differ at isolated positions because the source text was preprocessed upstream;
    # do not replace or remap those IDs. Use token positions as Chinese mask units.
    mismatch_examples = []
    for i in id_mismatch[:10]:
        positions = np.flatnonzero(got_ids[i] != token_ids[i])
        mismatch_examples.append({
            "row": int(i), "sample_id": str(part.ids[i]),
            "positions": positions[:10].astype(int).tolist(),
            "stored_ids": token_ids[i, positions[:10]].astype(int).tolist(),
            "roundtrip_ids": got_ids[i, positions[:10]].astype(int).tolist(),
            "stored_tokens": tokenizer.convert_ids_to_tokens(token_ids[i, positions[:10]].astype(int).tolist()),
            "roundtrip_tokens": tokenizer.convert_ids_to_tokens(got_ids[i, positions[:10]].astype(int).tolist()),
        })
    content_groups = []
    for i in range(len(texts)):
        if ds == "SIMS_V1":
            groups = _token_position_groups(part.vpos["text"][i])
        else:
            grouped = {}
            word_ids = encoded.word_ids(batch_index=i)
            for pos, word_id in enumerate(word_ids):
                if word_id is not None and part.vpos["text"][i, pos]:
                    grouped.setdefault(int(word_id), []).append(pos)
            groups = [np.asarray(v, dtype=np.int64) for _, v in sorted(grouped.items())]
        flattened = np.sort(np.concatenate(groups)) if groups else np.empty(0, dtype=np.int64)
        if not np.array_equal(flattened, np.flatnonzero(part.vpos["text"][i])):
            raise ValueError(f"{ds}/{split_name}: mask units do not cover content token positions exactly")
        content_groups.append(groups)
    part.content_groups = content_groups
    part.content_unit_type = "source_token_position" if ds == "SIMS_V1" else "fast_tokenizer_word"
    return {"split": split_name, "samples_verified": len(texts), "tokenizer": tokenizer.name_or_path,
            "tokenizer_vocab_size": len(tokenizer), "max_length": width, "token_fields": part.token_key,
            "canonical_input_ids": "dataset_text_bert",
            "mask_unit_type": part.content_unit_type,
            "token_id_roundtrip_mismatch_rows": int(len(id_mismatch)),
            "token_id_roundtrip_mismatch_positions": int(np.count_nonzero(got_ids != token_ids)),
            "token_id_roundtrip_mismatch_examples": mismatch_examples,
            "token_id_boundary_only_positions": int(np.count_nonzero(id_diff_outside_content)),
            "sequence_structure_mismatches": int(len(structure_tolerated)),
            "sequence_structure_details": structure_tolerated[:20],
            "stored_arrays_rewritten": False,
            "mask_units": int(sum(map(len, content_groups))),
            "content_tokens": int(sum(len(g) for row in content_groups for g in row))}


def _legacy_feature_audit(old, fresh, valid_positions, chunk_rows=256) -> dict:
    if old.shape != fresh.shape:
        raise ValueError(f"Legacy text feature shape {old.shape} differs from fresh BERT output {fresh.shape}")
    count = 0
    abs_sum = sq_sum = dot_sum = old_sq = new_sq = 0.0
    max_abs = 0.0
    for start in range(0, len(old), chunk_rows):
        end = min(len(old), start + chunk_rows)
        mask = valid_positions[start:end]
        if not mask.any():
            continue
        x = np.asarray(old[start:end][mask], dtype=np.float64)
        y = np.asarray(fresh[start:end][mask], dtype=np.float64)
        d = y - x
        count += int(x.size)
        abs_sum += float(np.abs(d).sum())
        sq_sum += float(np.square(d).sum())
        max_abs = max(max_abs, float(np.abs(d).max(initial=0.0)))
        dot_sum += float(np.sum(x * y))
        old_sq += float(np.sum(x * x))
        new_sq += float(np.sum(y * y))
    return {"compared_values": count, "mean_abs_difference": abs_sum / max(1, count),
            "rmse": math.sqrt(sq_sum / max(1, count)), "max_abs_difference": max_abs,
            "cosine_similarity": dot_sum / max(math.sqrt(old_sq * new_sq), 1e-12),
            "used_for_training": False}


def _csv_label_audit(ds: str, parts: dict, csv_path: Path) -> dict:
    if ds == "SIMS_V1":
        meta, reference = core.read_sims_v1_labels(csv_path)
        if not reference.get("user_reference_match"):
            raise ValueError("SIMS_V1 label semantic hash differs from the approved reference")
        id_values = [core.canon_id_pair(a, b) for a, b in zip(meta["video_id"], meta["clip_id"])]
        target_name, mode_values = "label", meta["mode"].astype(str).tolist()
        raw_meta = meta
    else:
        raw_meta = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        lower = {str(c).strip().casefold(): c for c in raw_meta.columns}
        if "video_id" in lower and "clip_id" in lower:
            id_values = [core.canon_id_pair(a, b) for a, b in zip(raw_meta[lower["video_id"]], raw_meta[lower["clip_id"]])]
        elif "id" in lower:
            id_values = raw_meta[lower["id"]].astype(str).str.strip().tolist()
        else:
            raise ValueError(f"{ds}: label CSV has no recognized sample-ID columns")
        target_name = next((lower[k] for k in ("label", "regression_labels", "sentiment_score", "sentiment") if k in lower), None)
        if target_name is None:
            raise ValueError(f"{ds}: label CSV has no regression target column")
        mode_name = next((lower[k] for k in ("mode", "split", "partition") if k in lower), None)
        mode_values = raw_meta[mode_name].astype(str).tolist() if mode_name else None
    if len(set(id_values)) != len(id_values):
        raise ValueError(f"{ds}: duplicate IDs in label CSV")
    lookup = {str(k): j for j, k in enumerate(id_values)}
    errors, rows = [], []
    for split_name, part in parts.items():
        matched = 0
        for i, sample_id in enumerate(part.ids):
            choices = [core.canonical_pkl_id(str(sample_id), "literal"),
                       core.canonical_pkl_id(str(sample_id), "underscore_tail")]
            key = next((k for k in choices if k in lookup), None)
            if key is None:
                errors.append(f"{split_name}/{sample_id}: missing CSV ID")
                continue
            j = lookup[key]
            try:
                label = float(raw_meta.iloc[j][target_name])
            except (TypeError, ValueError):
                errors.append(f"{split_name}/{sample_id}: invalid CSV label")
                continue
            if not math.isfinite(label) or abs(label - float(part.y[i])) > core.CSV_LABEL_ABS_TOL:
                errors.append(f"{split_name}/{sample_id}: CSV/PKL label mismatch")
            if mode_values is not None and _normalise_mode(mode_values[j]) != split_name:
                errors.append(f"{split_name}/{sample_id}: CSV/PKL split mismatch")
            matched += 1
        rows.append({"split": split_name, "samples_checked": len(part.y), "matched": matched})
    if errors:
        raise ValueError("CSV identity audit failed: " + "; ".join(errors[:5]))
    return {"status": "passed", "csv_sha256": _sha_file(csv_path), "target_column": str(target_name),
            "rows": rows}


def load_splits(cfg: Config, ds: str, root: Path, splits=("train", "valid")):
    core.INPUT_ROOT = Path(cfg.input_root)
    core.OUTPUT_ROOT = root / "data_audit"
    if cfg.dataset_paths.get(ds, {}).get("pkl"):
        pkl_path = Path(cfg.dataset_paths[ds]["pkl"])
    else:
        pkl_path = core.resolve_known_input(ds, "pkl", required=True)
    if cfg.dataset_paths.get(ds, {}).get("csv"):
        csv_path = Path(cfg.dataset_paths[ds]["csv"])
    else:
        csv_path = core.resolve_known_input(ds, "csv", required=True)
    if not pkl_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError(f"Missing source file for {ds}: {pkl_path} / {csv_path}")
    pkl_hash = _sha_file(pkl_path)
    expected = core.DATASET_FILES[ds]["expected_sha256"]
    if expected and pkl_hash != expected:
        raise ValueError(f"{ds}: raw PKL SHA-256 mismatch; refusing a different feature release")
    raw = core.read_trusted_pkl(pkl_path)
    parts, raw_splits = {}, {}
    for name in splits:
        raw_split = core.get_split(raw, name)
        part = core.parse_split(name, raw_split, ds)
        token_ids, token_attention, token_types, token_key = _text_token_arrays(
            raw_split, len(part.y), part.feat["text"].shape[1]
        )
        part.token_ids = token_ids
        part.token_attention = token_attention
        part.token_types = token_types
        part.token_key = token_key
        parts[name] = part
        raw_splits[name] = raw_split
    _csv_label_audit(ds, parts, csv_path)
    ids_sha = {s: _sha_array(np.asarray(p.ids, dtype="U")) for s, p in parts.items()}
    return parts, raw_splits, pkl_path, csv_path, pkl_hash, ids_sha


def _load_hf(cfg: Config, ds: str, device):
    from transformers import AutoModel, AutoTokenizer

    model_path = cfg.bert_paths.get(ds, "") 
    if not model_path:
        raise ValueError(f"Set bert_paths.{ds} in config.yaml")
    local = Path(model_path).expanduser()
    if local.is_dir():
        # Read the local model and tokenizer together.
        if not (local / "config.json").is_file():
            raise FileNotFoundError(
                f"{ds}: {local} is not a Hugging Face BERT folder (config.json missing); "
                "mount the full model+tokenizer directory and set bert_paths in config.yaml"
            )
        tokenizer = AutoTokenizer.from_pretrained(str(local), use_fast=True, local_files_only=True)
        model = AutoModel.from_pretrained(str(local), local_files_only=True)
        model_source = "local_dir"
    else:
        raise FileNotFoundError(f"{ds}: local BERT directory not mounted: {model_path}; set bert_paths in config.yaml")
    if model.config.model_type != "bert":
        raise ValueError(f"Expected BERT checkpoint for {ds}, got {model.config.model_type}")
    if int(model.config.vocab_size) != len(tokenizer):
        raise ValueError(f"{ds}: model/tokenizer vocabulary size differs; refusing token-ID remapping")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval().to(device)
    commit = getattr(model.config, "_commit_hash", None)
    tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    return model, tokenizer, {"model_path": model_path, "model_source": model_source,
                              "local_files_sha256": {p.relative_to(local).as_posix(): _sha_file(p)
                                  for p in sorted(local.rglob('*')) if p.is_file()
                                  and p.suffix in ('.json', '.txt', '.bin', '.safetensors', '.model')},
                              "resolved_commit": commit, "tokenizer_resolved_commit": tokenizer_commit,
                              "model_type": model.config.model_type,
                              "model_vocab_size": int(model.config.vocab_size),
                              "tokenizer_vocab_size": len(tokenizer),
                              "special_tokens": {"cls": tokenizer.cls_token_id, "sep": tokenizer.sep_token_id,
                                                 "pad": tokenizer.pad_token_id},
                              "license_source": "Hugging Face google-bert model card"}


@torch.inference_mode()
def _bert_forward(model, ids, attention, token_types, device, batch_size=32, budget_check=None):
    pieces = []
    for start in tqdm(range(0, len(ids), batch_size), desc="Frozen BERT encode", leave=False, dynamic_ncols=True):
        if budget_check is not None:
            budget_check()
        end = min(len(ids), start + batch_size)
        x = torch.as_tensor(ids[start:end], dtype=torch.long, device=device)
        a = torch.as_tensor(attention[start:end], dtype=torch.long, device=device)
        t = torch.as_tensor(token_types[start:end], dtype=torch.long, device=device)
        out = model(input_ids=x, attention_mask=a, token_type_ids=t, return_dict=True).last_hidden_state
        pieces.append(out.float().cpu().numpy())
    return np.concatenate(pieces, axis=0)


def _delete_bank(part, ds: str, split_name: str, seed: int) -> dict[float, np.ndarray]:
    valid = part.vpos["text"] & part.available["text"][:, None]
    banks = {}
    for rate in RATES:
        masks = np.zeros((N_MASKS, *valid.shape), dtype=bool)
        for repeat in range(N_MASKS):
            for i, sample_id in enumerate(part.ids):
                groups = [g for g in part.content_groups[i] if len(g) and np.all(valid[i, g])]
                if not np.array_equal(np.sort(np.concatenate(groups)) if groups else np.empty(0, dtype=np.int64),
                                      np.flatnonzero(valid[i])):
                    raise ValueError(f"{ds}/{split_name}/{sample_id}: mask units do not match valid text positions")
                k = min(int(math.floor(float(rate) * len(groups))), max(0, len(groups) - 1))
                if len(groups) > 1 and rate > 0:
                    k = max(1, k)
                if k:
                    rng = core.rng_for(PROTOCOL, seed, ds, split_name, str(sample_id), rate, repeat)
                    selected = rng.choice(len(groups), size=k, replace=False)
                    positions = np.concatenate([groups[j] for j in selected])
                    masks[repeat, i, positions] = True
        if np.any(masks & ~valid[None, ...]):
            raise AssertionError("Text mask selected padding, special tokens, or unavailable positions")
        banks[rate] = masks
    return banks


def _mask_rate_audit(part, bank) -> dict:
    rows = {}
    valid = part.vpos["text"] & part.available["text"][:, None]
    token_den = valid.sum(axis=1)
    unit_den = np.asarray([
        sum(bool(np.all(valid[i, group])) for group in groups)
        for i, groups in enumerate(part.content_groups)
    ], dtype=np.int64)
    for rate, masks in bank.items():
        token_fraction = masks.sum(axis=2) / np.maximum(1, token_den[None, :])
        deleted_units = np.zeros((N_MASKS, len(part.y)), dtype=np.int64)
        for i, groups in enumerate(part.content_groups):
            for group in groups:
                if unit_den[i] and np.all(valid[i, group]):
                    deleted_units[:, i] += masks[:, i, group].all(axis=1)
        unit_fraction = deleted_units / np.maximum(1, unit_den[None, :])
        token_eligible = token_den > 0
        unit_eligible = unit_den > 0
        rows[str(rate)] = {"requested_rate": float(rate),
                           "mask_unit_type": getattr(part, "content_unit_type", "content_token"),
                           "mean_deleted_unit_fraction": float(unit_fraction[:, unit_eligible].mean()) if unit_eligible.any() else 0.0,
                           "mean_deleted_token_fraction": float(token_fraction[:, token_eligible].mean()) if token_eligible.any() else 0.0,
                           "zero_deletion_sample_repeats": int(np.count_nonzero(masks.sum(axis=2) == 0)),
                           "sample_repeats": int(masks.shape[0] * masks.shape[1]),
                           "eligible_unit_sample_repeats": int(masks.shape[0] * unit_eligible.sum()),
                           "eligible_token_sample_repeats": int(masks.shape[0] * token_eligible.sum())}
    return rows


def _save_post_raw(model, part, output: Path, device, batch_size=32, budget_check=None, persist=True):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "post_raw.npy"
    values = _bert_forward(model, part.token_ids, part.token_attention, part.token_types,
                           device, batch_size, budget_check=budget_check)
    if values.shape != part.feat["text"].shape or not np.isfinite(values).all():
        raise ValueError(f"BERT output shape/value mismatch: {values.shape} vs {part.feat['text'].shape}")
    # Baseline caching already retains values in RAM; its raw disk copy is never read.
    # Sensitivity diagnostics still request and receive the original raw .npy file.
    if persist:
        storage.reserve_file(output, values.size*4 + (1 << 20), "Raw BERT cache")
        if path.exists():
            path.unlink()
        np.save(path, values.astype(np.float32), allow_pickle=False)
    return path, values


def _build_pre_cache(model, part, banks, scalers, out: Path, device, bert_meta,
                     batch_size=32, budget_check=None):
    out.mkdir(parents=True, exist_ok=True)
    mu, sd = scalers["text"]["mean"], scalers["text"]["std"]
    n, length, dim = part.feat["text"].shape
    token_types = part.token_types
    made = {}
    content = part.vpos["text"] & part.available["text"][:, None]
    for rate in RATES:
        bank = banks[rate]
        path = out / f"pre_r{int(rate*100):02d}.npy"
        keep_all = content[None] & ~bank
        context = {"ids": _sha_array(part.token_ids), "attention": _sha_array(part.token_attention),
                   "types": _sha_array(token_types), "mask": _sha_array(bank),
                   "mean": _sha_array(mu), "std": _sha_array(sd), "bert": bert_meta,
                   "encoder_batch_size": batch_size}
        writer = storage.PackedWriter(path, keep_all, dim, context)
        try:
            for repeat in range(N_MASKS):
                deleted = bank[repeat]
                ids = part.token_ids.copy()
                attention = part.token_attention.copy()
                types = token_types.copy()
                ids[deleted] = 0
                attention[deleted] = 0
                types[deleted] = 0
                hidden = _bert_forward(model, ids, attention, types, device, batch_size,
                                       budget_check=budget_check)
                keep = content & ~deleted
                hidden = (hidden - mu[None, None, :]) / sd[None, None, :]
                hidden[~keep] = 0.0
                if not np.isfinite(hidden).all():
                    raise FloatingPointError(f"nonfinite PRE cache at {out.name}/{rate}/{repeat}")
                writer.write_repeat(repeat, hidden)
            stored = writer.finish()
        except Exception:
            writer.abort()
            raise
        made[rate] = {"path": str(path), "sha256": stored["data_sha256"], "dtype": "float16",
                      "shape": [N_MASKS, n, length, dim], "storage": stored}
    return made


def _normalise_data(parts, scalers, skip_text=False):
    for part in parts.values():
        for m in core.MODALITIES:
            if skip_text and m == "text":
                continue
            x = np.asarray(part.feat[m], dtype=np.float32)
            valid = part.vpos[m] & part.available[m][:, None]
            mu, sd = scalers[m]["mean"], scalers[m]["std"]
            x -= mu[None, None, :]
            x /= sd[None, None, :]
            x[~valid] = 0.0
            if not np.isfinite(x).all():
                raise FloatingPointError(f"nonfinite normalized {part.name}/{m}")
            part.feat[m] = x
        part.normalized = True


def _cache_dataset(cfg: Config, ds: str, root: Path, device, smoke=False, budget_check=None):
    parts, raw_splits, pkl_path, csv_path, pkl_hash, ids_sha = load_splits(cfg, ds, root)
    bert, tokenizer, bert_meta = _load_hf(cfg, ds, device)
    token_audits = [_tokenizer_audit(tokenizer, raw_splits[s], parts[s], ds, s) for s in parts]
    # For SMOKE, prove the input/output masking semantics on real rows without building full caches.
    if smoke:
        p = parts["train"]
        ids, attn, types = p.token_ids[: min(2, len(p.y))], p.token_attention[: min(2, len(p.y))], p.token_types[: min(2, len(p.y))]
        full = _bert_forward(bert, ids, attn, types, device, budget_check=budget_check)
        no_del = _bert_forward(bert, ids.copy(), attn.copy(), types.copy(), device, budget_check=budget_check)
        if np.max(np.abs(full - no_del)) > 1e-6:
            raise AssertionError("no-deletion PRE/POST identity failed")
        groups = p.content_groups[0]
        n_delete = min(int(math.floor(0.2 * len(groups))), max(0, len(groups) - 1))
        if len(groups) > 1:
            n_delete = max(1, n_delete)
        deleted = np.concatenate(groups[:n_delete]) if n_delete else np.empty(0, dtype=np.int64)
        a_ids, a_attn, a_types = ids[:1].copy(), attn[:1].copy(), types[:1].copy()
        keep = p.vpos["text"][:1].copy()
        a_ids[0, deleted] = 0
        a_attn[0, deleted] = 0
        a_types[0, deleted] = 0
        keep[0, deleted] = False
        out_a = _bert_forward(bert, a_ids, a_attn, a_types, device, budget_check=budget_check)
        a_ids[0, deleted] = (a_ids[0, deleted] + 1) % int(bert.config.vocab_size)
        out_b = _bert_forward(bert, a_ids, a_attn, a_types, device, budget_check=budget_check)
        diff = float(np.max(np.abs(out_a[0, keep[0]] - out_b[0, keep[0]]))) if keep.any() else 0.0
        if diff > 1e-5:
            raise AssertionError(f"deleted token changed retained PRE states: max_diff={diff}")
        dims = {m: int(parts["train"].feat[m].shape[-1]) for m in core.MODALITIES}
        model = build_model("B1_FUSION", dims, {m: p.feat[m].shape[1] for m in core.MODALITIES}, float(np.median(p.y))).to(device).eval()
        fake = {"x_text": torch.zeros(2, p.feat["text"].shape[1], dims["text"], device=device),
                "keep_text": torch.ones(2, p.feat["text"].shape[1], device=device),
                "plen_text": torch.full((2,), p.feat["text"].shape[1], dtype=torch.long, device=device),
                "vlen_text": torch.full((2,), float(p.feat["text"].shape[1]), device=device)}
        for m in ("audio", "vision"):
            fake[f"x_{m}"] = torch.zeros(2, p.feat[m].shape[1], dims[m], device=device)
            fake[f"keep_{m}"] = torch.ones(2, p.feat[m].shape[1], device=device)
            fake[f"plen_{m}"] = torch.full((2,), p.feat[m].shape[1], dtype=torch.long, device=device)
            fake[f"vlen_{m}"] = torch.full((2,), float(p.feat[m].shape[1]), device=device)
        forwards = {}
        lengths = {m: int(p.feat[m].shape[1]) for m in core.MODALITIES}
        for name in cfg.models:
            model = build_model(name, dims, lengths, float(np.median(p.y))).to(device).eval()
            with torch.no_grad():
                out = model(fake)
                single = model({k: v[:1] for k, v in fake.items()})
                empty = {k: (torch.zeros_like(v) if k.startswith('keep_') else v) for k, v in fake.items()}
                absent = model(empty)
            if out.shape != (2,) or not bool(torch.isfinite(out).all()):
                raise AssertionError(f"{name}: smoke shape/nonfinite failure")
            if not torch.allclose(single, out[:1], atol=2e-5, rtol=1e-5) or not bool(torch.isfinite(absent).all()):
                raise AssertionError(f"{name}: batch1/empty smoke failure")
            forwards[name] = 'passed'
        smoke_result = {"status": "smoke_passed", "dataset": ds, "pkl_sha256": pkl_hash,
                        "token_audits": token_audits, "pre_mask_retained_state_max_diff": diff,
                        "legacy_text_cache_audit_first_rows": _legacy_feature_audit(
                            p.feat["text"][:len(full)], full, p.vpos["text"][:len(full)] & p.available["text"][:len(full), None]),
                        "model_forwards": forwards, "test_inference": False, "bert": bert_meta}
        return smoke_result, {}, {}, {}, {"pkl_sha256": pkl_hash}, token_audits

    cache_root = root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    banks = {s: _delete_bank(p, ds, s, cfg.mask_seed) for s, p in parts.items()}
    token_identity = {s: {"token_ids_sha256": _sha_array(p.token_ids),
                          "attention_sha256": _sha_array(p.token_attention),
                          "token_types_sha256": _sha_array(p.token_types),
                          "raw_text_sha256": _fingerprint(_raw_texts(raw_splits[s], len(p.y)))}
                     for s, p in parts.items()}
    mask_sha = {s: {str(r): _sha_array(m) for r, m in banks[s].items()} for s in banks}
    mask_audit = {s: _mask_rate_audit(parts[s], banks[s]) for s in banks}
    source_hashes = {"encoding": _sha_file(Path(__file__)), "features": _sha_file(Path(core.__file__)),
                     "storage": _sha_file(Path(storage.__file__))}
    cache_inputs = {"protocol": PROTOCOL, "dataset": ds, "pkl_sha256": pkl_hash,
                    "csv_sha256": _sha_file(csv_path), "token_audits": token_audits,
                    "bert": bert_meta, "ids_sha256": ids_sha,
                    "token_identity": token_identity, "mask_sha256": mask_sha,
                    "source_hashes": source_hashes, "encoder_batch_size": cfg.bert_batch_size}
    cache_key = _fingerprint(cache_inputs)
    manifest_path = root / "cache_manifest.json"
    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        scaler_path = root / "train_scalers.npz"
        files_ok = cached.get("cache_key") == cache_key and scaler_path.is_file()
        if files_ok:
            files_ok = _sha_file(scaler_path) == cached.get("scalers_sha256")
        if files_ok:
            for s in parts:
                post_path = cache_root / s / "post_normalized.npy"
                files_ok &= post_path.is_file() and _sha_file(post_path) == cached.get("post_normalized_files_sha256", {}).get(s)
                for rate in RATES:
                    pre_path = cache_root / s / f"pre_r{int(rate*100):02d}.npy"
                    pre_item = cached.get("pre_cache", {}).get(s, {}).get(str(rate), {})
                    files_ok &= (pre_path.is_file() and _sha_file(pre_path) == pre_item.get("sha256")
                                 and storage.verify(pre_path, expected=pre_item.get("storage")))
        if files_ok:
            with np.load(scaler_path, allow_pickle=False) as z:
                scalers = {m: {k: z[f"{m}_{k}"].copy() for k in ("mean", "std", "positions")}
                           for m in core.MODALITIES}
            _normalise_data(parts, scalers, skip_text=True)
            post, pre_files = {}, {s: {} for s in parts}
            for s, part in parts.items():
                post_path = cache_root / s / "post_normalized.npy"
                post[s] = np.load(post_path, mmap_mode="r", allow_pickle=False)
                part.feat["text"] = post[s]
                for rate in RATES:
                    pre_files[s][rate] = cache_root / s / f"pre_r{int(rate*100):02d}.npy"
            del bert
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return parts, post, banks, pre_files, cached, token_audits
    storage.cache_plan(parts, banks, root)
    for s, bank in banks.items():
        for rate, masks in bank.items():
            np.save(cache_root / f"{s}_delete_r{int(rate*100):02d}.npy", masks, allow_pickle=False)
    post_raw = {}
    legacy_audit = {}
    for s, part in parts.items():
        legacy_text = part.feat["text"]
        path, values = _save_post_raw(bert, part, cache_root / s, device, batch_size=cfg.bert_batch_size, budget_check=budget_check, persist=False)
        post_raw[s] = values
        valid_positions = part.vpos["text"] & part.available["text"][:, None]
        legacy_audit[s] = _legacy_feature_audit(legacy_text, values, valid_positions)
    for s in parts:
        parts[s].feat["text"] = post_raw[s]
    scalers = core.fit_scalers(parts["train"])
    _normalise_data(parts, scalers)
    # Matched fp16 storage keeps POST and PRE cache quantization identical at the model input.
    for s in parts:
        parts[s].feat["text"] = parts[s].feat["text"].astype(np.float16)
    core.OUTPUT_ROOT = root.parent
    np.savez(root / "train_scalers.npz", **{f"{m}_{k}": v for m, stats in scalers.items() for k, v in stats.items()})
    post = {s: p.feat["text"] for s, p in parts.items()}
    post_file_hashes = {}
    for s in parts:
        post_path = cache_root / s / "post_normalized.npy"
        storage.reserve_file(post_path.parent, post[s].nbytes + (1 << 20), "Clean POST cache")
        np.save(post_path, post[s], allow_pickle=False)
        post_file_hashes[s] = _sha_file(post_path)
        post[s] = np.load(post_path, mmap_mode="r", allow_pickle=False)
        parts[s].feat["text"] = post[s]
    pre_files = {s: {} for s in parts}
    pre_meta = {s: {} for s in parts}
    for s, part in parts.items():
        pre_meta[s] = _build_pre_cache(bert, part, banks[s], scalers, cache_root / s,
                                       device, bert_meta, batch_size=cfg.bert_batch_size, budget_check=budget_check)
        for rate in RATES:
            pre_files[s][rate] = Path(pre_meta[s][rate]["path"])
    identity = {"protocol": PROTOCOL, "dataset": ds, "pkl_sha256": pkl_hash,
                "csv_sha256": _sha_file(csv_path), "token_audits": token_audits,
                "bert": bert_meta, "scalers_sha256": _sha_file(root / "train_scalers.npz"),
                "legacy_text_feature_audit": legacy_audit,
                "mask_sha256": mask_sha, "mask_rate_audit": mask_audit,
                "ids_sha256": ids_sha, "token_identity": token_identity,
                "pre_cache": pre_meta,
                "bert_raw_output_sha256": {s: _sha_array(post_raw[s]) for s in post_raw},
                "post_cache_sha256": {s: _sha_array(post[s]) for s in post},
                "post_normalized_files_sha256": post_file_hashes,
                "cache_inputs": cache_inputs, "cache_key": cache_key,
                "source_hashes": source_hashes,
                "post_cache_dtype": "float16", "pre_cache_dtype": "float16"}
    _json(root / "cache_manifest.json", identity)
    del bert
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return parts, post, banks, pre_files, identity, token_audits
