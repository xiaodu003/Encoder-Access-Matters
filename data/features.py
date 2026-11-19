"""Feature loading, token validation, normalization and batch utilities."""
import csv, hashlib, json, math, os, pickle, random, re, sys, time
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

class _QuietProgress:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
        self.n = 0
    def __iter__(self):
        return iter(self.iterable)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def update(self, n=1):
        self.n += n
    def set_postfix(self, *args, **kwargs):
        pass
    def set_description(self, *args, **kwargs):
        pass
    def close(self):
        pass


tqdm = _QuietProgress


OUTPUT_ROOT = Path("runs/data_audit")


INPUT_ROOT = Path("inputs")


QUICK_CHECK = False


SIMS_V1_LABEL_COLUMNS = (
    "video_id",
    "clip_id",
    "text",
    "label",
    "label_T",
    "label_A",
    "label_V",
    "annotation",
    "mode",
)


SIMS_V1_REFERENCE_SEMANTIC_SHA256 = (
    "887b6d0b65f1f06b7705e49e67a74002f407678e0b032f7531100fd9daf7ca13"
)


SIMS_V1_EXPECTED_SPLITS = {"train": 1368, "valid": 456, "test": 457}


CSV_LABEL_ABS_TOL = 1e-4  # precision of published decimal labels


ALLOW_UNVERIFIED_TEXT_POSITIONS = (
    False  # no guessing for unknown CLS/SEP; True only for quick diagnostic
)


ALLOW_LENGTH_INFERENCE_FORMAL = False  # formal runs require released A/V lengths


VISION_ALL_ZERO_RULE = {"MOSI": False, "MOSEI": True, "SIMS_V1": False}


SCALER_CHUNK_ROWS = 96


MODALITIES = ("text", "audio", "vision")


SEP_TOKEN = "$_$"


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


DATASET_FILES = {
    "MOSI": {
        "pkl": [
            "datasets/yutian0967/ytmmsa/Datasets/MOSI/unaligned_50.pkl",
            "ytmmsa/Datasets/MOSI/unaligned_50.pkl",
        ],
        "csv": [
            "datasets/yutian0967/ytmmsa/Datasets/MOSI/MOSI-label.csv",
            "ytmmsa/Datasets/MOSI/MOSI-label.csv",
        ],
        "expected_sha256": "78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524",
        "source_url": "https://www.kaggle.com/datasets/yutian0967/ytmmsa",
    },
    "MOSEI": {
        "pkl": [
            "datasets/yutian0967/ytmmsa/Datasets/MOSEI/unaligned_50.pkl",
            "ytmmsa/Datasets/MOSEI/unaligned_50.pkl",
        ],
        "csv": [
            "datasets/yutian0967/ytmmsa/Datasets/MOSEI/MOSEI-label.csv",
            "ytmmsa/Datasets/MOSEI/MOSEI-label.csv",
        ],
        "expected_sha256": "ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f",
        "source_url": "https://www.kaggle.com/datasets/yutian0967/ytmmsa",
    },
    "SIMS_V1": {
        "pkl": [
            "datasets/yutian0967/ytmmsa/Datasets/SIMS/unaligned_39.pkl",
            "ytmmsa/Datasets/SIMS/unaligned_39.pkl",
        ],
        "csv": [
            "datasets/yutian0967/ytmmsa/Datasets/SIMS/SIMS-label.csv",
            "ytmmsa/Datasets/SIMS/SIMS-label.csv",
        ],
        "expected_sha256": "c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f",
        "source_url": "https://github.com/thuiar/MMSA",
        "kaggle_mirror": "https://www.kaggle.com/datasets/yutian0967/ytmmsa",
        "reference_label_sha256_semantic": SIMS_V1_REFERENCE_SEMANTIC_SHA256,
    },
}


def stable_seed(*parts):
    v = hashlib.blake2b(
        "|".join(map(str, parts)).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(v, "little") % (2**32 - 1)


def rng_for(*parts):
    return np.random.default_rng(stable_seed(*parts))


def json_ready(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return json_ready(float(x))
    if isinstance(x, (np.ndarray,)):
        return [json_ready(q) for q in x.tolist()]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): json_ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_ready(v) for v in x]
    return x


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(json_ready(data), f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def atomic_csv(path, records, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df = (
        records
        if isinstance(records, pd.DataFrame)
        else pd.DataFrame.from_records(records)
    )
    if columns is not None and df.empty:
        df = pd.DataFrame(columns=columns)
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def path_case_insensitive(root, relative):
    here = root
    if not here.is_dir():
        return None
    for part in Path(relative).parts:
        choices = [q for q in here.iterdir() if q.name.casefold() == part.casefold()]
        if len(choices) != 1:
            return None
        here = choices[0]
    return here if here.is_file() else None


def resolve_known_input(ds_name, kind, required=True):
    # Important: never recursively select an unknown `unaligned.pkl` (old SIMS may coexist).
    candidates = DATASET_FILES[ds_name][kind]
    found = [
        p
        for rel in candidates
        if (p := path_case_insensitive(INPUT_ROOT, rel)) is not None
    ]
    canonical = {str(p.resolve()): p for p in found}
    if len(canonical) > 1:
        raise RuntimeError(
            f"Ambiguous {ds_name}/{kind}: {list(canonical.values())}; choose audited source explicitly"
        )
    if canonical:
        return list(canonical.values())[0]
    if not required:
        return None
    listing = (
        [str(p) for p in INPUT_ROOT.iterdir()]
        if INPUT_ROOT.is_dir()
        else [f"Not mounted: {INPUT_ROOT}"]
    )
    raise FileNotFoundError(
        f"{ds_name}/{kind} missing. Expected one of {candidates}. Mount Kaggle inputs. "
        f"Top-level: {listing[:30]}"
    )


class ProgressReader:
    """Track actual consumed bytes during trusted pickle deserialization (not a streaming PKL parser)."""

    def __init__(self, handle, bar):
        self.handle, self.bar = handle, bar

    def read(self, n=-1):
        z = self.handle.read(n)
        self.bar.update(len(z))
        return z

    def readline(self, n=-1):
        z = self.handle.readline(n)
        self.bar.update(len(z))
        return z

    def readinto(self, buf):
        n = self.handle.readinto(buf)
        self.bar.update(n or 0)
        return n


def read_trusted_pkl(path):
    # WARNING: pickle executes arbitrary code! Only open the exact trusted dataset path.
    size = path.stat().st_size
    with path.open("rb") as f, tqdm(
        total=size, unit="B", unit_scale=True, desc="PKL deserialize", leave=True
    ) as bar:
        raw = pickle.load(ProgressReader(f, bar), encoding="latin1")
    if not isinstance(raw, dict):
        raise TypeError(f"PKL root must be dict, got {type(raw)}")
    return raw


def dict_get_ci(obj, names):
    if not isinstance(obj, dict):
        return None, None
    low = {str(k).lower(): k for k in obj}
    for name in names:
        if name.lower() in low:
            return obj[low[name.lower()]], str(low[name.lower()])
    return None, None


def get_split(raw, name):
    value, key = dict_get_ci(
        raw,
        {"train": ("train",), "valid": ("valid", "val", "dev"), "test": ("test",)}[
            name
        ],
    )
    if value is None or not isinstance(value, dict):
        raise KeyError(f"{name} split not found as dict. Top-level keys: {list(raw)}")
    return value


def tensor_to_np(o):
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().numpy()
    if isinstance(o, np.ndarray):
        return o
    return None


def extract_feature(obj, n, mod, field):
    if isinstance(obj, dict):
        sub, key = dict_get_ci(obj, (mod, f"{mod}_features", "features"))
        if sub is None:
            raise TypeError(f"{field} is dict without feature array: {list(obj)}")
        return extract_feature(sub, n, mod, field + "." + key)
    a = tensor_to_np(obj)
    if a is None and isinstance(obj, (list, tuple)) and len(obj) == n:
        mats = [np.asarray(x) for x in tqdm(obj, desc=f"ragged {mod}", leave=False)]
        if not all(x.ndim == 2 for x in mats):
            raise ValueError(f"{field}: ragged item is not a matrix")
        dims = set(x.shape[-1] for x in mats)
        if len(dims) != 1:
            raise ValueError(f"{field}: inconsistent ragged feature dimensions {dims}")
        maxlen = max(x.shape[0] for x in mats)
        arr = np.zeros((n, max(1, maxlen), mats[0].shape[1]), dtype=np.float32)
        lengths = np.asarray([x.shape[0] for x in mats], dtype=np.int64)
        for i, x in enumerate(tqdm(mats, desc=f"pad {mod}", leave=False)):
            arr[i, : len(x), :] = x
        return arr, lengths
    if a is None:
        a = np.asarray(obj)
    if a.dtype.kind not in "fiu" or a.ndim != 3 or a.shape[0] != n or a.shape[-1] < 2:
        raise ValueError(
            f"{field}: expected numeric features N,L,D (D>=2), N={n}; got {a.shape}/{a.dtype}"
        )
    # text_bert token IDs (D=3,L=39) should never be misread as cached embeddings.
    if mod == "text" and a.shape[-1] < 16:
        raise ValueError(
            f"{field}: text must be embedding features, not BERT token IDs/masks, shape={a.shape}"
        )
    return np.ascontiguousarray(a, dtype=np.float32), None


def extract_label(d, n, ds_name):
    forbidden = {"classification_labels", "classification_labels_t", "annotations"}
    choices = (
        "regression_labels",
        "regression_label",
        "label",
        "sentiment_score",
        "sentiment",
        "M",
        "labels",
    )
    x, field = dict_get_ci(d, choices)
    if x is None:
        raise KeyError(f"No continuous target among {choices}; actual keys: {list(d)}")
    if field.lower() in forbidden:
        raise ValueError(f"Classification label field selected erroneously: {field}")
    if isinstance(x, dict):
        sub, key = dict_get_ci(x, ("M", "regression", "label", "multimodal"))
        if sub is None:
            raise ValueError(
                f"Multi-head labels must have explicit multimodal field M: {list(x)}"
            )
        x, field = sub, field + "." + key
    a = np.asarray(x)
    if a.dtype == object:
        items = [np.asarray(q).reshape(-1) for q in x]
        if any(len(q) != 1 for q in items):
            raise ValueError(f"{field}: not a scalar regression target")
        a = np.asarray([q[0] for q in items])
    if len(a) != n or a.reshape(n, -1).shape[1] != 1:
        raise ValueError(f"{field}: not (N,) or (N,1), N={n}; actual {a.shape}")
    y = a.reshape(n).astype(np.float32)
    if not np.isfinite(y).all():
        raise ValueError(f"{field}: target has NaN/Inf")
    bound = 1.001 if ds_name == "SIMS_V1" else 3.001
    if (np.abs(y) > bound).any():
        raise ValueError(
            f"{ds_name}: {field} out of expected regression range ±{bound}"
        )
    if ds_name == "SIMS_V1" and field != "regression_labels":
        raise ValueError(
            f"SIMS_V1 main task requires PKL regression_labels, found {field}; refuse T/A/V target substitution"
        )
    return y, field


def valid_lengths(d, n, mod, L, ragged, allow_fallback=False):
    x, field = dict_get_ci(
        d, (f"{mod}_lengths", f"{mod}_length", f"{mod}_len", f"{mod}_valid_lengths")
    )
    if x is None and ragged is not None:
        return ragged, "ragged_array_extent"
    if x is None:
        return None, "absent"
    a = np.asarray(x).reshape(-1)
    if len(a) != n or not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all():
        raise ValueError(f"{field}: invalid lengths / shape")
    if (a != np.rint(a)).any():
        raise ValueError(f"{field}: non-integer length value")
    lengths = a.astype(np.int64)
    if (lengths < 0).any() or (lengths > L).any():
        raise ValueError(f"{field}: lengths outside [0,{L}] (not silently clipped)")
    return lengths, field


def get_text_positions(d, n, L, ds_name=None, split_name=None):
    """Use the BERT attention mask, removing only recognized special-token IDs.

    The historical strict (first==101 AND last==102) check falsely rejected truncated
    or one-boundary-absent rows. It is not evidence that the other boundary is a
    special token. Never silently drop such rows, replace labels, or infer text
    boundaries from nonzero embedding values in the official BERT-token path.

    Writes per-split boundary reports before any exception, so a Kaggle failure ZIP
    retains the actual token IDs and sample IDs needed to investigate anomalies.
    """
    tok, key = dict_get_ci(d, ("text_bert", "text_tokens", "bert_inputs"))
    if tok is not None:
        arr = np.asarray(tok)
        if not (
            arr.ndim == 3
            and arr.shape[0] == n
            and arr.shape[1] in (2, 3)
            and arr.shape[2] == L
        ):
            raise ValueError(f"{key}: expected (N,2/3,{L}), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{key}: token IDs or masks contain nonfinite values")
        mask = arr[:, 1, :]
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{key} row1 is not a binary attention mask")
        observed = mask.astype(bool)
        ids = arr[:, 0, :]
        has_any = observed.any(axis=1)
        idx = np.arange(n)
        first_idx = observed.argmax(axis=1)
        last_idx = L - 1 - observed[:, ::-1].argmax(axis=1)
        first_tok = ids[idx, first_idx]
        last_tok = ids[idx, last_idx]
        first_cls = has_any & (first_tok == 101)
        last_sep = has_any & (last_tok == 102)
        cls_token = observed & (ids == 101)
        sep_token = observed & (ids == 102)
        pad_attended = observed & (ids == 0)
        cls_count = cls_token.sum(axis=1)
        sep_count = sep_token.sum(axis=1)
        interior_special = (
            cls_token & (np.arange(L)[None, :] != first_idx[:, None])
        ).any(axis=1) | (sep_token & (np.arange(L)[None, :] != last_idx[:, None])).any(
            axis=1
        )
        noncontiguous_attention = np.asarray(
            [
                (len(pos) > 0 and (pos[-1] - pos[0] + 1) != len(pos))
                for pos in (np.flatnonzero(row) for row in observed)
            ],
            dtype=bool,
        )
        partial_missing_sep = first_cls & ~last_sep & (sep_count == 0)
        partial_missing_cls = last_sep & ~first_cls & (cls_count == 0)
        standard = first_cls & last_sep & (cls_count == 1) & (sep_count == 1)
        zero_mask = ~has_any
        unexplained = has_any & ~(standard | partial_missing_sep | partial_missing_cls)
        hard_bad = pad_attended.any(axis=1) | interior_special | noncontiguous_attention
        uncertain = unexplained & ~hard_bad

        # Verify that this split overwhelmingly uses the declared BERT 101/102 IDs.
        anchor_count = int((first_cls | last_sep).sum())
        nonempty_count = int(has_any.sum())
        common_tokenizer = nonempty_count == 0 or anchor_count / nonempty_count >= 0.90
        # Confirmed 101 and 102 are excluded even on a row where just one is present.
        content = observed & ~cls_token & ~sep_token
        # A hidden interior separator would break contiguous-burst semantics; hard stop.
        noncontiguous_content = np.asarray(
            [
                (len(pos) > 0 and (pos[-1] - pos[0] + 1) != len(pos))
                for pos in (np.flatnonzero(row) for row in content)
            ],
            dtype=bool,
        )
        hard_bad |= noncontiguous_content

        info = {
            "source": f"{key}.attention_mask minus observed token IDs 101/102",
            "samples_with_any_token": nonempty_count,
            "samples_with_known_boundaries": int(standard.sum()),
            "partial_missing_sep": int(partial_missing_sep.sum()),
            "partial_missing_cls": int(partial_missing_cls.sum()),
            "unexplained_boundary_rows": int(unexplained.sum()),
            "attended_PAD_rows": int(pad_attended.any(1).sum()),
            "interior_special_rows": int(interior_special.sum()),
            "noncontiguous_attention_rows": int(noncontiguous_attention.sum()),
            "noncontiguous_content_rows": int(noncontiguous_content.sum()),
            "empty_attention_rows": int(zero_mask.sum()),
            "tokenizer_anchor_fraction": (
                float(anchor_count / nonempty_count) if nonempty_count else None
            ),
            "note": "Partial endpoints are recorded; absent SEP is not automatically proof of truncation.",
        }
        report_path = None
        if ds_name is not None and split_name is not None:
            # Only atypical rows are written: keeps diagnostic CSV compact on MOSEI.
            raw_ids, _ = dict_get_ci(d, ("id", "ids", "sample_id"))
            selected = np.flatnonzero(~standard)
            rows = []
            for row in tqdm(
                selected, desc=f"{ds_name}/{split_name} token audit", leave=False
            ):
                i = int(row)
                rows.append(
                    {
                        "dataset": ds_name,
                        "split": split_name,
                        "row_index": i,
                        "id": (
                            normalize_raw_id(raw_ids[i])
                            if raw_ids is not None and len(raw_ids) == n
                            else None
                        ),
                        "attention_length": int(observed[i].sum()),
                        "max_length": int(L),
                        "first_position": int(first_idx[i]) if has_any[i] else None,
                        "last_position": int(last_idx[i]) if has_any[i] else None,
                        "first_token_id": int(first_tok[i]) if has_any[i] else None,
                        "last_token_id": int(last_tok[i]) if has_any[i] else None,
                        "CLS_count": int(cls_count[i]),
                        "SEP_count": int(sep_count[i]),
                        "has_CLS_at_start": bool(first_cls[i]),
                        "has_SEP_at_end": bool(last_sep[i]),
                        "partial_missing_SEP": bool(partial_missing_sep[i]),
                        "partial_missing_CLS": bool(partial_missing_cls[i]),
                        "both_boundaries_unverified": bool(uncertain[i]),
                        "attended_PAD": bool(pad_attended[i].any()),
                        "interior_special": bool(interior_special[i]),
                        "noncontiguous_attention": bool(noncontiguous_attention[i]),
                        "noncontiguous_content": bool(noncontiguous_content[i]),
                        "full_length_without_SEP": bool(
                            partial_missing_sep[i] and observed[i].sum() == L
                        ),
                    }
                )
            report_path = OUTPUT_ROOT / ds_name / f"text_token_audit_{split_name}.csv"
            atomic_csv(
                report_path,
                rows,
                columns=[
                    "dataset",
                    "split",
                    "row_index",
                    "id",
                    "attention_length",
                    "max_length",
                    "first_position",
                    "last_position",
                    "first_token_id",
                    "last_token_id",
                    "CLS_count",
                    "SEP_count",
                    "has_CLS_at_start",
                    "has_SEP_at_end",
                    "partial_missing_SEP",
                    "partial_missing_CLS",
                    "both_boundaries_unverified",
                    "attended_PAD",
                    "interior_special",
                    "noncontiguous_attention",
                    "noncontiguous_content",
                    "full_length_without_SEP",
                ],
            )
            atomic_json(
                OUTPUT_ROOT / ds_name / f"text_token_audit_{split_name}.json", info
            )
            print(
                f"{ds_name}/{split_name} text token audit:",
                {
                    k: info[k]
                    for k in (
                        "samples_with_known_boundaries",
                        "partial_missing_sep",
                        "partial_missing_cls",
                        "unexplained_boundary_rows",
                        "interior_special_rows",
                        "attended_PAD_rows",
                    )
                },
                f"details={report_path}",
            )
        if not common_tokenizer:
            raise ValueError(
                f"{key}: fewer than 90% attended rows confirm BERT 101/102 boundaries; "
                f"tokenizer must be verified. Audit={report_path}"
            )
        if hard_bad.any():
            raise ValueError(
                f"{key}: {int(hard_bad.sum())} rows have attended PAD, interior special tokens, "
                f"or non-contiguous attention/content masks. Audit={report_path}"
            )
        if uncertain.any() and not (QUICK_CHECK and ALLOW_UNVERIFIED_TEXT_POSITIONS):
            raise ValueError(
                f"{key}: {int(uncertain.sum())} rows have neither verifiable BERT boundary. "
                f"Inspect actual token IDs in {report_path}. "
                "Do not turn off checks for formal experiments."
            )
        if uncertain.any():
            print(
                f"WARNING: QUICK-ONLY opted-in ambiguous text boundaries: {int(uncertain.sum())}. "
                "Results MUST NOT be used as formal evidence."
            )
        return content, info
    lengths, src = valid_lengths(d, n, "text", L, None)
    if lengths is not None:
        if not ALLOW_UNVERIFIED_TEXT_POSITIONS:
            raise ValueError(
                "Only text_lengths provided: cannot tell whether CLS/SEP included; "
                "set ALLOW_UNVERIFIED_TEXT_POSITIONS=True only after inspecting source"
            )
        pos = np.arange(L)[None, :] < lengths[:, None]
        return pos, {"source": src, "warning": "unverified token boundaries"}
    if not ALLOW_UNVERIFIED_TEXT_POSITIONS:
        raise ValueError(
            "No text_bert or vetted content positions; refused to treat padding as text"
        )
    feat, _ = dict_get_ci(d, ("text",))
    x = np.asarray(feat)
    if x.ndim != 3:
        raise ValueError("cannot infer text extent")
    extent = np.any(x != 0, axis=2)
    return extent, {"source": "heuristic nonzero positions", "warning": "UNVERIFIED"}


def normalize_raw_id(x):
    if isinstance(x, bytes):
        return x.decode("utf8")
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x).reshape(-1)
        if arr.size == 1:
            return normalize_raw_id(arr[0])
        if arr.size == 2:
            return normalize_raw_id(arr[0]) + SEP_TOKEN + normalize_raw_id(arr[1])
        raise ValueError(f"ID must be scalar or (video_id,clip_id), got {arr.shape}")
    return str(x)


def src_from_id(s, ds_name):
    if SEP_TOKEN in s:
        return s.split(SEP_TOKEN, 1)[0]
    # CH-SIMS v1: source video_id can include an underscore; only parse final numeric clip.
    if ds_name == "SIMS_V1" and "_" in s and s.rsplit("_", 1)[1].isdigit():
        return s.rsplit("_", 1)[0]
    return s


@dataclass
class SplitData:
    name: str
    feat: dict
    vpos: dict
    plen: dict
    available: dict
    y: np.ndarray
    ids: np.ndarray
    sources: np.ndarray
    vision_zero: np.ndarray
    meta: dict
    normalized: bool = False


def parse_split(name, raw, ds_name, progress=True):
    # Fail on missing multimodal cached features; `text_bert` is tokens, not features.
    field_map = {
        "text": ("text", "text_features", "language"),
        "audio": ("audio", "audio_features", "acoustic"),
        "vision": ("vision", "visual", "vision_features", "visual_features"),
    }
    txt, _ = dict_get_ci(raw, field_map["text"])
    if txt is None:
        raise KeyError(f"{name}: no text embedding array, fields={list(raw)}")
    n = len(txt)
    if n <= 0:
        raise ValueError(f"{name}: empty split")
    feats, ragged, feat_fields, vpos, plen, avail, meta = {}, {}, {}, {}, {}, {}, {}
    raw_zero = np.zeros(n, dtype=bool)
    for m in tqdm(MODALITIES, desc=f"Parse {name} modalities", leave=True):
        obj, key = dict_get_ci(raw, field_map[m])
        if obj is None:
            raise KeyError(f"{name}: missing {m} feature, keys={list(raw)}")
        feats[m], ragged[m] = extract_feature(obj, n, m, f"{name}.{key}")
        feat_fields[m] = key
        if not np.isfinite(feats[m]).all():
            # Official Published feature adaptation turns -inf A/V values into 0; never alter +inf/NaN.
            neginf = int(np.isneginf(feats[m]).sum())
            other = int((np.isnan(feats[m]) | np.isposinf(feats[m])).sum())
            if other or m == "text":
                raise ValueError(
                    f"{name}.{m}: non-finite features (-inf={neginf}, other={other})"
                )
            feats[m][np.isneginf(feats[m])] = 0.0
            meta.setdefault(m, {})["neginf_to_zero"] = neginf
        if m == "text":
            pos, info = get_text_positions(
                raw, n, feats[m].shape[1], ds_name=ds_name, split_name=name
            )
            meta[m] = info
        else:
            lens, src = valid_lengths(raw, n, m, feats[m].shape[1], ragged[m])
            if lens is None:
                if not QUICK_CHECK and not ALLOW_LENGTH_INFERENCE_FORMAL:
                    raise ValueError(f"{name}.{m}: missing lengths; no formal fallback")
                x = feats[m]
                nz = np.any(x != 0, axis=2)
                lens = np.where(
                    nz.any(1), x.shape[1] - np.argmax(nz[:, ::-1], axis=1), 0
                )
                src = "unverified_nonzero_extent_QUICK_ONLY"
            pos = np.arange(feats[m].shape[1])[None, :] < lens[:, None]
            meta.setdefault(m, {})["length_source"] = src
        vpos[m] = pos.astype(bool)
        plen[m] = np.where(
            pos.any(1), pos.shape[1] - np.argmax(pos[:, ::-1], axis=1), 0
        ).astype(np.int64)
        avail[m] = pos.any(1)
        meta.setdefault(m, {})["feature_field"] = feat_fields[m]
        _present = pos.any(axis=1)
        _first = pos.argmax(axis=1)
        _last = pos.shape[1] - 1 - pos[:, ::-1].argmax(axis=1)
        _contig = (~_present) | ((_last - _first + 1) == pos.sum(axis=1))
        meta[m]["noncontiguous_valid_rows"] = int(np.count_nonzero(~_contig))
        meta[m]["min_max_pack_len"] = [int(plen[m].min()), int(plen[m].max())]
        if m == "vision":
            raw_zero = np.asarray(
                [
                    not np.any(feats[m][i][pos[i]])
                    for i in tqdm(range(n), desc=f"{name} vision quality", leave=False)
                ],
                dtype=bool,
            )
            if VISION_ALL_ZERO_RULE[ds_name]:
                avail[m] &= ~raw_zero
    labels, label_field = extract_label(raw, n, ds_name)
    ids_raw, id_field = dict_get_ci(raw, ("id", "ids", "sample_id"))
    if ids_raw is None or len(ids_raw) != n:
        raise ValueError(
            f"{name}: missing trustworthy sample IDs; formal alignment cannot be checked"
        )
    ids = np.asarray(
        [normalize_raw_id(x) for x in tqdm(ids_raw, desc=f"{name} IDs", leave=False)],
        dtype=object,
    )
    if len(set(ids)) != n:
        raise ValueError(f"{name}: duplicate IDs within split")
    sources = np.asarray([src_from_id(s, ds_name) for s in ids], dtype=object)
    meta.update(
        {
            "label_field": label_field,
            "id_field": id_field,
            "count": int(n),
            "feature_dims": {m: int(feats[m].shape[-1]) for m in MODALITIES},
            "feature_shapes": {m: list(feats[m].shape) for m in MODALITIES},
            "all_zero_vision": int(raw_zero.sum()),
            "unavailable": {m: int((~avail[m]).sum()) for m in MODALITIES},
            "source_count": int(len(set(sources))),
        }
    )
    return SplitData(
        name, feats, vpos, plen, avail, labels, ids, sources, raw_zero, meta
    )


def canon_id_pair(video, clip):
    def norm(s):
        s = str(s).strip()
        if s.endswith(".0") and s[:-2].isdigit():
            return s[:-2]
        return s

    return norm(video) + SEP_TOKEN + norm(clip)


def canonical_pkl_id(raw_id, form):
    s = str(raw_id).strip()
    if form == "literal":
        return s
    if form == "underscore_tail":
        if SEP_TOKEN in s:
            return s
        if "_" in s and s.rsplit("_", 1)[1].isdigit():
            a, b = s.rsplit("_", 1)
            return canon_id_pair(a, b)
    return s


def sims_v1_label_semantic_digest(meta):
    """Version-robust SHA-256: sorted sample IDs, splits and four scalar labels.

    Header and order do not affect digest. Transcript differences do not silently
    change the regression reference; text remains original metadata for review.
    """
    import json as _json

    normalized = []
    for row in meta.itertuples(index=False):
        d = dict(zip(meta.columns, row))
        vals = []
        for name in ("label", "label_T", "label_A", "label_V"):
            try:
                value = float(d[name])
            except (ValueError, TypeError):
                raise ValueError(f"SIMS_V1 {name}: nonnumeric label")
            if not math.isfinite(value) or abs(value) > 1.001:
                raise ValueError(f"SIMS_V1 {name}: invalid label {value}")
            vals.append(format(value, ".6f"))
        split = str(d["mode"]).strip().lower()
        if split not in SIMS_V1_EXPECTED_SPLITS:
            raise ValueError(f"SIMS_V1: invalid split {split!r}")
        normalized.append(
            [str(d["video_id"]).strip(), str(d["clip_id"]).strip(), split, *vals]
        )
    normalized.sort()
    payload = "\n".join(
        _json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        for row in normalized
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_sims_v1_labels(csv_path):
    """Authoritative SIMS-label.csv has a real header; never infer headerless layout.

    All string fields are read literally to preserve zero-padded clip IDs.
    CSV data/sample identity is checked against the user's supplied version.
    """
    import csv as _csv

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        first = next(_csv.reader(handle), None)
    required = list(SIMS_V1_LABEL_COLUMNS)
    if first is None or len(first) != len(required):
        raise ValueError(f"SIMS_V1 requires exactly 9 header columns, found {first!r}")
    raw_header = [str(x).strip() for x in first]
    if len({x.casefold() for x in raw_header}) != len(required) or {
        x.casefold() for x in raw_header
    } != {x.casefold() for x in required}:
        raise ValueError(
            "SIMS_V1: expected HEADERED SIMS-label.csv with "
            f"{required}; got first CSV row {first!r}. "
            "The first row is a header, never a sample."
        )
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    raw.columns = [str(c).strip() for c in raw.columns]
    by_lower = {c.casefold(): c for c in raw.columns}
    meta = raw.rename(columns={by_lower[k.casefold()]: k for k in required})[
        required
    ].copy()
    if len(meta) != sum(SIMS_V1_EXPECTED_SPLITS.values()):
        raise ValueError(
            f"SIMS_V1 CSV: expected 2281 data rows AFTER header, got {len(meta)}"
        )
    for c in required:
        if meta[c].astype(str).str.strip().eq("").any():
            raise ValueError(f"SIMS_V1 CSV: missing/empty field in column {c}")
    if not meta["video_id"].str.fullmatch(r"video_\d+").all():
        raise ValueError(
            "SIMS_V1: unexpected video_id formatting; inspect authoritative headered CSV"
        )
    if not meta["clip_id"].str.fullmatch(r"\d+").all():
        raise ValueError("SIMS_V1: clip_id contains nonnumeric values")
    n_by_split = meta["mode"].str.strip().str.lower().value_counts().to_dict()
    if n_by_split != SIMS_V1_EXPECTED_SPLITS:
        raise ValueError(
            f"SIMS_V1 CSV split counts mismatch: {n_by_split} != {SIMS_V1_EXPECTED_SPLITS}"
        )
    if meta[["video_id", "clip_id"]].duplicated().any():
        raise ValueError("SIMS_V1 CSV: duplicate video_id/clip_id pairs")
    label = pd.to_numeric(meta["label"], errors="raise")
    expected_annotation = np.where(
        label > 0, "Positive", np.where(label < 0, "Negative", "Neutral")
    )
    if not np.array_equal(
        meta["annotation"].str.strip().to_numpy(), expected_annotation
    ):
        raise ValueError("SIMS_V1: annotation inconsistent with sign(label)")
    semantic = sims_v1_label_semantic_digest(meta)
    print(
        f"SIMS_V1 labels: {len(meta)} data rows AFTER header; "
        f"source videos={meta.video_id.nunique()}; semantic_sha256={semantic}"
    )
    return meta, {
        "format": "headered",
        "row_count_excluding_header": int(len(meta)),
        "semantic_sha256": semantic,
        "user_reference_match": semantic == SIMS_V1_REFERENCE_SEMANTIC_SHA256,
        "column_roles": {
            "label": "multimodal MAIN regression target",
            "label_T": "text-only external annotation",
            "label_A": "audio-only external annotation",
            "label_V": "vision-only external annotation",
        },
    }


def fit_scalers(train):
    """Numerically stable chunk-merged (Welford) moments without multi-GB float64 temporary matrices."""
    stats = {}
    for m in tqdm(MODALITIES, desc="Train-only scaler modalities", leave=True):
        arr = train.feat[m]
        dim = arr.shape[-1]
        mean = np.zeros(dim, np.float64)
        m2 = np.zeros(dim, np.float64)
        n = 0
        for start in tqdm(
            range(0, len(arr), SCALER_CHUNK_ROWS), desc=f"Train stats {m}", leave=False
        ):
            end = min(len(arr), start + SCALER_CHUNK_ROWS)
            observed = train.vpos[m][start:end] & train.available[m][start:end, None]
            if not observed.any():
                continue
            batch = arr[start:end][observed].astype(np.float64)
            if not np.isfinite(batch).all():
                raise ValueError(f"Train/{m}: nonfinite normalized inputs")
            k = len(batch)
            bmean = batch.mean(axis=0)
            bm2 = np.square(batch - bmean).sum(axis=0)
            new_n = n + k
            delta = bmean - mean
            mean = mean + delta * (k / new_n)
            m2 = m2 + bm2 + delta * delta * (n * k / new_n)
            n = new_n
        if not n:
            raise ValueError(f"Train/{m}: zero observed features")
        std = np.maximum(np.sqrt(np.maximum(m2 / n, 0)), 1e-6)
        stats[m] = {
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "positions": n,
        }
        print(
            "train stats",
            m,
            "observed frames",
            n,
            "D",
            dim,
            "std min",
            float(std.min()),
        )
    return stats


STRING_FIELDS = {"id", "source_id", "mask_sha256"}


def collate_batch(items):
    out = {}
    for k in items[0]:
        if k in STRING_FIELDS:
            out[k] = [x[k] for x in items]
        elif k.startswith(("x_", "keep_", "vpos_")):
            mod = k.split("_", 1)[1]
            L = max(1, max(int(row["plen_" + mod]) for row in items))
            out[k] = torch.stack([row[k][:L] for row in items])
        else:
            out[k] = torch.stack([row[k] for row in items])
    return out


def move_batch(batch):
    return {
        k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def binary_metrics(target, pred):
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if not (np.isfinite(target).all() and np.isfinite(pred).all()):
        raise FloatingPointError("non-finite labels or predictions")
    y = target >= 0
    h = pred >= 0
    n = len(y)
    included = float(np.mean(y == h)) if n else None
    nonzero = target != 0
    excluded = float(np.mean((y == h)[nonzero])) if nonzero.any() else None

    def weighted_f1(subset):
        yt = y[subset]
        yp = h[subset]
        count = len(yt)
        if not count:
            return None
        score = 0.0
        for c in (False, True):
            tp = int(np.sum((yt == c) & (yp == c)))
            fp = int(np.sum((yt != c) & (yp == c)))
            fn = int(np.sum((yt == c) & (yp != c)))
            supp = int(np.sum(yt == c))
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom else 0.0
            score += supp * f1 / count
        return float(score)

    corr = (
        float(np.corrcoef(target, pred)[0, 1])
        if n > 1 and np.std(target) > 0 and np.std(pred) > 0
        else None
    )
    return {
        "acc2_including_zero": included,
        "acc2_excluding_zero": excluded,
        "weighted_f1_including_zero": weighted_f1(np.ones(n, dtype=bool)),
        "weighted_f1_excluding_zero": weighted_f1(nonzero),
        "pearson_r": corr,
        "n_total": n,
        "n_nonzero": int(nonzero.sum()),
        "n_zero": int((~nonzero).sum()),
    }
