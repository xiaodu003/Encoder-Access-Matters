"""Random and contiguous masks, and alternative attention-key interventions."""
import json, math
import numpy as np
from tqdm.auto import tqdm
from common import PROTOCOL, RATES, N_MASKS, _sha_array, _sha_file, _json
from data import features as core, cache as storage
from data.encoding import _delete_bank, _bert_forward
from data.batches import MaskedSentimentDataset
from evaluation.metrics import _predict_dataset

def custom_banks(part, ds, split, seed, style='random', matched=None):
    if style == 'random':
        return _delete_bank(part, ds, split, seed)
    valid = part.vpos['text'] & part.available['text'][:, None]
    out = {}
    budget = []
    for rate in RATES:
        bank = np.zeros((N_MASKS, *valid.shape), dtype=bool)
        for rep in range(N_MASKS):
            for i, sid in enumerate(part.ids):
                groups = [g for g in part.content_groups[i] if len(g) and valid[i, g].all()]
                target = int(matched[rate][rep, i].sum()) if matched is not None else None
                if not groups:
                    budget.append({'dataset': ds, 'split': split, 'id': str(sid), 'rate': rate, 'repeat': rep,
                                   'deleted_units': 0, 'deleted_tokens': 0, 'random_deleted_tokens': target,
                                   'token_budget_exact': target in (None, 0)})
                    continue
                k = min(int(math.floor(rate*len(groups))), max(0, len(groups)-1))
                if len(groups) > 1:
                    k = max(1, k)
                rng = core.rng_for(PROTOCOL, ds, split, seed, rate, rep, str(sid))
                if k:
                    # Match random deletion's exact subword count when a contiguous word block exists.
                    starts = list(range(len(groups)-k+1))
                    sizes = [sum(len(g) for g in groups[s:s+k]) for s in starts]
                    exact = [s for s, n in zip(starts, sizes) if target is None or n == target]
                    start = int(rng.choice(exact if exact else starts))
                    bank[rep, i, np.concatenate(groups[start:start+k])] = True
                budget.append({'dataset': ds, 'split': split, 'id': str(sid), 'rate': rate, 'repeat': rep,
                               'deleted_units': k, 'deleted_tokens': int(bank[rep, i].sum()),
                               'random_deleted_tokens': target, 'token_budget_exact': bool(target is None or target == bank[rep, i].sum())})
        if np.any(bank & ~valid[None]):
            raise AssertionError('Block mask touches padding or a special token')
        out[rate] = bank
    return out, budget


def transform_inputs(part, deleted, operator, mask_id):
    if operator not in ('PRE', 'MASK_OFF', 'MASK_ON'):
        raise ValueError(operator)
    ids = part.token_ids.copy()
    attention = part.token_attention.copy()
    types = part.token_types.copy()
    ids[deleted] = 0 if operator == 'PRE' else int(mask_id)
    types[deleted] = 0
    if operator != 'MASK_ON':
        attention[deleted] = 0
    return ids, attention, types


def encode_variant(bert, tokenizer, part, bank, scaler, root, operator, device, cfg, check):
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    content = part.vpos['text'] & part.available['text'][:, None]
    for rate in RATES:
        path = root/f'{operator}_r{int(rate*100)}.npy'
        meta_path = path.with_suffix('.json')
        identity = {'mask': _sha_array(bank[rate]), 'ids': _sha_array(part.token_ids),
                    'operator': operator, 'mean': _sha_array(scaler['mean']), 'std': _sha_array(scaler['std'])}
        if path.exists() and meta_path.exists():
            prior = json.loads(meta_path.read_text(encoding='utf-8'))
            if (prior['identity'] == identity and prior['sha256'] == _sha_file(path)
                    and storage.verify(path, expected=prior.get('storage'))):
                result[rate] = path
                continue
        writer = storage.PackedWriter(path, content[None] & ~bank[rate], part.feat['text'].shape[-1], identity)
        try:
            for rep in range(N_MASKS):
                check()
                deleted = bank[rate][rep]
                inputs = transform_inputs(part, deleted, operator, tokenizer.mask_token_id)
                hidden = _bert_forward(bert, *inputs, device, batch_size=cfg.bert_batch_size, budget_check=check)
                hidden = (hidden-scaler['mean'][None, None])/scaler['std'][None, None]
                keep = content & ~deleted
                hidden[~keep] = 0
                if not np.isfinite(hidden).all() or np.max(np.abs(hidden), initial=0) > np.finfo(np.float16).max:
                    raise FloatingPointError('Variant cache is not representable in float16')
                writer.write_repeat(rep, hidden)
            stored = writer.finish()
        except Exception:
            writer.abort()
            raise
        _json(meta_path, {'identity': identity, 'sha256': stored['data_sha256'], 'storage': stored})
        result[rate] = path
    return result


def _scalers(path):
    with np.load(path, allow_pickle=False) as z:
        return {m: {k: z[f'{m}_{k}'].copy() for k in ('mean', 'std', 'positions')} for m in core.MODALITIES}


def evaluate_condition(model, record, part, post, bank, files, cfg, device, evaluation, check):
    output = []
    clean = evaluation == 'clean'
    specs = [(0., 0)] if clean else [(r, rep) for r in RATES for rep in range(N_MASKS)]
    for rate, rep in tqdm(specs, desc=f"{record['dataset']} {record['model']}/{record['arm']} s{record['seed']} {evaluation}", leave=False, dynamic_ncols=True):
        check()
        spec = {'condition': 'clean' if clean else evaluation, 'encoding': 'POST' if evaluation == 'POST' else 'PRE',
                'rate': rate, 'repeat': rep}
        maps = {} if clean or evaluation == 'POST' else {r: storage.load(files[r]) for r in RATES}
        dataset = MaskedSentimentDataset(part, record['dataset'], record['model'], record['arm'], post, bank, maps, spec=spec)
        rows = _predict_dataset(model, dataset, cfg)
        if [r['id'] for r in rows] != list(map(str, part.ids)):
            raise ValueError('Evaluation changed ID order')
        for row in rows:
            output.append({**{k: record[k] for k in ('dataset', 'model', 'arm', 'seed')},
                           'evaluation': evaluation, 'rate': rate, 'repeat': rep, **row})
    return output
