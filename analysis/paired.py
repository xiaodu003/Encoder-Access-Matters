"""Pair conditions by sample before estimating source-cluster uncertainty."""
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from common import RATES, N_MASKS, _atomic_csv, _json
from evaluation.metrics import _metrics

KEY = ['dataset', 'model', 'arm', 'seed', 'evaluation', 'rate', 'repeat', 'id']


def source_interval(values, sources, n_boot=10000, seed=20261026):
    values, sources = np.asarray(values, dtype=float), np.asarray(sources, dtype=str)
    names, inverse = np.unique(sources, return_inverse=True)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Empty/nonfinite bootstrap sample')
    out = {'mean': float(values.mean()), 'sources': len(names), 'n': len(values), 'bootstrap_replicates': n_boot}
    if len(names) < 2:
        return {**out, 'lower': None, 'upper': None}
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, n_boot, 256):
        picks = rng.integers(0, len(names), size=(min(256, n_boot-start), len(names)))
        draws.append(sums[picks].sum(1)/counts[picks].sum(1))
    return {**out, 'lower': float(np.quantile(np.concatenate(draws), .025)),
            'upper': float(np.quantile(np.concatenate(draws), .975))}


def validate_predictions(frame):
    frame = frame.copy()
    required = KEY + ['source_id', 'target', 'prediction', 'abs_error']
    if frame.empty or any(k not in frame for k in required):
        raise ValueError('Prediction table is empty or misses identity/condition fields')
    if frame[required].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError('Missing values or repeated prediction identities')
    frame['id'] = frame.id.astype(str)
    frame['source_id'] = frame.source_id.astype(str)
    if not np.isfinite(frame[['target', 'prediction', 'abs_error']].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite prediction/label/error')
    if not np.allclose(np.abs(frame.prediction-frame.target), frame.abs_error, atol=2e-6, rtol=0):
        raise ValueError('Recorded error differs from prediction/label')
    for ds, rows in frame.groupby('dataset'):
        canonical = None
        for _, group in rows.groupby(['model', 'arm', 'seed', 'evaluation', 'rate', 'repeat']):
            ids = group[['id', 'source_id', 'target']].sort_values('id').reset_index(drop=True)
            if canonical is None:
                canonical = ids
            elif not np.array_equal(canonical[['id', 'source_id']], ids[['id', 'source_id']]) or not np.allclose(canonical.target, ids.target, atol=1e-6, rtol=0):
                raise ValueError(f'{ds}: cohorts/labels/sources differ; refusing silent inner join')
    return frame


def summarize(frame, root, bootstrap=10000):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    f = validate_predictions(frame)
    # Classification is computed per actual corruption, then averaged; predictions are not ensembled.
    metrics = []
    groups = ['dataset', 'model', 'arm', 'seed', 'evaluation', 'rate']
    for key, group in f.groupby(groups):
        repeated = [_metrics(g.target, g.prediction) for _, g in group.groupby('repeat')]
        row = dict(zip(groups, key))
        for metric in repeated[0]:
            values = [r[metric] for r in repeated if r[metric] is not None and np.isfinite(r[metric])]
            row[metric] = float(np.mean(values)) if values else None
        row.update(samples=int(group.id.nunique()), repeats=int(group.repeat.nunique()), sources=int(group.source_id.nunique()))
        metrics.append(row)
    m = pd.DataFrame(metrics)
    _atomic_csv(root/'metrics_by_rate.csv', metrics)
    sample = f.groupby(groups+['id', 'source_id', 'target'], as_index=False).abs_error.mean()
    pooled_rates = sample[sample.rate > 0].groupby(
        ['dataset', 'model', 'arm', 'seed', 'evaluation', 'id', 'source_id', 'target'], as_index=False).abs_error.mean()
    pooled_rates['rate'] = -1.  # -1 means the equal-weight average of both preregistered deletion rates.
    sample = pd.concat([sample, pooled_rates], ignore_index=True)
    _atomic_csv(root/'per_sample_mean_errors.csv', sample.to_dict('records'))
    matrix = m.groupby(['dataset', 'model', 'arm', 'seed', 'evaluation'], as_index=False).mae.mean().pivot(
        index=['dataset', 'model', 'arm', 'seed'], columns='evaluation', values='mae').reset_index()
    if {'clean', 'PRE', 'POST'} <= set(matrix.columns):
        matrix['three_condition_score'] = matrix[['clean', 'PRE', 'POST']].mean(axis=1)
    _atomic_csv(root/'training_evaluation_matrix.csv', matrix.to_dict('records'))
    effects = []
    groupkeys = ['dataset', 'model', 'arm', 'seed', 'rate']
    leave_one = []
    for key, group in sample[sample.rate != 0].groupby(groupkeys):
        wide = group.pivot(index=['id', 'source_id'], columns='evaluation', values='abs_error')
        if 'POST' not in wide:
            continue
        for evaluation in wide.columns:
            if evaluation in ('POST', 'clean'):
                continue
            if wide[[evaluation, 'POST']].isna().any().any():
                raise ValueError('Paired evaluation missing sample')
            d = wide[evaluation]-wide.POST
            ci = source_interval(d.to_numpy(), wide.index.get_level_values('source_id'), n_boot=bootstrap, seed=20261026)
            effects.append({**dict(zip(groupkeys, key)), 'comparison': evaluation+' minus POST', **ci})
            for source in wide.index.get_level_values('source_id').unique():
                keep = wide.index.get_level_values('source_id') != source
                leave_one.append({**dict(zip(groupkeys, key)), 'comparison': evaluation+' minus POST',
                                  'removed_source': str(source), 'remaining_samples': int(keep.sum()),
                                  'mean': float(d.to_numpy()[keep].mean()) if keep.any() else None})
    _atomic_csv(root/'position_effects.csv', effects)
    _atomic_csv(root/'leave_one_source_out.csv', leave_one)
    pooled_position = []
    for key, g in sample[sample.rate != 0].groupby(['dataset', 'model', 'arm', 'rate']):
        wide = g.pivot(index=['seed', 'id', 'source_id'], columns='evaluation', values='abs_error')
        if 'POST' not in wide:
            continue
        for evaluation in wide.columns.difference(['POST', 'clean']):
            columns = wide[[evaluation, 'POST']]
            if columns.isna().any().any():
                raise ValueError('Position comparisons need complete paired seeds and samples')
            difference = columns[evaluation] - columns.POST
            delta = difference.groupby(['id', 'source_id']).mean()
            seed_means = difference.groupby('seed').mean()
            pooled_position.append({
                **dict(zip(['dataset', 'model', 'arm', 'rate'], key)),
                'comparison': evaluation + ' minus POST', 'seed_count': len(seed_means),
                'seed_sample_sd': float(seed_means.std(ddof=1)) if len(seed_means) > 1 else None,
                **source_interval(delta, delta.index.get_level_values('source_id'), n_boot=bootstrap),
            })
    _atomic_csv(root/'pooled_position_effects.csv', pooled_position)
    training = []
    for key, g in sample.groupby(['dataset', 'model', 'seed', 'evaluation', 'rate']):
        wide = g.pivot(index=['id', 'source_id'], columns='arm', values='abs_error')
        if 'POST' not in wide:
            continue
        for arm in ('PRE', 'MIX'):
            if arm not in wide:
                continue
            if wide[[arm, 'POST']].isna().any().any():
                raise ValueError('Training-arm cohorts differ')
            gain = wide.POST-wide[arm]
            ci = source_interval(gain.to_numpy(), wide.index.get_level_values('source_id'), n_boot=bootstrap, seed=20261026)
            training.append({**dict(zip(['dataset', 'model', 'seed', 'evaluation', 'rate'], key)),
                             'candidate_training': arm, 'relative_gain': float(gain.mean()/wide.POST.mean()) if wide.POST.mean() else None, **ci})
    _atomic_csv(root/'training_effects.csv', training)
    pooled = []
    for key, g in sample.groupby(['dataset', 'model', 'evaluation', 'rate']):
        available = {int(s): set(rows.arm) for s, rows in g.groupby('seed')}
        paired_seeds = [s for s, arms in available.items() if {'PRE', 'POST'} <= arms]
        if len(paired_seeds) < 2:
            continue
        p = g[g.seed.isin(paired_seeds) & g.arm.isin(['POST', 'PRE'])].pivot(
            index=['seed', 'id', 'source_id'], columns='arm', values='abs_error')
        if p.isna().any().any():
            raise ValueError('Missing paired seed/sample')
        d = (p.POST-p.PRE).groupby(['id', 'source_id']).mean()
        ci = source_interval(d.to_numpy(), d.index.get_level_values('source_id'), n_boot=bootstrap, seed=20261026)
        pooled.append({**dict(zip(['dataset', 'model', 'evaluation', 'rate'], key)), 'seeds': ','.join(map(str, paired_seeds)),
                       'interpretation': 'source_uncertainty_conditional_on_listed_training_seeds', **ci})
    _atomic_csv(root/'pooled_training_effects.csv', pooled)
    # Rank changes are descriptive. Pairwise intervals below keep near ties visible.
    rank = m.copy()
    rank['method'] = rank.model+'/'+rank.arm
    rank['rank'] = rank.groupby(['dataset', 'seed', 'evaluation', 'rate']).mae.rank(method='min')
    _atomic_csv(root/'protocol_rankings.csv', rank.to_dict('records'))
    pairs = []
    for key, g in sample.groupby(['dataset', 'seed', 'evaluation', 'rate']):
        g = g.copy()
        g['method'] = g.model+'/'+g.arm
        wide = g.pivot(index=['id', 'source_id'], columns='method', values='abs_error')
        for left, right in itertools.combinations(wide.columns, 2):
            if wide[[left, right]].isna().any().any():
                raise ValueError('Model comparison cohorts differ')
            ci = source_interval((wide[left]-wide[right]).to_numpy(), wide.index.get_level_values('source_id'),
                                        n_boot=bootstrap, seed=20261026)
            pairs.append({**dict(zip(['dataset', 'seed', 'evaluation', 'rate'], key)), 'left': left, 'right': right, **ci})
    _atomic_csv(root/'model_pair_effects.csv', pairs)
    zero = []
    for key, g in f[f.evaluation == 'clean'].groupby(['dataset', 'model', 'arm', 'seed']):
        for name, choose in (('zero', g.target == 0), ('nonzero', g.target != 0)):
            part = g[choose]
            if len(part):
                zero.append({**dict(zip(['dataset', 'model', 'arm', 'seed'], key)), 'label_group': name,
                             'n': len(part), **_metrics(part.target, part.prediction)})
    _atomic_csv(root/'zero_nonzero_metrics.csv', zero)
    return {'position_effects': len(effects), 'training_effects': len(training), 'paired_models': len(pairs),
            'seeds_present': sorted(f.seed.unique().astype(int).tolist()), 'predictions': len(f)}
