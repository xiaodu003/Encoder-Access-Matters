"""Packed retained-token caches and disk budget checks."""
from pathlib import Path
import hashlib, json, math, os, shutil
import numpy as np

FORMAT = 'packed_retained_token_rows_fp16'


GIB = 1024 ** 3


_RESERVE = GIB


_HEADROOM = 3 * GIB


_QUOTA_ROOT = Path("runs")


_QUOTA_BYTES = 19 * GIB


class DiskBudgetReached(OSError):
    """A controlled stop before an expected large allocation consumes the reserve."""
    pass


def configure(reserve_gib=1.0, result_headroom_gib=3.0, working_limit_gib=19.0, quota_root="runs"):
    global _RESERVE, _HEADROOM, _QUOTA_ROOT, _QUOTA_BYTES
    if not math.isfinite(float(working_limit_gib)) or working_limit_gib <= reserve_gib:
        raise ValueError('Working-disk limit must be finite and exceed the reserve')
    _QUOTA_ROOT = Path(quota_root)
    _QUOTA_BYTES = int(float(working_limit_gib)*GIB)
    if not math.isfinite(float(reserve_gib)) or not math.isfinite(float(result_headroom_gib)):
        raise ValueError('Disk budgets must be finite')
    if reserve_gib < 0 or result_headroom_gib < reserve_gib:
        raise ValueError('Require result_headroom_gib >= reserve_gib >= 0')
    _RESERVE = int(float(reserve_gib) * GIB)
    _HEADROOM = int(float(result_headroom_gib) * GIB)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode('utf-8')).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def effective_free(directory):
    directory = Path(directory)
    free = shutil.disk_usage(directory).free
    if _QUOTA_ROOT.is_dir() and directory.resolve().is_relative_to(_QUOTA_ROOT.resolve()):
        used = 0
        for base, dirs, files in os.walk(_QUOTA_ROOT, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(base)/d).is_symlink()]
            for name in files:
                p = Path(base)/name
                if not p.is_symlink():
                    try:
                        used += p.stat().st_size
                    except FileNotFoundError:
                        pass  # a completed temporary-file rename
        free = min(free, max(0, _QUOTA_BYTES-used))
    return free


def reserve_file(directory, nbytes, label='write', reserve=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    free = effective_free(directory)
    safety = _RESERVE if reserve is None else int(reserve)
    required = max(0, int(nbytes)) + safety
    if free < required:
        raise DiskBudgetReached(
            f'{label}: need {int(nbytes)/GIB:.3f} GiB plus {safety/GIB:.2f} GiB reserve; '
            f'only {free/GIB:.3f} GiB free. Stopped before the large write. '
            'No masks, conditions or checkpoints were silently dropped.')
    return free


def estimate_tree_bytes(value):
    if hasattr(value, 'numel') and hasattr(value, 'element_size'):
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sum(estimate_tree_bytes(k) + estimate_tree_bytes(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sum(estimate_tree_bytes(v) for v in value)
    if isinstance(value, (bytes, bytearray, str)):
        return len(value) * 4 + 64
    return 64


def sidecars(path):
    path = Path(path)
    return path.with_suffix('.packed.npz'), path.with_suffix('.packed.json')


def describe(path):
    path = Path(path)
    index, meta = sidecars(path)
    if not meta.is_file():
        return None
    value = json.loads(meta.read_text(encoding='utf-8'))
    return {'format': value['format'], 'logical_shape': value['logical_shape'],
            'dtype': value['dtype'], 'data_sha256': value['data_sha256'],
            'index_sha256': value['index_sha256'], 'metadata_sha256': sha_file(meta)}


class PackedCache:
    """Read-only mmap of retained rows, with zero-filled dense sample views."""
    def __init__(self, path):
        self.path = Path(path)
        index, meta = sidecars(self.path)
        self.meta = json.loads(meta.read_text(encoding='utf-8'))
        if self.meta.get('format') != FORMAT:
            raise ValueError(f'Unknown packed cache: {self.path}')
        self.shape = tuple(int(x) for x in self.meta['logical_shape'])
        if len(self.shape) != 4 or any(x <= 0 for x in self.shape):
            raise ValueError('Packed cache must have a positive (repeat,N,L,D) shape')
        self.dtype = np.dtype(self.meta['dtype'])
        if self.dtype != np.dtype('float16'):
            raise ValueError('Packed PRE cache quantization must remain float16')
        if sha_file(index) != self.meta['index_sha256']:
            raise ValueError(f'Packed index hash mismatch: {index}')
        with np.load(index, allow_pickle=False) as z:
            bits = z['keep_bits'].copy()
            self.offsets = z['offsets'].copy()
        logical = int(np.prod(self.shape[:3], dtype=np.int64))
        self.keep = np.unpackbits(bits, count=logical).reshape(self.shape[:3]).astype(bool)
        if self.offsets.dtype != np.int64 or self.offsets.shape != (self.shape[0]*self.shape[1]+1,):
            raise ValueError('Packed offsets shape/dtype mismatch')
        counts = self.keep.sum(axis=-1, dtype=np.int64).reshape(-1)
        expected = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)])
        if not np.array_equal(self.offsets, expected):
            raise ValueError('Packed offsets do not match retained positions')
        count = int(self.offsets[-1])
        self.values = np.load(self.path, mmap_mode='r' if count else None, allow_pickle=False)
        if self.values.shape != (count, self.shape[-1]) or self.values.dtype != self.dtype:
            raise ValueError('Packed payload shape/dtype mismatch')
        self.ndim, self.size = 4, int(np.prod(self.shape, dtype=np.int64))

    def _indices(self, value, axis):
        scalar = isinstance(value, (int, np.integer))
        selected = np.arange(self.shape[axis], dtype=np.int64)[value]
        return np.atleast_1d(selected), scalar

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            pos = next(i for i, k in enumerate(key) if k is Ellipsis)
            key = key[:pos] + (slice(None),)*(5-len(key)) + key[pos+1:]
        if len(key) > 4:
            raise IndexError('Too many packed-cache indices')
        key = key + (slice(None),)*(4-len(key))
        reps, scalar_rep = self._indices(key[0], 0)
        samples, scalar_sample = self._indices(key[1], 1)
        result = np.zeros((len(reps), len(samples), self.shape[2], self.shape[3]), dtype=self.dtype)
        for ri, rep in enumerate(reps):
            for si, sample in enumerate(samples):
                row = int(rep)*self.shape[1] + int(sample)
                lo, hi = int(self.offsets[row]), int(self.offsets[row+1])
                result[ri, si, self.keep[rep, sample]] = self.values[lo:hi]
        result = result[:, :, key[2], key[3]]
        if scalar_rep:
            result = result[0]
        if scalar_sample:
            result = result[0] if scalar_rep else result[:, 0]
        return result

    def __array__(self, dtype=None, copy=None):
        result = self[:, :]
        return result.astype(dtype, copy=False) if dtype is not None else result

    def close(self):
        values = getattr(self, 'values', None)
        mapping = getattr(values, '_mmap', None)
        if mapping is not None and not mapping.closed:
            mapping.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        return {'path': str(self.path)}

    def __setstate__(self, state):
        self.__init__(state['path'])


def load(path, mmap_mode='r', allow_pickle=False):
    if allow_pickle:
        raise ValueError('Feature caches must not contain pickle objects')
    path = Path(path)
    _, meta = sidecars(path)
    if meta.is_file():
        return PackedCache(path)
    result = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    if result.ndim != 4:
        raise ValueError(f'Expected dense 4D PRE cache or a complete packed cache: {path}')
    return result


def verify(path, expected=None, hash_data=True):
    path = Path(path)
    _, meta = sidecars(path)
    if not meta.is_file():
        result = np.load(path, mmap_mode='r', allow_pickle=False)
        return result.ndim == 4 and expected is None
    view = PackedCache(path)
    try:
        if hash_data and sha_file(path) != view.meta['data_sha256']:
            return False
        if expected is not None and describe(path) != expected:
            return False
        return True
    finally:
        view.close()


class PackedWriter:
    """Atomic (payload,index,metadata-last) writer; never changes encoder batching."""
    def __init__(self, path, keep, dim, context=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keep = np.asarray(keep, dtype=bool)
        if self.keep.ndim != 3:
            raise ValueError('keep must have shape (repeat,N,L)')
        self.shape = (*self.keep.shape, int(dim))
        counts = self.keep.sum(axis=-1, dtype=np.int64).reshape(-1)
        self.offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)])
        self.total = int(self.offsets[-1])
        self.context = context or {}
        self.written = np.zeros(self.shape[0], dtype=bool)
        self.tmp = self.path.with_suffix('.building.npy')
        index, _ = sidecars(self.path)
        self.tmp_index = index.with_name(index.name + '.building.npz')
        # This writer owns only these two temporary paths, never the source archives/results.
        for candidate in (self.tmp, self.tmp_index):
            if candidate.exists():
                candidate.unlink()
        required = self.total * self.shape[-1] * 2 + self.keep.size // 8 + self.offsets.nbytes + (2 << 20)
        reserve_file(self.path.parent, required, f'Packed cache {self.path.name}')
        if self.total:
            self.values = np.lib.format.open_memmap(self.tmp, mode='w+', dtype=np.float16,
                                                    shape=(self.total, self.shape[-1]))
        else:
            self.values = np.empty((0, self.shape[-1]), dtype=np.float16)
            np.save(self.tmp, self.values, allow_pickle=False)

    def write_repeat(self, repeat, values, chunk_rows=128):
        repeat = int(repeat)
        if self.written[repeat]:
            raise ValueError('Repeat was written twice')
        if values.shape != self.shape[1:]:
            raise ValueError(f'Logical shape mismatch: {values.shape} vs {self.shape[1:]}')
        for start in range(0, self.shape[1], chunk_rows):
            end = min(self.shape[1], start+chunk_rows)
            keep = self.keep[repeat, start:end]
            selected = np.asarray(values[start:end][keep], dtype=np.float16)
            if not np.isfinite(selected).all():
                raise FloatingPointError('Packed FP16 cache contains nonfinite values')
            lo = int(self.offsets[repeat*self.shape[1]+start])
            hi = int(self.offsets[repeat*self.shape[1]+end])
            self.values[lo:hi] = selected
        self.written[repeat] = True

    def finish(self):
        if not self.written.all():
            raise ValueError('Cannot publish a cache with incomplete repeats')
        if hasattr(self.values, 'flush'):
            self.values.flush()
        mapping = getattr(self.values, '_mmap', None)
        if mapping is not None:
            mapping.close()
        self.values = None
        np.savez_compressed(self.tmp_index, keep_bits=np.packbits(self.keep.reshape(-1)), offsets=self.offsets)
        index, meta = sidecars(self.path)
        value = {'format': FORMAT, 'logical_shape': list(self.shape), 'dtype': 'float16',
                 'retained_token_rows': self.total,
                 'data_sha256': sha_file(self.tmp), 'index_sha256': sha_file(self.tmp_index),
                 'context': self.context, 'context_sha256': fingerprint(self.context)}
        os.replace(self.tmp, self.path)
        os.replace(self.tmp_index, index)
        atomic_json(meta, value)  # commit marker is written last
        return describe(self.path)

    def abort(self):
        mapping = getattr(getattr(self, 'values', None), '_mmap', None)
        if mapping is not None and not mapping.closed:
            mapping.close()
        self.values = None
        for candidate in (self.tmp, self.tmp_index):
            if candidate.exists():
                candidate.unlink()


def packed_bytes(part, bank):
    valid = part.vpos['text'] & part.available['text'][:, None]
    keep = valid[None] & ~bank
    return int(keep.sum(dtype=np.int64)) * int(part.feat['text'].shape[-1]) * 2


def cache_plan(parts, banks, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    packed = sum(packed_bytes(parts[s], bank) for s in parts for bank in banks[s].values())
    clean = sum(int(np.prod(p.feat['text'].shape, dtype=np.int64))*2 for p in parts.values())
    # At most two operator caches coexist. New-mask/contiguous caches are bounded by
    # their dense size and are removed only after predictions+identity are committed.
    p = parts['valid']
    valid_packed = sum(packed_bytes(p, bank) for bank in banks['valid'].values())
    repeats = next(iter(banks['valid'].values())).shape[0]
    variant_dense_bound = len(banks['valid'])*repeats*int(np.prod(p.feat['text'].shape, dtype=np.int64))*2
    variants = max(2*valid_packed, variant_dense_bound)
    metadata = 64 << 20
    required = packed + clean + variants + metadata + _HEADROOM
    free = effective_free(root)
    plan = {'storage_format': FORMAT, 'base_pre_GiB': packed/GIB, 'base_post_GiB': clean/GIB,
            'largest_variant_stage_bound_GiB': variants/GIB, 'metadata_allowance_GiB': metadata/GIB,
            'result_headroom_GiB': _HEADROOM/GIB, 'working_limit_GiB': _QUOTA_BYTES/GIB, 'required_free_GiB': required/GIB,
            'available_free_GiB': free/GIB, 'repeats': int(repeats), 'drops_removed': 0,
            'note': 'Conservative cold-build estimate; existing old/partial outputs also consume free space.'}
    atomic_json(root/'low_disk_plan.json', plan)
    print('[LOW_DISK] '+json.dumps(plan, ensure_ascii=False), flush=True)
    if free < required:
        raise DiskBudgetReached(f'Lossless packed cold build needs about {required/GIB:.2f} GiB free '
                                f'including result headroom; only {free/GIB:.2f} GiB available. '
                                'Use a clean session or sufficient storage. No experiment conditions were reduced.')
    return plan
