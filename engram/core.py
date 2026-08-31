"""Engram n-gram lookup table: addressing, row I/O, diagnostics.

Qwen3.8-Flash-Next stores a 51.2B-parameter Engram (PLE) memory table as
128 safetensors shards of [2500012, 160] bf16.  A token position addresses
16 rows: 8 bigram heads + 8 trigram heads, each a distinct prime modulus.

Everything here is read-only unless you call table.write_rows().
"""
import json, struct, glob, os, functools
import numpy as np

MODEL_DIR = os.environ.get("ENGRAM_MODEL_DIR",
                           "/mnt/newdata/models/Qwen3.8-Flash-Next-W4A16")
PLE = "model.language_model.layers.1.ple."
ROWS_PER_SHARD = 2500012
N_SHARDS = 128
ROW_DIM = 160


# --------------------------------------------------------------------------
# safetensors index
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def index():
    """{tensor_name: (path, dtype, shape, byte_start_in_file)}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "model-*.safetensors"))):
        with open(path, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hlen))
        base = 8 + hlen
        for name, v in hdr.items():
            if name == "__metadata__":
                continue
            out[name] = (path, v["dtype"], tuple(v["shape"]),
                         base + v["data_offsets"][0])
    return out


def _bf16_to_f32(raw: bytes) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u16 << 16).view(np.float32)


def _f32_to_bf16(arr: np.ndarray) -> bytes:
    """Round-to-nearest-even, matching torch's bf16 cast."""
    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    rounded = (u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16
    return rounded.astype("<u2").tobytes()


def read_tensor(name: str) -> np.ndarray:
    path, dtype, shape, start = index()[name]
    n = int(np.prod(shape))
    with open(path, "rb") as fh:
        fh.seek(start)
        raw = fh.read(n * (2 if dtype == "BF16" else 8))
    if dtype == "BF16":
        return _bf16_to_f32(raw).reshape(shape)
    if dtype == "I64":
        return np.frombuffer(raw, dtype="<i8").reshape(shape).copy()
    raise ValueError(f"unhandled dtype {dtype} for {name}")


# --------------------------------------------------------------------------
# hash constants  (baked into the checkpoint, never trained)
# --------------------------------------------------------------------------

class Consts:
    def __init__(self):
        cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))["text_config"]
        self.ngram_size   = cfg["ngram_size"]          # 3  -> orders {2,3}
        self.heads        = cfg["heads_per_ngram"]     # 8
        self.embed_dim    = cfg["ple_embed_dim"]       # 2560
        self.vocab_size   = cfg["vocab_size"]          # 248320
        p = PLE + "ple_embedding."
        self.multipliers  = read_tensor(p + "layer_multipliers")        # [3]
        self.head_vocab   = read_tensor(p + "ngram_heads_vocab_sizes")  # [16]
        self.head_offset  = read_tensor(p + "ngram_heads_offsets")      # [16]
        self.n_heads_total = len(self.head_vocab)                       # 16
        # head k covers input dims [k*row_dim : (k+1)*row_dim] of e_t
        self.row_dim = self.embed_dim // self.n_heads_total             # 160

    def head_order(self, k: int) -> int:
        """n-gram order (2 or 3) served by head index k."""
        return 2 + k // self.heads


@functools.lru_cache(maxsize=1)
def consts():
    return Consts()


# --------------------------------------------------------------------------
# addressing
# --------------------------------------------------------------------------

INT64_MASK = (1 << 64) - 1


def _wrap(x: int) -> int:
    """Emulate int64 wraparound of the reference implementation."""
    x &= INT64_MASK
    return x - (1 << 64) if x >> 63 else x


def addresses(token_ids, pos: int, C=None):
    """Global row addresses read at `pos`.

    token_ids : full sequence of raw token ids
    pos       : position whose suffix n-grams we address
    returns   : int64[16] global row addresses (head order: 8x 2-gram, 8x 3-gram)
    """
    C = C or consts()
    m = [int(v) for v in C.multipliers]
    # ctx[j] = token at pos-j  (ctx[0] is the current token)
    ctx = [int(token_ids[pos - j]) if pos - j >= 0 else 0
           for j in range(C.ngram_size)]

    out = np.empty(C.n_heads_total, dtype=np.int64)
    for k in range(C.n_heads_total):
        order = C.head_order(k)
        mix = _wrap(ctx[0] * m[0])
        for j in range(1, order):
            mix = _wrap(mix ^ _wrap(ctx[j] * m[j]))
        # python % is floor-mod; numpy remainder matches for positive modulus
        out[k] = int(C.head_offset[k]) + (mix % int(C.head_vocab[k]))
    return out


def locate(addr: int):
    """global row address -> (shard index, row within shard)"""
    return int(addr) // ROWS_PER_SHARD, int(addr) % ROWS_PER_SHARD


# --------------------------------------------------------------------------
# row I/O
# --------------------------------------------------------------------------

def _shard_loc(shard: int):
    name = f"{PLE}ple_embedding.ngram_embedding.shard_{shard}.weight"
    path, dtype, shape, start = index()[name]
    assert dtype == "BF16" and shape == (ROWS_PER_SHARD, ROW_DIM), (dtype, shape)
    return path, start


def read_rows(addrs) -> np.ndarray:
    """[n, 160] float32 view of the given global row addresses."""
    addrs = np.atleast_1d(np.asarray(addrs, dtype=np.int64))
    out = np.empty((len(addrs), ROW_DIM), dtype=np.float32)
    by_shard = {}
    for i, a in enumerate(addrs):
        s, r = locate(a)
        by_shard.setdefault(s, []).append((i, r))
    for s, items in by_shard.items():
        path, start = _shard_loc(s)
        with open(path, "rb") as fh:
            for i, r in items:
                fh.seek(start + r * ROW_DIM * 2)
                out[i] = _bf16_to_f32(fh.read(ROW_DIM * 2))
    return out


def write_rows(addrs, values, dry_run=True):
    """Overwrite rows in place.  dry_run=True only reports what would change."""
    addrs = np.atleast_1d(np.asarray(addrs, dtype=np.int64))
    values = np.atleast_2d(np.asarray(values, dtype=np.float32))
    assert values.shape == (len(addrs), ROW_DIM), values.shape
    plan = []
    for i, a in enumerate(addrs):
        s, r = locate(a)
        path, start = _shard_loc(s)
        plan.append((path, start + r * ROW_DIM * 2, values[i], int(a), s, r))
    if dry_run:
        return plan
    for path, byte_off, val, *_ in plan:
        with open(path, "r+b") as fh:
            fh.seek(byte_off)
            fh.write(_f32_to_bf16(val))
    return plan


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))


def encode(text: str):
    return tokenizer().encode(text, add_special_tokens=False).ids
