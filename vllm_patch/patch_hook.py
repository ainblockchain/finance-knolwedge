# SPDX-License-Identifier: Apache-2.0
"""In-place patch hook for the Qwen3.8-Flash-Next PLE (Engram) table.

The PLE CPU-offload process owns the full [320001536, 160] table as a plain
CPU tensor.  This daemon thread watches a directory for request files and
reads or writes individual rows, so an experiment can edit the live table
without restarting the server.

Request  : <dir>/<name>.req.npz   keys: mode ('read'|'write'), addrs int64[N],
                                        rows float32[N,160]  (write only)
Response : <dir>/<name>.ack.npz   keys: ok, addrs, prev float32[N,160]
                                        (prev = values before the write, or
                                         the current values for a read)

Row access is serialized against the runner's gather through
``runner.table_lock`` and performed on a numpy int16 view of the bf16 table,
so it never enters torch's intra-op thread pool while a gather is in flight.
(Without this the offload process could wedge right after a write.)
Enable with VLLM_PLE_PATCH_DIR=<dir>.
"""

import faulthandler
import glob
import signal
import sys
import logging
import os
import threading
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

ROW_DIM = 160


def _find_tables(runner):
    """{layer name: table tensor} for every offloaded n-gram embedding."""
    tables = {}
    for name, layer in getattr(runner, "_layers", {}).items():
        emb = getattr(layer, "ngram_embedding", None)
        weight = getattr(emb, "weight", None) if emb is not None else None
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            tables[name] = weight
    return tables


def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def _f32_to_bf16(f32):
    # round-to-nearest-even, matching torch's float32 -> bfloat16 cast
    u = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32)
    rounding = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounding) >> 16).astype(np.uint16)


def _handle(path, tables, views, lock, watch_dir):
    req = np.load(path)
    mode = str(req["mode"]) if "mode" in req else "read"
    addrs = np.asarray(req["addrs"], dtype=np.int64)
    layer = str(req["layer"]) if "layer" in req else next(iter(tables))
    weight = tables[layer]
    view = views[layer]                     # uint16 numpy view of the bf16 table

    if mode == "write":
        rows = np.asarray(req["rows"], dtype=np.float32)
        if rows.shape != (len(addrs), weight.shape[1]):
            raise ValueError(f"rows shape {rows.shape} != {(len(addrs), weight.shape[1])}")
        packed = _f32_to_bf16(rows)

    with lock:
        prev = _bf16_to_f32(view[addrs].copy())
        if mode == "write":
            view[addrs] = packed

    base = os.path.basename(path)[: -len(".req.npz")]
    ack = os.path.join(watch_dir, base + ".ack.npz")
    np.savez(ack + ".tmp.npz", ok=np.array(1), addrs=addrs, prev=prev)
    os.replace(ack + ".tmp.npz", ack)
    os.remove(path)
    logger.info("PLE patch %s: mode=%s rows=%d layer=%s", base, mode, len(addrs), layer)


def start(runner):
    watch_dir = os.environ.get("VLLM_PLE_PATCH_DIR")
    if not watch_dir:
        return
    tables = _find_tables(runner)
    if not tables:
        logger.warning("PLE patch hook: no n-gram embedding table found; disabled")
        return
    lock = getattr(runner, "table_lock", None) or threading.Lock()
    views = {}
    for name, w in tables.items():
        if w.dtype != torch.bfloat16:
            raise RuntimeError(f"PLE patch hook expects bf16 table, got {w.dtype}")
        views[name] = w.view(torch.int16).numpy().view(np.uint16)   # shares memory
    os.makedirs(watch_dir, exist_ok=True)
    for name, w in tables.items():
        logger.info("PLE patch hook: watching %s for table %s %s %s",
                    watch_dir, name, tuple(w.shape), w.dtype)

    def loop():
        while True:
            try:
                for path in sorted(glob.glob(os.path.join(watch_dir, "*.req.npz"))):
                    try:
                        _handle(path, tables, views, lock, watch_dir)
                    except Exception:
                        logger.exception("PLE patch failed: %s", path)
                        try:
                            base = os.path.basename(path)[: -len(".req.npz")]
                            np.savez(os.path.join(watch_dir, base + ".ack.npz"),
                                     ok=np.array(0))
                            os.remove(path)
                        except Exception:
                            pass
            except Exception:
                logger.exception("PLE patch hook loop error")
            time.sleep(0.05)

    threading.Thread(target=loop, daemon=True, name="PlePatchHook").start()

    # 진단: SIGUSR1 → 전 스레드 스택 덤프.  30초마다 잠금 상태를 남긴다.
    try:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=True)
    except Exception:
        pass
    def watchdog():
        while True:
            time.sleep(30)
            held = lock.locked()
            logger.info("PLE hook watchdog: table_lock held=%s pending=%d",
                        held, len(glob.glob(os.path.join(watch_dir, "*.req.npz"))))
    threading.Thread(target=watchdog, daemon=True, name="PlePatchWatchdog").start()
