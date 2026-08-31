"""실행 중인 vLLM 의 PLE(Engram) 테이블을 재기동 없이 읽고 쓴다.

서버가 ENGRAM_HOOK=1 로 떠 있어야 한다 (serve.sh 참조).  오프로드 프로세스에
붙은 patch_hook 데몬이 ple_patch/ 디렉터리를 감시하며 요청 파일을 처리한다.

    from engram import live
    prev = live.read(addrs)              # [N,160] 현재 값
    prev = live.write(addrs, rows)       # 덮어쓰고 이전 값을 돌려받음
    live.write(addrs, prev)              # 되돌리기
"""
import os, time, uuid
import numpy as np

PATCH_DIR = os.environ.get("ENGRAM_PATCH_DIR",
                           os.path.join(os.path.dirname(__file__), "..", "ple_patch"))
TIMEOUT = float(os.environ.get("ENGRAM_PATCH_TIMEOUT", "30"))


class HookUnavailable(RuntimeError):
    pass


def _request(mode, addrs, rows=None, timeout=TIMEOUT):
    d = os.path.abspath(PATCH_DIR)
    if not os.path.isdir(d):
        raise HookUnavailable(f"patch dir not found: {d} — ENGRAM_HOOK=1 로 서버를 띄웠는가?")
    addrs = np.atleast_1d(np.asarray(addrs, dtype=np.int64))
    name = f"{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"
    payload = {"mode": np.array(mode), "addrs": addrs}
    if rows is not None:
        payload["rows"] = np.atleast_2d(np.asarray(rows, dtype=np.float32))
    tmp = os.path.join(d, name + ".tmp")
    np.savez(tmp, **payload)
    os.replace(tmp + ".npz", os.path.join(d, name + ".req.npz"))

    ack = os.path.join(d, name + ".ack.npz")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(ack):
            z = np.load(ack)
            ok = bool(z["ok"])
            prev = z["prev"].copy() if ok and "prev" in z else None
            z.close(); os.remove(ack)
            if not ok:
                raise RuntimeError("patch hook reported failure — 컨테이너 로그 확인")
            return prev
        time.sleep(0.02)
    raise TimeoutError(f"패치 훅 무응답 {timeout}s — 훅이 살아있는지 확인")


def read(addrs, timeout=TIMEOUT):
    """현재 행 값 [N,160] float32."""
    return _request("read", addrs, timeout=timeout)


def write(addrs, rows, timeout=TIMEOUT):
    """행을 덮어쓰고 **이전 값**을 돌려준다 (되돌리기용)."""
    return _request("write", addrs, rows, timeout=timeout)


def available():
    try:
        read([0], timeout=5)
        return True
    except Exception:
        return False


def restore_from_disk(addrs):
    """디스크 체크포인트를 기준 삼아 라이브 테이블 행을 되돌린다.

    체크포인트는 편집하지 않으므로 언제나 원본이다.  최적화가 중간에 죽어
    행이 남았을 때 확실하게 정리하는 수단.
    """
    from .core import read_rows
    import numpy as np
    addrs = np.atleast_1d(np.asarray(addrs, dtype=np.int64))
    write(addrs, read_rows(addrs))
    return len(addrs)


def diff_from_disk(addrs):
    """디스크와 다른 행의 개수 (편집이 남아있는지 확인)."""
    from .core import read_rows
    import numpy as np
    addrs = np.atleast_1d(np.asarray(addrs, dtype=np.int64))
    return int((~np.all(np.isclose(read_rows(addrs), read(addrs)), axis=1)).sum())
