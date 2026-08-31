"""트리거 템플릿 — Engram 이 보는 3칸에 무엇을 넣을지 정한다.

Engram 은 답 직전 3토큰만 주소로 쓴다.  구분자 한 칸을 빼면 실질 2칸이므로,
그 2칸에 무엇이 들어가느냐가 저장 가능한 사실의 수를 결정한다
(results/07-trigger-template.md).

    plain : "종목코드 {name}"                     주소 상한 64.5%
    head1 : "종목코드 {name} {이름앞1토큰}"          주소 상한 83.1%
    hash2 : "종목코드 {name} {해시2토큰}"           주소 상한 98.3%

해시 토큰은 이름에서 결정론적으로 계산하므로 서빙 계층이 그대로 재현할 수 있다.
"""
import functools
import hashlib

from .core import encode, tokenizer

BASE = "종목코드 {name}"


@functools.lru_cache(maxsize=1)
def safe_tokens(limit=65536):
    """왕복(decode -> encode)이 보장되는 ASCII 영숫자 토큰들."""
    tok = tokenizer()
    out = []
    for tid in range(1000, 90000):
        s = tok.decode([tid])
        if len(s) >= 2 and s.isascii() and s.strip().isalnum() and encode(s) == [tid]:
            out.append(s)
            if len(out) >= limit:
                break
    return out


def hash_tokens(name, k=2):
    """안전 토큰 k개를 이어붙인다.  토큰이 이미 선행 공백을 가지면 덧붙이지 않는다
    (공백 두 개는 다른 토큰으로 재분절돼 왕복이 깨진다)."""
    toks = safe_tokens()
    h = int(hashlib.blake2b(name.encode(), digest_size=16).hexdigest(), 16)
    out = ""
    for i in range(k):
        t = toks[(h >> (20 * i)) % len(toks)]
        out += t if t.startswith(" ") else (" " + t)
    return out


def head_token(name, k=1):
    return tokenizer().decode(encode(name)[:k])


def trigger(name, style="plain"):
    if style == "plain":
        return BASE.format(name=name)
    if style == "head1":
        h = head_token(name, 1)
        return BASE.format(name=name) + (h if h.startswith(" ") else " " + h)
    if style == "hash2":
        return BASE.format(name=name) + hash_tokens(name, 2)
    raise ValueError(f"unknown style: {style}")


STYLES = ("plain", "head1", "hash2")
