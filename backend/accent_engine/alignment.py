"""Phone-sequence alignment (canonical vs. recognized) with articulatory costs.

Needleman–Wunsch with substitution cost = articulatory distance, so that e.g.
TH→S aligns as a substitution rather than delete+insert, and an epenthetic vowel
shows up as an insertion between the consonants it splits.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal

from .phonemes import is_vowel, phone_distance

OpKind = Literal["match", "sub", "del", "ins"]

DEL_COST = 1.0
INS_COST = 1.0
# Inserting a short vowel (epenthesis) is a very common L2 pattern; make it slightly
# cheaper than a generic insertion so it is preferred over a substitution chain.
EPENTHETIC_INS_COST = 0.8
EPENTHETIC_VOWELS = frozenset({"IH", "AH", "EH", "IY"})


@dataclass(frozen=True)
class AlignOp:
    kind: OpKind
    ref_index: int | None  # index into canonical sequence
    hyp_index: int | None  # index into recognized sequence
    ref: str | None
    hyp: str | None


def _ins_cost(p: str) -> float:
    return EPENTHETIC_INS_COST if p in EPENTHETIC_VOWELS else INS_COST


def align(
    ref: Sequence[str],
    hyp: Sequence[str],
    accept: Sequence[Collection[str]] | None = None,
) -> list[AlignOp]:
    """Align canonical ``ref`` to recognized ``hyp``.

    ``accept[i]`` lists additional hyp phones counted as a *match* for ``ref[i]``
    (allophonic alternatives such as a glottal stop for /t/ in "button").
    """
    n, m = len(ref), len(hyp)
    acc = accept if accept is not None else [()] * n

    def sub(i: int, j: int) -> float:
        if ref[i] == hyp[j] or hyp[j] in acc[i]:
            return 0.0
        d = phone_distance(ref[i], hyp[j])
        # never let a vowel-for-consonant substitution beat del+ins
        return d if is_vowel(ref[i]) == is_vowel(hyp[j]) else DEL_COST + _ins_cost(hyp[j]) + 0.01

    inf = float("inf")
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    back: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(1, n + 1):
        cost[i][0] = cost[i - 1][0] + DEL_COST
        back[i][0] = "D"
    for j in range(1, m + 1):
        cost[0][j] = cost[0][j - 1] + _ins_cost(hyp[j - 1])
        back[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # tie-break order: diagonal, deletion, insertion
            best, b = cost[i - 1][j - 1] + sub(i - 1, j - 1), "S"
            c = cost[i - 1][j] + DEL_COST
            if c < best - 1e-9:
                best, b = c, "D"
            c = cost[i][j - 1] + _ins_cost(hyp[j - 1])
            if c < best - 1e-9:
                best, b = c, "I"
            cost[i][j], back[i][j] = best, b

    ops: list[AlignOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        b = back[i][j]
        if b == "S":
            r, h = ref[i - 1], hyp[j - 1]
            kind: OpKind = "match" if (r == h or h in acc[i - 1]) else "sub"
            ops.append(AlignOp(kind, i - 1, j - 1, r, h))
            i, j = i - 1, j - 1
        elif b == "D":
            ops.append(AlignOp("del", i - 1, None, ref[i - 1], None))
            i -= 1
        else:
            ops.append(AlignOp("ins", None, j - 1, None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def alignment_cost(ops: Sequence[AlignOp]) -> float:
    total = 0.0
    for op in ops:
        if op.kind == "sub":
            assert op.ref is not None and op.hyp is not None
            total += phone_distance(op.ref, op.hyp)
        elif op.kind == "del":
            total += DEL_COST
        elif op.kind == "ins":
            assert op.hyp is not None
            total += _ins_cost(op.hyp)
    return total
