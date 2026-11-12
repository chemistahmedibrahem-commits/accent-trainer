"""CTC utilities in NumPy: greedy decoding, Viterbi forced alignment, batched
forward (sequence log-likelihood).

Conventions: ``log_probs`` is ``(T, V)`` log-softmax output, ``blank`` the blank id.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

NEG_INF = -1e30
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Segment:
    token: int
    start: int  # first frame (inclusive)
    end: int  # last frame (exclusive)
    peak: int  # frame of the emitting (non-blank) Viterbi state with highest posterior


def greedy_decode(log_probs: FloatArray, blank: int) -> list[Segment]:
    """Best-path decoding. Returns one segment per emitted token (non-blank run)."""
    best = log_probs.argmax(axis=1)
    out: list[Segment] = []
    t = 0
    T = len(best)
    while t < T:
        tok = int(best[t])
        if tok == blank:
            t += 1
            continue
        s = t
        while t < T and best[t] == tok:
            t += 1
        peak = s + int(np.argmax(log_probs[s:t, tok]))
        out.append(Segment(tok, s, t, peak))
    return out


def _extend(targets: Sequence[int], blank: int) -> NDArray[np.int64]:
    ext = np.full(2 * len(targets) + 1, blank, dtype=np.int64)
    ext[1::2] = targets
    return ext


def _skip_allowed(ext: NDArray[np.int64], blank: int) -> NDArray[np.bool_]:
    allowed = np.zeros(len(ext), dtype=bool)
    if len(ext) > 2:
        allowed[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    return allowed


def forced_align(log_probs: FloatArray, targets: Sequence[int], blank: int) -> list[Segment]:
    """Viterbi CTC alignment of ``targets``.

    Returns one segment per target token. Blank frames between two tokens are split at
    their midpoint so segments tile the utterance between the first and last emission
    (leading/trailing silence is excluded). This gives usable phone boundaries for
    formant/duration analysis; for publication-grade boundaries use MFA.
    """
    T = log_probs.shape[0]
    L = len(targets)
    if L == 0:
        return []
    min_frames = L + sum(1 for a, b in itertools.pairwise(targets) if a == b)
    if min_frames > T:
        raise ValueError(f"utterance too short: {T} frames for {L} tokens")

    ext = _extend(targets, blank)
    S = len(ext)
    skip = _skip_allowed(ext, blank)
    emit = log_probs[:, ext]  # (T, S)

    delta = np.full(S, NEG_INF)
    delta[0] = emit[0, 0]
    delta[1] = emit[0, 1]
    bp = np.zeros((T, S), dtype=np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    for t in range(1, T):
        stay = delta
        prev1 = np.concatenate(([NEG_INF], delta[:-1]))
        prev2 = np.where(skip, np.concatenate(([NEG_INF, NEG_INF], delta[:-2])), NEG_INF)
        stacked = np.stack([stay, prev1, prev2])
        choice = stacked.argmax(axis=0)
        delta = stacked[choice, np.arange(S)] + emit[t]
        bp[t] = choice

    s = S - 1 if delta[S - 1] >= delta[S - 2] else S - 2
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])

    # frames emitted by token k are those whose state == 2k+1
    first = [-1] * L
    last = [-1] * L
    for t, st in enumerate(path.tolist()):
        if st % 2 == 1:
            tok = st // 2
            if first[tok] < 0:
                first[tok] = t
            last[tok] = t
    if min(first) < 0 or max(delta[S - 1], delta[S - 2]) < NEG_INF / 2:
        raise ValueError("forced alignment failed: target not reachable under the posteriors")
    segs: list[Segment] = []
    for k in range(L):
        start = first[k]
        end = last[k] + 1
        if k > 0:
            gap_start = last[k - 1] + 1
            start = (gap_start + first[k]) // 2 if gap_start < first[k] else first[k]
        if k < L - 1:
            gap_end = first[k + 1]
            end = (last[k] + 1 + gap_end) // 2 if last[k] + 1 < gap_end else last[k] + 1
            end = max(end, first[k] + 1)
        frames = np.arange(first[k], last[k] + 1)
        peak = int(frames[np.argmax(log_probs[frames, targets[k]])])
        segs.append(Segment(int(targets[k]), int(start), int(end), peak))
    return segs


def _logsumexp3(a: FloatArray, b: FloatArray, c: FloatArray) -> FloatArray:
    m = np.maximum(np.maximum(a, b), c)
    m_safe = np.where(m <= NEG_INF / 2, 0.0, m)
    with np.errstate(divide="ignore"):
        out = m_safe + np.log(np.exp(a - m_safe) + np.exp(b - m_safe) + np.exp(c - m_safe))
    return np.where(m <= NEG_INF / 2, NEG_INF, out)


def sequence_log_likelihood(
    log_probs: FloatArray, sequences: Sequence[Sequence[int]], blank: int
) -> FloatArray:
    """Batched CTC forward: ``log P(seq | X)`` for each sequence (vectorised over batch).

    Sequences may have different lengths; shorter ones are padded and read out at their
    own final states (padding states are only reachable *after* the final states, so
    they never contaminate the result).
    """
    T = log_probs.shape[0]
    B = len(sequences)
    if B == 0:
        return np.zeros(0)
    Lmax = max(len(s) for s in sequences)
    S = 2 * Lmax + 1
    ext = np.full((B, S), blank, dtype=np.int64)
    lens = np.array([len(s) for s in sequences])
    for b, seq in enumerate(sequences):
        ext[b, 1 : 2 * len(seq) : 2] = seq
    skip = np.zeros((B, S), dtype=bool)
    skip[:, 2:] = (ext[:, 2:] != blank) & (ext[:, 2:] != ext[:, :-2])
    pad = np.arange(S)[None, :] > (2 * lens[:, None])
    skip &= ~pad

    alpha = np.full((B, S), NEG_INF)
    alpha[:, 0] = log_probs[0, blank]
    has_tok = lens > 0
    alpha[has_tok, 1] = log_probs[0, ext[has_tok, 1]]
    neg_col = np.full((B, 1), NEG_INF)
    for t in range(1, T):
        prev1 = np.concatenate([neg_col, alpha[:, :-1]], axis=1)
        prev2 = np.where(skip, np.concatenate([neg_col, neg_col, alpha[:, :-2]], axis=1), NEG_INF)
        alpha = _logsumexp3(alpha, prev1, prev2) + log_probs[t][ext]

    last = 2 * lens
    a_last = alpha[np.arange(B), last]
    a_prev = np.where(lens > 0, alpha[np.arange(B), np.maximum(last - 1, 0)], NEG_INF)
    m = np.maximum(a_last, a_prev)
    return np.asarray(m + np.log(np.exp(a_last - m) + np.exp(a_prev - m)), dtype=np.float64)
