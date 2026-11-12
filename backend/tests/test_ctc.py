import itertools

import numpy as np
import pytest

from accent_engine.ctc import forced_align, greedy_decode, sequence_log_likelihood


def _random_lp(T: int, V: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(T, V))
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def _collapse(path: tuple[int, ...], blank: int) -> list[int]:
    out = []
    prev = None
    for p in path:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return out


@pytest.mark.parametrize("target", [[1, 2], [1, 1], [2], [1, 2, 1]])
def test_forward_matches_brute_force(target: list[int]) -> None:
    T, V = 5, 3
    lp = _random_lp(T, V)
    brute = 0.0
    for path in itertools.product(range(V), repeat=T):
        if _collapse(path, 0) == target:
            brute += float(np.exp(sum(lp[t, path[t]] for t in range(T))))
    got = sequence_log_likelihood(lp, [target], blank=0)[0]
    assert np.isclose(got, np.log(brute))


def test_batched_equals_individual() -> None:
    lp = _random_lp(30, 6, seed=3)
    seqs = [[1, 2, 3], [4], [1, 1, 5, 2], [2, 3]]
    batched = sequence_log_likelihood(lp, seqs, blank=0)
    single = [sequence_log_likelihood(lp, [s], blank=0)[0] for s in seqs]
    assert np.allclose(batched, single)


def _peaky(T: int, V: int, peaks: dict[int, int]) -> np.ndarray:
    p = np.full((T, V), 1e-3)
    p[:, 0] = 1.0
    for t, tok in peaks.items():
        p[t] = 1e-3
        p[t, tok] = 1.0
    p /= p.sum(axis=1, keepdims=True)
    return np.log(p)


def test_forced_align_finds_peaks_and_tiles() -> None:
    lp = _peaky(40, 5, {5: 1, 15: 2, 16: 2, 30: 3})
    segs = forced_align(lp, [1, 2, 3], blank=0)
    assert [s.peak for s in segs] == [5, 15, 30]
    assert segs[0].start == 5 and segs[-1].end == 31
    for a, b in itertools.pairwise(segs):
        assert a.end == b.start  # tiles between first/last emission


def test_forced_align_repeated_tokens() -> None:
    lp = _peaky(20, 4, {4: 2, 10: 2})
    segs = forced_align(lp, [2, 2], blank=0)
    assert [s.peak for s in segs] == [4, 10]


def test_forced_align_too_short() -> None:
    with pytest.raises(ValueError):
        forced_align(_random_lp(2, 4), [1, 1, 2], blank=0)


def test_greedy_decode() -> None:
    lp = _peaky(20, 4, {2: 1, 3: 1, 8: 3, 12: 3})
    assert [s.token for s in greedy_decode(lp, 0)] == [1, 3, 3]
