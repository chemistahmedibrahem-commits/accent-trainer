"""Acoustic models producing frame-level phone posteriors in the engine's ARPAbet vocab.

Every backend returns :class:`Posteriors` over :data:`VOCAB` (index 0 = CTC blank,
last = ``<other>`` pooling any model token that has no GA ARPAbet equivalent). That keeps
GOP / alignment code independent of the underlying network and its label set.

Backends:
* :class:`Wav2Vec2PhonemeModel` – HF CTC phoneme model (default
  ``facebook/wav2vec2-xlsr-53-espeak-cv-ft``); IPA token posteriors are pooled into
  ARPAbet classes with log-sum-exp. Requires ``torch`` + ``transformers`` (extra ``ml``).
* :class:`SyntheticAcousticModel` – deterministic posteriors generated from a "spoken"
  phone string. Used for unit tests and pipeline benchmarking without a GPU/model.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .phonemes import PHONES

BLANK = "<blank>"
OTHER = "<other>"
VOCAB: tuple[str, ...] = (BLANK, *PHONES, OTHER)
PHONE_TO_ID: dict[str, int] = {p: i for i, p in enumerate(VOCAB)}
BLANK_ID = 0
OTHER_ID = len(VOCAB) - 1
SAMPLE_RATE = 16000


@dataclass
class Posteriors:
    log_probs: NDArray[np.float64]  # (T, len(VOCAB)), rows are log-softmax
    frame_shift: float  # seconds per frame (wav2vec2: 0.02)

    @property
    def num_frames(self) -> int:
        return int(self.log_probs.shape[0])

    def frame_to_time(self, frame: int) -> float:
        return frame * self.frame_shift


class AcousticModel(Protocol):
    name: str

    def posteriors(self, audio: NDArray[np.float32]) -> Posteriors:
        """``audio``: mono float32 at 16 kHz in [-1, 1]."""
        ...


# ---------------------------------------------------------------------------
# IPA (espeak) → ARPAbet pooling
# ---------------------------------------------------------------------------

IPA_TO_ARPABET: dict[str, str] = {
    "ɑ": "AA",
    "a": "AA",
    "ɒ": "AA",
    "æ": "AE",
    "ʌ": "AH",
    "ə": "AH",
    "ɐ": "AH",
    "ɔ": "AO",
    "aʊ": "AW",
    "aɪ": "AY",
    "ɛ": "EH",
    "e": "EH",
    "ɝ": "ER",
    "ɚ": "ER",
    "ɜ": "ER",
    "ɜɹ": "ER",
    "əɹ": "ER",
    "eɪ": "EY",
    "ɪ": "IH",
    "ᵻ": "IH",
    "ɨ": "IH",
    "i": "IY",
    "oʊ": "OW",
    "o": "OW",
    "əʊ": "OW",
    "ɔɪ": "OY",
    "ʊ": "UH",
    "u": "UW",
    "ʉ": "UW",
    "b": "B",
    "tʃ": "CH",
    "d": "D",
    "ð": "DH",
    "f": "F",
    "ɡ": "G",
    "g": "G",
    "h": "HH",
    "dʒ": "JH",
    "k": "K",
    "l": "L",
    "ɫ": "L",
    "m": "M",
    "n": "N",
    "ŋ": "NG",
    "p": "P",
    "ɹ": "R",
    "r": "R",
    "ɻ": "R",
    "s": "S",
    "ʃ": "SH",
    "t": "T",
    "θ": "TH",
    "v": "V",
    "w": "W",
    "j": "Y",
    "z": "Z",
    "ʒ": "ZH",
    "ɾ": "DX",
    "ʔ": "Q",
    "n̩": "N",
    "l̩": "L",
    "m̩": "M",
}


def ipa_token_to_arpabet(token: str) -> str:
    t = unicodedata.normalize("NFC", token).replace("ː", "").replace("ˑ", "").strip()
    if t in IPA_TO_ARPABET:
        return IPA_TO_ARPABET[t]
    t2 = "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c))
    return IPA_TO_ARPABET.get(t2, OTHER)


def pooling_matrix(model_vocab: list[str], blank_token: str) -> NDArray[np.int64]:
    """For each model token, the engine VOCAB index it pools into."""
    out = np.empty(len(model_vocab), dtype=np.int64)
    for i, tok in enumerate(model_vocab):
        if tok == blank_token:
            out[i] = BLANK_ID
        elif tok.startswith("<") or tok in ("|", "[PAD]", "[UNK]"):
            out[i] = OTHER_ID
        else:
            out[i] = (
                PHONE_TO_ID[ipa_token_to_arpabet(tok)] if ipa_token_to_arpabet(tok) != OTHER else OTHER_ID
            )
    return out


def pool_log_probs(model_log_probs: NDArray[np.float64], mapping: NDArray[np.int64]) -> NDArray[np.float64]:
    T = model_log_probs.shape[0]
    out = np.full((T, len(VOCAB)), -1e30)
    for target in np.unique(mapping):
        cols = model_log_probs[:, mapping == target]
        m = cols.max(axis=1, keepdims=True)
        out[:, target] = (m + np.log(np.exp(cols - m).sum(axis=1, keepdims=True)))[:, 0]
    # renormalise (pooling is exact in probability space, but guard numeric drift)
    m = out.max(axis=1, keepdims=True)
    out -= m + np.log(np.exp(out - m).sum(axis=1, keepdims=True))
    return out


class Wav2Vec2PhonemeModel:
    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        self.name = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id).to(self.device).eval()
        vocab = self.processor.tokenizer.get_vocab()
        id_to_tok = [""] * len(vocab)
        for tok, i in vocab.items():
            id_to_tok[i] = tok
        blank_tok = self.processor.tokenizer.pad_token
        self._mapping = pooling_matrix(id_to_tok, blank_tok)
        self._torch = torch

    def posteriors(self, audio: NDArray[np.float32]) -> Posteriors:
        torch = self._torch
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
            lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64)
        return Posteriors(pool_log_probs(lp, self._mapping), frame_shift=0.02)


# ---------------------------------------------------------------------------
# Synthetic backend (tests / benchmarks)
# ---------------------------------------------------------------------------


class SyntheticAcousticModel:
    """Produce CTC-like peaky posteriors for a given spoken phone sequence.

    ``spoken`` items may be ``"PH"`` or ``("PH", confidence)``; ``confidence`` is the
    posterior mass on PH at its peak frame (rest spread on ``confusions`` if given via
    ``("PH", conf, "ALT")``, else uniformly). Audio duration is honoured: phones are spread
    evenly across the non-silent middle 80 % of the frames.
    """

    name = "synthetic"

    def __init__(
        self,
        spoken: list[str | tuple[str, float] | tuple[str, float, str]],
        frame_shift: float = 0.02,
        seed: int = 0,
    ) -> None:
        self.spoken = spoken
        self.frame_shift = frame_shift
        self.rng = np.random.default_rng(seed)

    def posteriors(self, audio: NDArray[np.float32]) -> Posteriors:
        T = max(int(len(audio) / SAMPLE_RATE / self.frame_shift), 4 * len(self.spoken) + 4)
        V = len(VOCAB)
        probs = np.full((T, V), 1e-4)
        probs[:, BLANK_ID] = 1.0
        n = len(self.spoken)
        lead = int(T * 0.1)
        span = T - 2 * lead
        for k, item in enumerate(self.spoken):
            if isinstance(item, str):
                ph, conf, alt = item, 0.95, None
            elif len(item) == 2:
                ph, conf, alt = item[0], float(item[1]), None  # type: ignore[misc]
            else:
                ph, conf, alt = item[0], float(item[1]), item[2]  # type: ignore[misc]
            centre = lead + int((k + 0.5) * span / n)
            for t in (centre, centre + 1):
                row = np.full(V, 1e-4)
                row[PHONE_TO_ID[ph]] = conf
                rest = 1.0 - conf
                if alt is not None:
                    row[PHONE_TO_ID[alt]] = rest * 0.9
                    row[BLANK_ID] = rest * 0.1
                else:
                    row[BLANK_ID] = rest * 0.5
                    row[1:-1] += rest * 0.5 / (V - 2)
                probs[t] = row
        probs += self.rng.uniform(0, 1e-5, size=probs.shape)
        probs /= probs.sum(axis=1, keepdims=True)
        return Posteriors(np.log(probs), self.frame_shift)
