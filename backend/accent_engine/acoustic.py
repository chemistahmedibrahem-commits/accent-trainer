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
from dataclasses import dataclass, field
from typing import Any, Protocol

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
    # per-frame auxiliary channels, e.g. "err": posterior mass of explicit mispronunciation
    # tokens for models trained with them (slplab L2 model: "aa_err", "b*", ...)
    aux: dict[str, NDArray[np.float64]] = field(default_factory=dict)

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


# Two-phone tokens in the espeak inventory (GA r-coloured vowels, syllabic l, triphthongs).
# They are split exactly by frame doubling, see :func:`pool_log_probs`.
COMPOSITE_IPA: dict[str, tuple[str, str]] = {
    "ɑːɹ": ("AA", "R"),
    "ɔːɹ": ("AO", "R"),
    "oːɹ": ("AO", "R"),
    "ɛɹ": ("EH", "R"),
    "ɪɹ": ("IH", "R"),
    "ʊɹ": ("UH", "R"),
    "aɪɚ": ("AY", "ER"),
    "aɪə": ("AY", "AH"),
    "əl": ("AH", "L"),
    "iə": ("IY", "AH"),
    "ʊə": ("UH", "AH"),
    "eə": ("EH", "AH"),
}
_ASPIRATED = {
    "pʰ": "P",
    "tʰ": "T",
    "kʰ": "K",
    "ph": "P",
    "th": "T",
    "kh": "K",
    "tʃʰ": "CH",
    "ɛɪ": "EY",
    "oɪ": "OY",
    "ɐɐ": "AH",
    "e̞e̞": "EH",
    "o̞o̞": "OW",
    "ää": "AA",
}


def ipa_token_to_arpabet(token: str) -> str | tuple[str, str]:
    """Single ARPAbet phone, a (first, second) pair for composite tokens, or ``OTHER``."""
    raw = unicodedata.normalize("NFC", token).strip()
    if raw in COMPOSITE_IPA:
        return COMPOSITE_IPA[raw]
    if raw in _ASPIRATED:
        return _ASPIRATED[raw]
    t = raw.replace("ː", "").replace("ˑ", "")
    if t in IPA_TO_ARPABET:
        return IPA_TO_ARPABET[t]
    if t in _ASPIRATED:
        return _ASPIRATED[t]
    t2 = "".join(c for c in unicodedata.normalize("NFD", t) if not unicodedata.combining(c))
    return IPA_TO_ARPABET.get(t2, OTHER)


# TIMIT / ARPAbet-labelled models (lower-case tokens, possibly with "_err" / "*" error marks).
TIMIT_TO_ARPABET: dict[str, str | tuple[str, str]] = {
    "ax": "AH",
    "ax-h": "AH",
    "axr": "ER",
    "ix": "IH",
    "ux": "UW",
    "nx": "N",
    "hv": "HH",
    "eng": "NG",
    "el": ("AH", "L"),
    "em": ("AH", "M"),
    "en": ("AH", "N"),
    "dx": "DX",
    "ts": ("T", "S"),
    "o": "OW",
    "q": "Q",
}
EXTRA_IPA: dict[str, str] = {
    "ʧ": "CH",
    "ʤ": "JH",
    "t͡ʃ": "CH",
    "d͡ʒ": "JH",
    "ɹ̩": "ER",
    "ej": "EY",
    "aj": "AY",
    "oj": "OY",
    "aw": "AW",
    "ow": "OW",
    "g": "G",
}
_ARPA_SET = {p.lower() for p in PHONES}


def is_arpabet_vocab(model_vocab: list[str]) -> bool:
    toks = [t.split("_")[0].rstrip("*") for t in model_vocab]
    return sum(t in _ARPA_SET for t in toks) >= 30 and "iy" in toks


def token_to_arpabet(token: str, arpabet_vocab: bool) -> str | tuple[str, str]:
    if arpabet_vocab:
        t = token.split("_")[0].rstrip("*").lower()
        if t in TIMIT_TO_ARPABET:
            return TIMIT_TO_ARPABET[t]
        return t.upper() if t in _ARPA_SET else OTHER
    raw = unicodedata.normalize("NFC", token).strip()
    if raw in EXTRA_IPA:
        return EXTRA_IPA[raw]
    return ipa_token_to_arpabet(token)


def pooling_matrix(model_vocab: list[str], blank_token: str) -> NDArray[np.int64]:
    """``(2, V_model)``: engine VOCAB index each model token pools into, for the first and the
    second half of a doubled frame. Rows differ only for composite tokens."""
    out = np.empty((2, len(model_vocab)), dtype=np.int64)
    arpa = is_arpabet_vocab(model_vocab)
    for i, tok in enumerate(model_vocab):
        if tok == blank_token:
            out[:, i] = BLANK_ID
            continue
        if tok.startswith("<") or tok in ("|", "[PAD]", "[UNK]"):
            out[:, i] = OTHER_ID
            continue
        a = token_to_arpabet(tok, arpa)
        if isinstance(a, tuple):
            out[0, i], out[1, i] = PHONE_TO_ID[a[0]], PHONE_TO_ID[a[1]]
        else:
            out[:, i] = PHONE_TO_ID[a] if a != OTHER else OTHER_ID
    return out


# Engine classes a model may lack (TIMIT-39 folds AO→AA, has no flap or glottal stop). They
# borrow the posterior of their closest class so targets stay alignable; the engine already
# accepts these aliases as correct realisations, so this never manufactures errors.
CLASS_ALIASES: dict[str, tuple[str, ...]] = {
    "DX": ("D", "T"),
    "AO": ("AA",),
    "Q": ("T",),
    "ZH": ("SH",),
    "OY": ("AO", "OW"),
    "AW": ("AA",),
    "ER": ("R",),
    "HH": (),
    "NG": ("N",),
}
FLOOR_LOGP = -30.0
ALIAS_PENALTY = 0.7  # nats


def _pool(lp: NDArray[np.float64], mapping: NDArray[np.int64]) -> NDArray[np.float64]:
    T = lp.shape[0]
    if mapping.shape[0] < lp.shape[1]:  # output layer larger than vocab.json: extra units → <other>
        mapping = np.concatenate([mapping, np.full(lp.shape[1] - mapping.shape[0], OTHER_ID)])
    lp = lp[:, : mapping.shape[0]]
    out = np.full((T, len(VOCAB)), -1e30)
    present = set(np.unique(mapping).tolist())
    for target in present:
        cols = lp[:, mapping == target]
        m = cols.max(axis=1, keepdims=True)
        out[:, target] = (m + np.log(np.exp(cols - m).sum(axis=1, keepdims=True)))[:, 0]
    for cls in PHONES:
        k = PHONE_TO_ID[cls]
        if k in present:
            continue
        alias = next((PHONE_TO_ID[a] for a in CLASS_ALIASES.get(cls, ()) if PHONE_TO_ID[a] in present), None)
        # slightly below the real class so recognition never picks the borrowed label
        out[:, k] = out[:, alias] - ALIAS_PENALTY if alias is not None else FLOOR_LOGP
    m = out.max(axis=1, keepdims=True)
    out -= m + np.log(np.exp(out - m).sum(axis=1, keepdims=True))
    return out


def pool_log_probs(model_log_probs: NDArray[np.float64], mapping: NDArray[np.int64]) -> NDArray[np.float64]:
    """Pool model-token posteriors into engine VOCAB classes.

    If ``mapping`` is 1-D (or both rows are equal) this is plain log-sum-exp pooling. With
    composite tokens every frame is doubled: copy 1 gives the composite's mass to its first
    phone, copy 2 to its second. Every CTC path probability is then exactly squared
    (composite paths included), so likelihood *ratios* are preserved in log space ×2.
    """
    if mapping.ndim == 1:
        return _pool(model_log_probs, mapping)
    if np.array_equal(mapping[0], mapping[1]):
        return _pool(model_log_probs, mapping[0])
    a = _pool(model_log_probs, mapping[0])
    b = _pool(model_log_probs, mapping[1])
    out = np.empty((2 * a.shape[0], a.shape[1]))
    out[0::2] = a
    out[1::2] = b
    return out


class Wav2Vec2PhonemeModel:
    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
        device: str | None = None,
    ) -> None:
        import json

        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoFeatureExtractor, AutoModelForCTC

        self.name = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Only the feature extractor + raw vocab are needed: the phoneme tokenizer class would
        # pull in `phonemizer`/espeak, which we never use (we never encode text with it).
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        with open(hf_hub_download(model_id, "vocab.json")) as f:
            raw_vocab: dict[str, Any] = json.load(f)
        first = next(iter(raw_vocab.values()), None)
        vocab: dict[str, int] = first if isinstance(first, dict) else raw_vocab  # some are {lang: vocab}
        id_to_tok = [""] * (max(vocab.values()) + 1)
        for tok, i in vocab.items():
            id_to_tok[i] = tok
        blank_id = int(self.model.config.pad_token_id)
        self._mapping = pooling_matrix(id_to_tok, id_to_tok[blank_id])
        self._err_idx = np.array(
            [i for i, t in enumerate(id_to_tok) if t.endswith("_err") or t.endswith("*")], dtype=np.int64
        )
        self._torch = torch

    def posteriors(self, audio: NDArray[np.float32]) -> Posteriors:
        torch = self._torch
        inputs = self.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        x = inputs.input_values.to(self.device)
        if self.device == "cuda":
            x = x.half()
        with torch.inference_mode():
            logits = self.model(x).logits[0]
            lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64)
        pooled = pool_log_probs(lp, self._mapping)
        aux: dict[str, NDArray[np.float64]] = {}
        if len(self._err_idx):
            err = np.exp(lp[:, self._err_idx]).sum(axis=1)
            aux["err"] = (
                np.repeat(err, pooled.shape[0] // lp.shape[0]) if pooled.shape[0] != lp.shape[0] else err
            )
        return Posteriors(pooled, frame_shift=0.02 * lp.shape[0] / pooled.shape[0], aux=aux)


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
