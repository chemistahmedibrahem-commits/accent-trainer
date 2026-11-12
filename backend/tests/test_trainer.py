import itertools
from datetime import UTC, datetime, timedelta

import numpy as np

from accent_engine import EngineConfig, PronunciationEngine
from accent_engine.acoustic import SyntheticAcousticModel
from accent_engine.g2p import g2p
from trainer import bkt, exercises, fsrs
from trainer.planner import SkillState, plan
from trainer.skills import observations, site_skills

NOW = datetime(2026, 1, 1, tzinfo=UTC)
FAST = EngineConfig(formants=False, prosody=False)


def test_bkt_moves_in_the_right_direction_and_converges() -> None:
    p = bkt.DEFAULT.p_init
    assert bkt.update(p, True) > p
    assert bkt.update(p, False) < bkt.update(p, True)
    for _ in range(15):
        p = bkt.update(p, True)
    assert p > bkt.MASTERED
    for _ in range(5):
        p = bkt.update(p, False)
    assert p < 0.5


def test_fsrs_intervals_grow_with_success_and_reset_on_lapse() -> None:
    c = fsrs.review(fsrs.Card(), fsrs.Rating.GOOD, NOW)
    assert c.reps == 1 and c.due is not None
    t = NOW
    ivls = []
    for _ in range(4):
        assert c.due is not None
        t = c.due
        prev_due = c.due
        c = fsrs.review(c, fsrs.Rating.GOOD, t)
        assert c.due is not None
        ivls.append((c.due - prev_due).total_seconds())
    assert all(b > a for a, b in itertools.pairwise(ivls))
    s_before = c.stability
    c = fsrs.review(c, fsrs.Rating.AGAIN, c.due or t)
    assert c.lapses == 1 and c.stability < s_before
    assert c.due is not None and c.last_review is not None
    assert c.due - c.last_review <= timedelta(minutes=10)


def test_fsrs_retrievability_is_90_percent_at_stability() -> None:
    assert abs(fsrs.retrievability(7.0, 7.0) - 0.9) < 1e-9
    assert fsrs.rating_from_score(30) == fsrs.Rating.AGAIN and fsrs.rating_from_score(95) == fsrs.Rating.EASY


def test_site_skills() -> None:
    t = g2p("think").phones
    assert site_skills(t, 0) == ["TH_SUB"]
    s = g2p("street").phones
    assert "EPENTHESIS" in site_skills(s, 0)
    w = g2p("water").phones
    assert "FLAP_T" in site_skills(w, 2) and "RHOTIC" in site_skills(w, 3)
    b = g2p("bag").phones
    assert "FINAL_DEVOICING" in site_skills(b, 2)


def test_observations_from_assessment() -> None:
    text = "think about the street"
    tp = g2p(text).phones
    spoken = [p.phone for p in tp]
    spoken[0] = "S"  # TH→S
    r = (
        PronunciationEngine(SyntheticAcousticModel(spoken), FAST)
        .assess(np.zeros(32000, np.float32), text)
        .to_dict()
    )
    obs = observations(r, tp)
    th = [o for o in obs if o.skill == "TH_SUB"]
    assert th and not th[0].correct
    assert all(o.correct for o in obs if o.skill == "EPENTHESIS")
    assert any(o.skill == "DH_SUB" and o.correct for o in obs)


def test_exercise_generation() -> None:
    words = exercises.words_for("TH_SUB", "initial", n=5, seed=1)
    assert len(words) == 5 and all(g2p(w).phones[0].phone == "TH" for w in words)
    pairs = exercises.minimal_pairs("IH_IY")
    assert len(pairs) >= 10
    assert any(
        {"ship", "sheep"} == set(p) or {"live", "leave"} == set(p) or {"bit", "beat"} == set(p) for p in pairs
    )
    dev = exercises.minimal_pairs("FINAL_DEVOICING")
    assert dev and all(g2p(a).phones[-1].phone != g2p(b).phones[-1].phone for a, b in dev)
    shifts = exercises.stress_shift_pairs()
    assert any(w in ("record", "present", "object", "project", "increase") for w, _, _ in shifts)


def test_planner_prioritises_weak_due_skills_and_starts_with_ear_training() -> None:
    weak = SkillState("TH_SUB")
    for _ in range(4):
        weak.observe(False)
    weak.review(30, NOW - timedelta(days=1))
    strong = SkillState("P_B")
    for _ in range(20):
        strong.observe(True)
    unseen = SkillState("RHOTIC")
    drills = plan([strong, weak, unseen], NOW, seed=0)
    assert drills and drills[0].skill == "TH_SUB" and drills[0].kind == "perception"
    assert {d.skill for d in drills} == {"TH_SUB"}
