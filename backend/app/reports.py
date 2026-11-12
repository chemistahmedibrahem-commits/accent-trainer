"""Progress reports: aggregation for the dashboard + CSV / PDF export."""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from trainer import bkt
from trainer.skills import SKILL_LABEL

from .db import Attempt, SkillObservation, SkillStat, User


def _week(d: datetime) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


async def dashboard(db: AsyncSession, user: User, weeks: int = 8) -> dict[str, Any]:
    since = datetime.now(UTC) - timedelta(weeks=weeks)
    obs = (
        (
            await db.execute(
                select(SkillObservation).where(
                    SkillObservation.user_id == user.id, SkillObservation.created_at >= since
                )
            )
        )
        .scalars()
        .all()
    )
    cell: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for o in obs:
        cell[(o.skill, _week(o.created_at))].append(o.correct)
    heatmap = [
        {
            "skill": sk,
            "label": SKILL_LABEL.get(sk, sk),
            "week": wk,
            "n": len(v),
            "correct_rate": sum(v) / len(v),
        }
        for (sk, wk), v in sorted(cell.items())
    ]
    attempts = (
        (await db.execute(select(Attempt).where(Attempt.user_id == user.id).order_by(Attempt.created_at)))
        .scalars()
        .all()
    )
    by_day: dict[str, list[Attempt]] = defaultdict(list)
    for a in attempts:
        by_day[a.created_at.date().isoformat()].append(a)

    def mean(xs: list[float | None]) -> float | None:
        v = [x for x in xs if x is not None]
        return sum(v) / len(v) if v else None

    trend = [
        {
            "day": d,
            "n": len(v),
            "total": mean([a.total for a in v]),
            "accuracy": mean([a.accuracy for a in v]),
            "fluency": mean([a.fluency for a in v]),
            "prosody": mean([a.prosody for a in v]),
        }
        for d, v in sorted(by_day.items())
    ]
    stats = (await db.execute(select(SkillStat).where(SkillStat.user_id == user.id))).scalars().all()
    skills: list[dict[str, Any]] = sorted(
        (
            {
                "skill": s.skill,
                "label": SKILL_LABEL.get(s.skill, s.skill),
                "mastery": s.p_known,
                "mastered": s.p_known >= bkt.MASTERED,
                "n_obs": s.n_obs,
                "correct_rate": s.n_correct / s.n_obs if s.n_obs else None,
                "due": s.due.isoformat() if s.due else None,
                "reps": s.reps,
                "lapses": s.lapses,
            }
            for s in stats
        ),
        key=lambda r: float(r["mastery"]),
    )
    weak = [s for s in skills if s["n_obs"] >= 3 and not s["mastered"]][:5]
    return {
        "user": user.name,
        "skills": skills,
        "weaknesses": weak,
        "heatmap": heatmap,
        "trend": trend,
        "n_attempts": len(attempts),
    }


def to_csv(dash: dict[str, Any], attempts: list[Attempt]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["section", "key", "label", "value1", "value2", "value3"])
    for s in dash["skills"]:
        w.writerow(
            [
                "skill",
                s["skill"],
                s["label"],
                f"{s['mastery']:.3f}",
                s["n_obs"],
                "" if s["correct_rate"] is None else f"{s['correct_rate']:.3f}",
            ]
        )
    for t in dash["trend"]:
        w.writerow(["daily", t["day"], t["n"], _f(t["total"]), _f(t["accuracy"]), _f(t["prosody"])])
    for a in attempts:
        w.writerow(["attempt", a.id, a.created_at.isoformat(), a.text, _f(a.total), _f(a.accuracy)])
    return buf.getvalue()


def _f(x: float | None) -> str:
    return "" if x is None else f"{x:.1f}"


def to_pdf(dash: dict[str, Any], attempts: list[Attempt]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=f"Accent report – {dash['user']}")
    ss = getSampleStyleSheet()
    els: list[Any] = [
        Paragraph(f"American Accent Trainer – progress report for {dash['user']}", ss["Title"]),
        Paragraph(
            f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC · {dash['n_attempts']} attempts", ss["Normal"]
        ),
        Spacer(1, 12),
        Paragraph("Top weaknesses", ss["Heading2"]),
    ]
    if dash["weaknesses"]:
        for s in dash["weaknesses"]:
            els.append(
                Paragraph(
                    f"• {s['label']} — mastery {s['mastery']:.0%} ({s['n_obs']} opportunities)", ss["Normal"]
                )
            )
    else:
        els.append(Paragraph("Not enough practice data yet.", ss["Normal"]))
    els += [Spacer(1, 12), Paragraph("Skills", ss["Heading2"])]
    rows = [["Skill", "Mastery", "Opportunities", "Correct", "Next review"]]
    for s in dash["skills"]:
        rows.append(
            [
                s["label"],
                f"{s['mastery']:.0%}",
                str(s["n_obs"]),
                "–" if s["correct_rate"] is None else f"{s['correct_rate']:.0%}",
                (s["due"] or "–")[:10],
            ]
        )
    t = Table(rows, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    els += [t, Spacer(1, 12), Paragraph("Daily scores", ss["Heading2"])]
    rows = [["Day", "Attempts", "Overall", "Accuracy", "Fluency", "Prosody"]]
    for d in dash["trend"][-30:]:
        rows.append(
            [d["day"], str(d["n"]), _f(d["total"]), _f(d["accuracy"]), _f(d["fluency"]), _f(d["prosody"])]
        )
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    els.append(t)
    doc.build(els)
    return buf.getvalue()
