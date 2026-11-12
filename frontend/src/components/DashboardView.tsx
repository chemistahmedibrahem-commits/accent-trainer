import * as d3 from "d3";
import { useEffect, useRef, useState } from "react";
import { api2, type Dashboard } from "../api";
import { fmt } from "../score";

// sequential blue ramp (light → dark = low → high correct rate)
const SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"];

export function DashboardView({ user }: { user: string }) {
  const [d, setD] = useState<Dashboard>();
  const [err, setErr] = useState<string>();
  useEffect(() => {
    api2.dashboard(user).then(setD).catch((e: Error) => setErr(e.message));
  }, [user]);
  if (err) return <section className="card">No data yet for “{user}”. Practise a few sentences first.</section>;
  if (!d) return <section className="card">Loading…</section>;
  const base = `/api/users/${encodeURIComponent(user)}/report`;
  return (
    <>
      <section className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2 style={{ margin: 0 }}>Top weaknesses</h2>
          <div className="row">
            <a className="btn" href={`${base}.csv`}>Export CSV</a>
            <a className="btn" href={`${base}.pdf`}>Export PDF</a>
          </div>
        </div>
        {d.weaknesses.length === 0 ? (
          <p className="status">Not enough evidence yet (each skill needs 3+ opportunities).</p>
        ) : (
          <ol>
            {d.weaknesses.map((w) => (
              <li key={w.skill}>
                {w.label} — mastery {Math.round(w.mastery * 100)}% ({w.n_obs} tries)
              </li>
            ))}
          </ol>
        )}
      </section>
      <section className="card">
        <h2>Skill mastery</h2>
        <table>
          <thead>
            <tr><th>Skill</th><th>Mastery</th><th className="num">Tries</th><th className="num">Correct</th><th>Next review</th></tr>
          </thead>
          <tbody>
            {d.skills.map((s) => (
              <tr key={s.skill}>
                <td>{s.label}</td>
                <td style={{ minWidth: 120 }}>
                  <div className="level" title={`${Math.round(s.mastery * 100)}%`}>
                    <div style={{ width: `${s.mastery * 100}%` }} />
                  </div>
                  <span className="status" style={{ fontSize: 12 }}>{Math.round(s.mastery * 100)}%{s.mastered ? " ✓ mastered" : ""}</span>
                </td>
                <td className="num">{s.n_obs}</td>
                <td className="num">{s.correct_rate == null ? "–" : `${Math.round(s.correct_rate * 100)}%`}</td>
                <td>{s.due ? new Date(s.due).toLocaleDateString() : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <div className="grid2">
        <section className="card">
          <h2>Correct rate by week</h2>
          <Heatmap cells={d.heatmap} />
        </section>
        <section className="card">
          <h2>Overall score by day</h2>
          <Trend points={d.trend} />
        </section>
      </div>
    </>
  );
}

function Heatmap({ cells }: { cells: Dashboard["heatmap"] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    root.innerHTML = "";
    if (!cells.length) return;
    const skills = [...new Set(cells.map((c) => c.label))];
    const weeks = [...new Set(cells.map((c) => c.week))].sort();
    const M = { top: 8, right: 8, bottom: 28, left: 170 };
    const cw = 44, ch = 24;
    const W = M.left + weeks.length * cw + M.right;
    const H = M.top + skills.length * ch + M.bottom;
    const x = d3.scaleBand().domain(weeks).range([M.left, W - M.right]).padding(0.08);
    const y = d3.scaleBand().domain(skills).range([M.top, H - M.bottom]).padding(0.08);
    const color = d3.scaleQuantize<string>().domain([0, 1]).range(SEQ);
    const svg = d3.select(root).append("svg").attr("viewBox", `0 0 ${W} ${H}`).style("max-width", `${W}px`)
      .attr("role", "img").attr("aria-label", "Weekly correct rate per skill");
    const tip = d3.select(root).append("div").attr("class", "tooltip").style("opacity", 0);
    svg.selectAll("rect").data(cells).join("rect")
      .attr("x", (c) => x(c.week) ?? 0).attr("y", (c) => y(c.label) ?? 0)
      .attr("width", x.bandwidth()).attr("height", y.bandwidth()).attr("rx", 3)
      .attr("fill", (c) => color(c.correct_rate))
      .on("pointerenter", (ev: PointerEvent, c) => {
        const [mx, my] = d3.pointer(ev, root);
        tip.style("opacity", 1).html(`<b>${c.label}</b><br>${c.week}: ${Math.round(c.correct_rate * 100)}% correct (${c.n} tries)`)
          .style("left", `${Math.min(mx + 10, root.clientWidth - 200)}px`).style("top", `${my + 10}px`);
      })
      .on("pointerleave", () => tip.style("opacity", 0));
    svg.append("g").attr("transform", `translate(${M.left - 6},0)`).selectAll("text").data(skills).join("text")
      .attr("x", 0).attr("y", (s) => (y(s) ?? 0) + y.bandwidth() / 2).attr("dy", "0.35em").attr("text-anchor", "end")
      .attr("font-size", 12).attr("fill", "var(--ink-2)").text((s) => s);
    svg.append("g").selectAll("text").data(weeks).join("text")
      .attr("x", (w) => (x(w) ?? 0) + x.bandwidth() / 2).attr("y", H - 10).attr("text-anchor", "middle")
      .attr("font-size", 11).attr("fill", "var(--muted)").text((w) => w.slice(5));
  }, [cells]);
  return (
    <div>
      <div className="chart" ref={ref} />
      <div className="legend">
        <span>0%</span>
        {SEQ.map((c) => <span key={c} className="swatch" style={{ background: c, margin: 0 }} />)}
        <span>100% correct</span>
      </div>
    </div>
  );
}

function Trend({ points }: { points: Dashboard["trend"] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    root.innerHTML = "";
    const pts = points.filter((p) => p.total != null).map((p) => ({ ...p, date: new Date(p.day) }));
    if (!pts.length) return;
    const W = 480, H = 220, M = { top: 12, right: 16, bottom: 28, left: 34 };
    const ext = d3.extent(pts, (p) => p.date) as [Date, Date];
    const x = d3.scaleTime().domain(ext[0].getTime() === ext[1].getTime() ? [d3.timeDay.offset(ext[0], -1), d3.timeDay.offset(ext[1], 1)] : ext).range([M.left, W - M.right]);
    const y = d3.scaleLinear().domain([0, 100]).range([H - M.bottom, M.top]);
    const svg = d3.select(root).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img").attr("aria-label", "Daily overall score");
    svg.append("g").attr("transform", `translate(${M.left},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - M.left - M.right)))
      .call((g) => g.select(".domain").remove()).call((g) => g.selectAll("line").attr("stroke", "var(--grid)"))
      .call((g) => g.selectAll("text").attr("fill", "var(--muted)"));
    svg.append("g").attr("transform", `translate(0,${H - M.bottom})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0))
      .call((g) => g.selectAll("text").attr("fill", "var(--muted)")).call((g) => g.selectAll("line,path").attr("stroke", "var(--axis)"));
    svg.append("path").datum(pts).attr("fill", "none").attr("stroke", "var(--series-1)").attr("stroke-width", 2)
      .attr("d", d3.line<(typeof pts)[number]>().x((p) => x(p.date)).y((p) => y(p.total as number)));
    const tip = d3.select(root).append("div").attr("class", "tooltip").style("opacity", 0);
    svg.selectAll("circle").data(pts).join("circle").attr("cx", (p) => x(p.date)).attr("cy", (p) => y(p.total as number))
      .attr("r", 5).attr("fill", "var(--series-1)").attr("stroke", "var(--surface)").attr("stroke-width", 2)
      .on("pointerenter", (ev: PointerEvent, p) => {
        const [mx, my] = d3.pointer(ev, root);
        tip.style("opacity", 1).html(`${p.day}<br>overall ${fmt(p.total)} · ${p.n} attempts`)
          .style("left", `${Math.min(mx + 10, root.clientWidth - 160)}px`).style("top", `${my}px`);
      })
      .on("pointerleave", () => tip.style("opacity", 0));
  }, [points]);
  return <div className="chart" ref={ref} />;
}
