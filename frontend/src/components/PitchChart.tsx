import * as d3 from "d3";
import { useEffect, useRef } from "react";
import type { Track } from "../api";
import { toSemitones } from "../score";

interface Props {
  user: Track;
  reference?: Track;
}

interface Pt {
  t: number;
  st: number | null;
}

const W = 520;
const H = 260;
const M = { top: 12, right: 72, bottom: 30, left: 40 };

/** F0 contour in semitones re each speaker's median (so male/female/TTS voices compare). */
export function PitchChart({ user, reference }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    root.innerHTML = "";
    const toPts = (tr: Track): Pt[] => {
      const st = toSemitones(tr.f0);
      return tr.times.map((t, i) => ({ t, st: st[i] }));
    };
    const series = [{ key: "You", cls: "var(--series-1)", pts: toPts(user) }];
    if (reference) series.push({ key: "Reference", cls: "var(--series-2)", pts: toPts(reference) });

    const allT = series.flatMap((s) => s.pts.map((p) => p.t));
    const allSt = series.flatMap((s) => s.pts.map((p) => p.st).filter((v): v is number => v != null));
    const x = d3.scaleLinear().domain([0, d3.max(allT) ?? 1]).range([M.left, W - M.right]);
    const ext = d3.extent(allSt) as [number, number];
    const y = d3
      .scaleLinear()
      .domain([Math.min(-6, ext[0] ?? -6), Math.max(6, ext[1] ?? 6)])
      .nice()
      .range([H - M.bottom, M.top]);

    const svg = d3.select(root).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img")
      .attr("aria-label", "Pitch contour in semitones over time");
    svg.append("g").attr("transform", `translate(0,${H - M.bottom})`)
      .call(d3.axisBottom(x).ticks(6).tickFormat((d) => `${d}s`).tickSizeOuter(0))
      .call((g) => g.selectAll("text").attr("fill", "var(--muted)"))
      .call((g) => g.selectAll("line,path").attr("stroke", "var(--axis)"));
    svg.append("g").attr("transform", `translate(${M.left},0)`)
      .call(d3.axisLeft(y).ticks(5).tickSize(-(W - M.left - M.right)))
      .call((g) => g.select(".domain").remove())
      .call((g) => g.selectAll("line").attr("stroke", "var(--grid)"))
      .call((g) => g.selectAll("text").attr("fill", "var(--muted)"));
    svg.append("text").attr("x", M.left).attr("y", M.top - 2).attr("font-size", 11).attr("fill", "var(--muted)")
      .text("semitones");

    const line = d3.line<Pt>().defined((p) => p.st != null).x((p) => x(p.t)).y((p) => y(p.st as number))
      .curve(d3.curveMonotoneX);
    for (const s of series) {
      svg.append("path").datum(s.pts).attr("fill", "none").attr("stroke", s.cls).attr("stroke-width", 2)
        .attr("stroke-linejoin", "round").attr("d", line);
      const last = [...s.pts].reverse().find((p) => p.st != null);
      if (last) {
        svg.append("text").attr("x", x(last.t) + 6).attr("y", y(last.st as number)).attr("dy", "0.35em")
          .attr("font-size", 12).attr("fill", "var(--ink-2)").text(s.key);
      }
    }

    // crosshair + tooltip
    const tip = d3.select(root).append("div").attr("class", "tooltip").style("opacity", 0);
    const cross = svg.append("line").attr("stroke", "var(--axis)").attr("y1", M.top).attr("y2", H - M.bottom)
      .style("opacity", 0);
    const dots = series.map((s) =>
      svg.append("circle").attr("r", 4).attr("fill", s.cls).attr("stroke", "var(--surface)").attr("stroke-width", 2)
        .style("opacity", 0));
    svg.append("rect").attr("x", M.left).attr("y", M.top).attr("width", W - M.left - M.right)
      .attr("height", H - M.top - M.bottom).attr("fill", "transparent")
      .on("pointermove", (ev: PointerEvent) => {
        const [mx] = d3.pointer(ev);
        const t = x.invert(mx);
        cross.attr("x1", mx).attr("x2", mx).style("opacity", 1);
        const rows: string[] = [`${t.toFixed(2)} s`];
        series.forEach((s, k) => {
          const i = d3.bisector((p: Pt) => p.t).center(s.pts, t);
          const p = s.pts[i];
          if (p?.st != null) {
            dots[k].attr("cx", x(p.t)).attr("cy", y(p.st)).style("opacity", 1);
            rows.push(`${s.key}: ${p.st.toFixed(1)} st`);
          } else {
            dots[k].style("opacity", 0);
            rows.push(`${s.key}: unvoiced`);
          }
        });
        const bw = root.getBoundingClientRect().width;
        tip.style("opacity", 1).html(rows.join("<br>"))
          .style("left", `${Math.min((mx / W) * bw + 10, bw - 120)}px`).style("top", "8px");
      })
      .on("pointerleave", () => {
        cross.style("opacity", 0);
        dots.forEach((d) => d.style("opacity", 0));
        tip.style("opacity", 0);
      });
  }, [user, reference]);

  return (
    <div>
      <div className="chart" ref={ref} />
      {reference && (
        <div className="legend">
          <span><span className="swatch" style={{ background: "var(--series-1)" }} />You</span>
          <span><span className="swatch" style={{ background: "var(--series-2)" }} />Reference</span>
        </div>
      )}
    </div>
  );
}
