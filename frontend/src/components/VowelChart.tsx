import * as d3 from "d3";
import { useEffect, useRef } from "react";
import type { PhoneResult, VowelReference } from "../api";

interface Props {
  reference: VowelReference;
  phones: { word: string; phone: PhoneResult }[];
}

const W = 420;
const H = 340;
const M = { top: 16, right: 16, bottom: 30, left: 36 };

/**
 * Speaker-normalised vowel space (Lobanov z-scores). Classic orientation: front vowels left
 * (high F2), high vowels top (low F1). Grey rings = General American targets (±1 SD).
 */
export function VowelChart({ reference, phones }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    root.innerHTML = "";
    const pts = phones.filter((p) => p.phone.vowel_score);
    const refs = Object.entries(reference.vowels);
    const xs = [...refs.map(([, v]) => v.z_f2), ...pts.map((p) => p.phone.vowel_score!.point[1])];
    const ys = [...refs.map(([, v]) => v.z_f1), ...pts.map((p) => p.phone.vowel_score!.point[0])];
    const pad = 0.6;
    const x = d3.scaleLinear().domain([(d3.max(xs) ?? 2) + pad, (d3.min(xs) ?? -2) - pad]).range([M.left, W - M.right]);
    const y = d3.scaleLinear().domain([(d3.min(ys) ?? -2) - pad, (d3.max(ys) ?? 2) + pad]).range([M.top, H - M.bottom]);

    const svg = d3.select(root).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("role", "img")
      .attr("aria-label", "Your vowels plotted against General American vowel targets");
    svg.append("g").attr("transform", `translate(0,${H - M.bottom})`).call(d3.axisBottom(x).ticks(5).tickSize(-(H - M.top - M.bottom)))
      .call((g) => g.select(".domain").remove()).call((g) => g.selectAll("line").attr("stroke", "var(--grid)"))
      .call((g) => g.selectAll("text").attr("fill", "var(--muted)"));
    svg.append("g").attr("transform", `translate(${M.left},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(W - M.left - M.right)))
      .call((g) => g.select(".domain").remove()).call((g) => g.selectAll("line").attr("stroke", "var(--grid)"))
      .call((g) => g.selectAll("text").attr("fill", "var(--muted)"));
    svg.append("text").attr("x", W - M.right).attr("y", H - 4).attr("text-anchor", "end").attr("font-size", 11)
      .attr("fill", "var(--muted)").text("← front   F2 (z)   back →");
    svg.append("text").attr("x", 4).attr("y", M.top + 4).attr("font-size", 11).attr("fill", "var(--muted)").text("F1 (z)");

    const rx = Math.abs(x(reference.sd) - x(0));
    const ry = Math.abs(y(reference.sd) - y(0));
    for (const [, v] of refs) {
      svg.append("ellipse").attr("cx", x(v.z_f2)).attr("cy", y(v.z_f1)).attr("rx", rx).attr("ry", ry)
        .attr("fill", "none").attr("stroke", "var(--axis)").attr("stroke-width", 1.5);
      svg.append("text").attr("x", x(v.z_f2)).attr("y", y(v.z_f1)).attr("dy", "0.35em").attr("text-anchor", "middle")
        .attr("font-size", 13).attr("fill", "var(--ink-2)").text(v.ipa);
    }

    const tip = d3.select(root).append("div").attr("class", "tooltip").style("opacity", 0);
    for (const { word, phone } of pts) {
      const vs = phone.vowel_score!;
      const px = x(vs.point[1]);
      const py = y(vs.point[0]);
      svg.append("line").attr("x1", px).attr("y1", py).attr("x2", x(vs.target[1])).attr("y2", y(vs.target[0]))
        .attr("stroke", "var(--series-1)").attr("stroke-width", 1).attr("stroke-dasharray", "3 3");
      svg.append("circle").attr("cx", px).attr("cy", py).attr("r", 6).attr("fill", "var(--series-1)")
        .attr("stroke", "var(--surface)").attr("stroke-width", 2);
      svg.append("text").attr("x", px + 9).attr("y", py).attr("dy", "0.35em").attr("font-size", 12)
        .attr("fill", "var(--ink)").text(`${phone.ipa} (${word})`);
      // hit target larger than the mark
      svg.append("circle").attr("cx", px).attr("cy", py).attr("r", 14).attr("fill", "transparent")
        .on("pointerenter", () => {
          const bw = root.getBoundingClientRect().width;
          tip.style("opacity", 1)
            .html(`<b>/${phone.ipa}/ in “${word}”</b><br>F1 ${Math.round(phone.formants!.f1)} Hz · F2 ${Math.round(phone.formants!.f2)} Hz<br>` +
              `closest to /${reference.vowels[vs.nearest]?.ipa ?? vs.nearest}/ · score ${Math.round(vs.score)}`)
            .style("left", `${Math.min((px / W) * bw + 12, bw - 180)}px`).style("top", `${(py / H) * root.getBoundingClientRect().height}px`);
        })
        .on("pointerleave", () => tip.style("opacity", 0));
    }
  }, [reference, phones]);

  return (
    <div>
      <div className="chart" ref={ref} />
      <div className="legend">
        <span><span className="swatch" style={{ background: "var(--series-1)" }} />your vowel → its target</span>
        <span><span className="swatch" style={{ border: "1.5px solid var(--axis)", borderRadius: 5 }} />General American target (±1 SD)</span>
      </div>
    </div>
  );
}
