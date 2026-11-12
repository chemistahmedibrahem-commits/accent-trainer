import { useEffect, useState } from "react";
import { api, type Progress } from "../api";
import { BAND_LABEL, band, fmt } from "../score";

export function ProgressView({ user }: { user: string }) {
  const [p, setP] = useState<Progress>();
  const [err, setErr] = useState<string>();
  useEffect(() => {
    api.progress(user).then(setP).catch((e: Error) => setErr(e.message));
  }, [user]);
  if (err) return <section className="card">No progress yet for “{user}”. Practice a few sentences first.</section>;
  if (!p) return <section className="card">Loading…</section>;
  const phones = [...p.phones].sort((a, b) => a.mean_score - b.mean_score);
  return (
    <>
      <section className="card">
        <h2>Top weaknesses</h2>
        {p.weaknesses.length === 0 ? (
          <p className="status">Need at least 3 attempts per sound before ranking weaknesses.</p>
        ) : (
          <ol>
            {p.weaknesses.map((w) => (
              <li key={`${w.phone}-${w.position}`}>
                {w.phone} ({w.position}) — {fmt(w.mean_score)} {BAND_LABEL[band(w.mean_score)]}
              </li>
            ))}
          </ol>
        )}
        {!p.calibrated && <p className="warnings">Vowel calibration not done yet — vowel scores use a default speaker profile.</p>}
      </section>
      <section className="card">
        <h2>Sounds (weakest first)</h2>
        <table>
          <thead><tr><th>Sound</th><th>Position</th><th className="num">Attempts</th><th className="num">Error rate</th><th className="num">Mean score</th><th>Status</th></tr></thead>
          <tbody>
            {phones.map((r) => (
              <tr key={`${r.phone}-${r.position}`}>
                <td>{r.phone}</td><td>{r.position}</td><td className="num">{r.n}</td>
                <td className="num">{Math.round(100 * r.error_rate)}%</td><td className="num">{fmt(r.mean_score)}</td>
                <td>{BAND_LABEL[band(r.mean_score)]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="card">
        <h2>Recent attempts</h2>
        <table>
          <tbody>
            {p.attempts.map((a) => (
              <tr key={a.id}><td>{new Date(a.created_at).toLocaleString()}</td><td>{a.text}</td><td className="num">{fmt(a.total)}</td></tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
