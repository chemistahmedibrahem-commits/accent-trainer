import { useEffect, useState } from "react";
import { api2, speak, type Drill } from "../api";

/** Adaptive study plan: what to practise next, from BKT mastery + FSRS due dates. */
export function Today({ user, onPractice }: { user: string; onPractice: (text: string) => void }) {
  const [drills, setDrills] = useState<Drill[]>();
  const [err, setErr] = useState<string>();
  useEffect(() => {
    api2.plan(user).then((p) => setDrills(p.drills)).catch((e: Error) => setErr(e.message));
  }, [user]);
  if (err) return <section className="card">Practise a few sentences first — your plan is built from your results.</section>;
  if (!drills) return <section className="card">Loading…</section>;
  if (drills.length === 0)
    return (
      <section className="card">
        <h2>Today</h2>
        <p>Nothing is due right now. Practise new sentences, or try a conversation.</p>
      </section>
    );
  return (
    <>
      {drills.map((d, i) => (
        <section className="card" key={i}>
          <h2>
            {d.label} — {{ perception: "ear training", words: "word drill", twister: "tongue twister", stress_pairs: "stress pairs" }[d.kind]}
          </h2>
          <p className="status">{d.reason}</p>
          {d.kind === "perception" && <PairQuiz pairs={d.items as string[][]} />}
          {d.kind === "words" && (
            <div className="row">
              {(d.items as string[]).map((w) => (
                <button key={w} className="btn" onClick={() => onPractice(w)}>{w}</button>
              ))}
              <button className="btn primary" onClick={() => onPractice((d.items as string[]).join(", "))}>Practise all</button>
            </div>
          )}
          {d.kind === "twister" &&
            (d.items as string[]).map((t) => (
              <div key={t} className="row" style={{ marginBottom: 6 }}>
                <span style={{ flex: 1 }}>{t}</span>
                <button className="btn" onClick={() => void speak(t)}>▶</button>
                <button className="btn" onClick={() => onPractice(t)}>Practise</button>
              </div>
            ))}
          {d.kind === "stress_pairs" &&
            (d.items as string[][]).slice(0, 5).map(([w]) => (
              <div key={w} className="row" style={{ marginBottom: 6 }}>
                <span style={{ flex: 1 }}>
                  a <b>{w}</b> (noun) · to <b>{w}</b> (verb)
                </span>
                <button className="btn" onClick={() => onPractice(`I want to ${w} the ${w}.`)}>Practise</button>
              </div>
            ))}
        </section>
      ))}
    </>
  );
}

function PairQuiz({ pairs }: { pairs: string[][] }) {
  const [k, setK] = useState(0);
  const [ans, setAns] = useState<number>();
  const [fb, setFb] = useState<string>();
  const [score, setScore] = useState({ n: 0, ok: 0 });
  const pair = pairs[k % pairs.length];
  const play = () => {
    const a = Math.random() < 0.5 ? 0 : 1;
    setAns(a);
    setFb(undefined);
    void speak(pair[a]);
  };
  const pick = (i: number) => {
    if (ans === undefined || fb) return;
    const ok = i === ans;
    setScore((s) => ({ n: s.n + 1, ok: s.ok + (ok ? 1 : 0) }));
    setFb(ok ? "✓ Correct" : `✗ It was “${pair[ans]}”`);
  };
  return (
    <div>
      <div className="row">
        <button className="btn primary" onClick={play}>▶ Play a word</button>
        {fb && <button className="btn" onClick={() => { setK(k + 1); setAns(undefined); setFb(undefined); }}>Next pair</button>}
      </div>
      <div className="pair" style={{ marginTop: 12 }}>
        {pair.map((w, i) => (
          <button key={w} className="btn" disabled={ans === undefined} onClick={() => pick(i)}>{w}</button>
        ))}
      </div>
      {fb && <p className={fb.startsWith("✓") ? "ok" : "bad"}>{fb}</p>}
      {score.n > 0 && <p className="status">{score.ok}/{score.n} correct</p>}
    </div>
  );
}
