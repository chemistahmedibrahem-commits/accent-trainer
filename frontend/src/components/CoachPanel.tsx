import { useEffect, useState } from "react";
import { api2, speak, type CoachReply } from "../api";

export function CoachPanel({ attemptId, onPractice }: { attemptId: number; onPractice?: (s: string) => void }) {
  const [c, setC] = useState<CoachReply>();
  const [err, setErr] = useState<string>();
  useEffect(() => {
    setC(undefined);
    api2.coach(attemptId).then(setC).catch((e: Error) => setErr(e.message));
  }, [attemptId]);
  if (err) return null;
  return (
    <section className="card">
      <h2>Coach{c?.source === "rules" ? " (offline mode)" : ""}</h2>
      {!c ? (
        <p className="status">Thinking about your attempt…</p>
      ) : (
        <>
          <p>{c.summary}</p>
          {c.focus.length > 0 && (
            <ul className="errors">
              {c.focus.map((f, i) => (
                <li key={i} className="major">
                  <div className="t">“{f.word}”</div>
                  <div>{f.explanation}</div>
                  <div className="tip">👄 {f.articulation}</div>
                </li>
              ))}
            </ul>
          )}
          <h2 style={{ marginTop: 16 }}>Drills</h2>
          <ul className="drills">
            {c.drills.map((d) => (
              <li key={d} className="row">
                <span style={{ flex: 1 }}>{d}</span>
                <button className="btn" onClick={() => void speak(d)} aria-label={`Listen to: ${d}`}>▶</button>
                {onPractice && <button className="btn" onClick={() => onPractice(d)}>Practice</button>}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
