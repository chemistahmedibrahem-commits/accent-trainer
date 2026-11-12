import { useEffect, useState } from "react";
import { api, speak, type Content } from "../api";

/** Minimal-pair perception test: hear one word (en-US TTS), pick which it was. */
export function EarTraining() {
  const [content, setContent] = useState<Content>();
  const [contrast, setContrast] = useState("IH_IY");
  const [trial, setTrial] = useState<{ pair: [string, string]; answer: number }>();
  const [feedback, setFeedback] = useState<string>();
  const [stats, setStats] = useState({ n: 0, correct: 0 });

  useEffect(() => {
    api.content().then(setContent).catch(() => undefined);
  }, []);
  const say = (w: string) => void speak(w);

  const next = () => {
    const pairs = content?.minimal_pairs[contrast];
    if (!pairs?.length) return;
    const pair = pairs[Math.floor(Math.random() * pairs.length)];
    const answer = Math.random() < 0.5 ? 0 : 1;
    setTrial({ pair, answer });
    setFeedback(undefined);
    say(pair[answer]);
  };

  const choose = (i: number) => {
    if (!trial || feedback) return;
    const ok = i === trial.answer;
    setStats((s) => ({ n: s.n + 1, correct: s.correct + (ok ? 1 : 0) }));
    setFeedback(ok ? "✓ Correct" : `✗ It was “${trial.pair[trial.answer]}”`);
  };

  if (!content) return <section className="card">Loading…</section>;
  return (
    <section className="card">
      <h2>Ear training — minimal pairs</h2>
      <p className="status">Train your ear before your mouth: listen and choose the word you heard.</p>
      <div className="row">
        <select value={contrast} onChange={(e) => { setContrast(e.target.value); setTrial(undefined); }}>
          {Object.keys(content.minimal_pairs).map((k) => (
            <option key={k} value={k}>{k.replace("_", " vs ")}</option>
          ))}
        </select>
        <button className="btn primary" onClick={next}>{trial ? "Next" : "Start"}</button>
        {trial && <button className="btn" onClick={() => say(trial.pair[trial.answer])}>▶ Replay</button>}
      </div>
      {trial && (
        <div className="pair" style={{ marginTop: 16 }}>
          {trial.pair.map((w, i) => (
            <button key={w} className="btn" onClick={() => choose(i)}>{w}</button>
          ))}
        </div>
      )}
      {feedback && <p className={feedback.startsWith("✓") ? "ok" : "bad"}>{feedback}</p>}
      {stats.n > 0 && <p className="status">{stats.correct}/{stats.n} correct ({Math.round((100 * stats.correct) / stats.n)}%)</p>}
    </section>
  );
}
