import { useEffect, useRef, useState } from "react";
import { api2, speak, type ConversationTurn } from "../api";
import { WavRecorder } from "../audio/wav";
import { BAND_LABEL, band, fmt } from "../score";

interface Line {
  role: "user" | "assistant";
  text: string;
  turn?: ConversationTurn;
}

export function Conversation({ user }: { user: string }) {
  const [scenarios, setScenarios] = useState<Record<string, { title: string; opener: string }>>();
  const [scenario, setScenario] = useState("small_talk");
  const [lines, setLines] = useState<Line[]>([]);
  const [state, setState] = useState<"idle" | "recording" | "thinking">("idle");
  const [err, setErr] = useState<string>();
  const rec = useRef<WavRecorder>();

  useEffect(() => {
    api2.scenarios().then(setScenarios).catch((e: Error) => setErr(e.message));
  }, []);

  const begin = (key: string) => {
    setScenario(key);
    const opener = scenarios?.[key]?.opener ?? "Hi!";
    setLines([{ role: "assistant", text: opener }]);
    void speak(opener);
  };

  const toggle = async () => {
    setErr(undefined);
    if (state === "idle") {
      rec.current = new WavRecorder();
      try {
        await rec.current.start();
        setState("recording");
      } catch (e) {
        setErr((e as Error).message);
      }
      return;
    }
    if (state !== "recording" || !rec.current) return;
    setState("thinking");
    const wav = await rec.current.stop();
    try {
      const history = lines.map((l) => ({ role: l.role, text: l.text }));
      const t = await api2.turn(scenario, history, user, wav);
      setLines((ls) => [...ls, { role: "user", text: t.transcript || "(not understood)", turn: t }, { role: "assistant", text: t.reply.text }]);
      void speak(t.reply.text);
    } catch (e) {
      setErr((e as Error).message);
    }
    setState("idle");
  };

  if (!scenarios) return <section className="card">{err ?? "Loading…"}</section>;
  return (
    <section className="card">
      <h2>Conversation practice</h2>
      <div className="row">
        {Object.entries(scenarios).map(([k, v]) => (
          <button key={k} className="btn" aria-pressed={k === scenario && lines.length > 0} onClick={() => begin(k)}>{v.title}</button>
        ))}
      </div>
      {lines.length > 0 && (
        <>
          <div className="chat">
            {lines.map((l, i) => (
              <div key={i} className={`bubble ${l.role}`}>
                <div>{l.text}</div>
                {l.turn?.assessment && (
                  <div className="status" style={{ fontSize: 13, marginTop: 4 }}>
                    Pronunciation {fmt(l.turn.assessment.scores.total)} {BAND_LABEL[band(l.turn.assessment.scores.total)]}
                    {l.turn.feedback?.focus[0] && <> · Tip for “{l.turn.feedback.focus[0].word}”: {l.turn.feedback.focus[0].articulation}</>}
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="row">
            <button className={`btn ${state === "recording" ? "" : "primary"}`} onClick={() => void toggle()} disabled={state === "thinking"}>
              {state === "recording" ? "■ Done speaking" : state === "thinking" ? "…" : "● Speak"}
            </button>
            <span className="status">{state === "recording" ? "Recording — press when you finish." : ""}</span>
          </div>
        </>
      )}
      {err && <p className="bad">{err}</p>}
    </section>
  );
}
