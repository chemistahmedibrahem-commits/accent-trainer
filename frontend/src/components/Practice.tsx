import { useEffect, useRef, useState } from "react";
import { api, speak, type Assessment, type Content, type ServerMessage } from "../api";
import { MicStreamer } from "../audio/streamer";
import { ResultView } from "./ResultView";

type State = "idle" | "listening" | "speaking" | "processing";

export function Practice({ user, initialText }: { user: string; initialText?: string }) {
  const [content, setContent] = useState<Content>();
  const [text, setText] = useState(initialText ?? "I'd like a bottle of water.");
  useEffect(() => {
    if (initialText) setText(initialText);
  }, [initialText]);
  const [state, setState] = useState<State>("idle");
  const [level, setLevel] = useState(0);
  const [result, setResult] = useState<Assessment & { latency_ms?: number; attempt_id?: number }>();
  const [recording, setRecording] = useState<string>();
  const [error, setError] = useState<string>();
  const streamer = useRef<MicStreamer>();

  useEffect(() => {
    api.content().then(setContent).catch(() => undefined);
    return () => streamer.current?.close();
  }, []);

  const onMessage = (m: ServerMessage) => {
    switch (m.type) {
      case "level":
        setLevel(Math.max(0, Math.min(1, (m.db + 60) / 50)));
        break;
      case "speech_start":
        setState("speaking");
        break;
      case "speech_end":
        setState("processing");
        break;
      case "result":
        setResult(m);
        setState("idle");
        streamer.current?.stop();
        break;
      case "error":
        setError(m.message);
        setState("idle");
        break;
    }
  };

  const start = async () => {
    setError(undefined);
    setResult(undefined);
    streamer.current?.close();
    const s = new MicStreamer();
    streamer.current = s;
    try {
      await s.start({ text, user, onMessage, onRecording: (b) => setRecording(URL.createObjectURL(b)) });
      setState("listening");
    } catch (e) {
      setError((e as Error).message);
      setState("idle");
    }
  };

  const listen = (rate = 1) => void speak(text, rate);

  const status: Record<State, string> = {
    idle: "Press Record, then read the sentence aloud. Recording stops automatically after you finish.",
    listening: "Listening… start speaking.",
    speaking: "Hearing you…",
    processing: "Analysing…",
  };

  return (
    <>
      <section className="card">
        <h2>Listen &amp; repeat</h2>
        <div className="row">
          <input type="text" value={text} onChange={(e) => setText(e.target.value)} aria-label="Target sentence" />
          {content && (
            <select aria-label="Practice set" value="" onChange={(e) => e.target.value && setText(e.target.value)}>
              <option value="">Practice sets…</option>
              {Object.entries(content.sentences).map(([k, list]) => (
                <optgroup key={k} label={k}>
                  {list.map((s) => <option key={s} value={s}>{s}</option>)}
                </optgroup>
              ))}
            </select>
          )}
        </div>
        <p className="target">{text}</p>
        <div className="row">
          <button className="btn" onClick={() => listen()}>▶ Listen</button>
          <button className="btn" onClick={() => listen(0.75)}>▶ Slow</button>
          {state === "idle" ? (
            <button className="btn primary" onClick={start}>● Record</button>
          ) : (
            <button className="btn" onClick={() => streamer.current?.stop()}>■ Stop</button>
          )}
          <div className="level" aria-hidden><div style={{ width: `${state === "idle" ? 0 : level * 100}%` }} /></div>
        </div>
        <p className="status">{status[state]}{result?.latency_ms != null && state === "idle" ? ` (analysed in ${Math.round(result.latency_ms)} ms)` : ""}</p>
        {error && <p className="bad">{error}</p>}
      </section>
      {result && <ResultView result={result} recording={recording} onPractice={setText} />}
    </>
  );
}
