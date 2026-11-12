import { useEffect, useState } from "react";
import { api, type Assessment, type VowelReference } from "../api";
import { BAND_LABEL, band, fmt } from "../score";
import { PitchChart } from "./PitchChart";
import { VowelChart } from "./VowelChart";

const TILES: [keyof Assessment["scores"], string][] = [
  ["total", "Overall"],
  ["accuracy", "Accuracy"],
  ["completeness", "Completeness"],
  ["fluency", "Fluency"],
  ["prosody", "Prosody"],
];

export function ResultView({ result, recording }: { result: Assessment; recording?: string }) {
  const [vref, setVref] = useState<VowelReference>();
  useEffect(() => {
    api.vowelReference().then(setVref).catch(() => undefined);
  }, []);
  const vowels = result.words.flatMap((w) => w.phones.filter((p) => p.vowel_score).map((phone) => ({ word: w.text, phone })));

  return (
    <>
      <section className="card">
        <h2>Scores</h2>
        <div className="tiles">
          {TILES.map(([k, label]) => (
            <div className="tile" key={k}>
              <div className="k">{label}</div>
              <div className="v">{fmt(result.scores[k])}</div>
              {result.scores[k] != null && <div className="b">{BAND_LABEL[band(result.scores[k])]}</div>}
            </div>
          ))}
        </div>
        {recording && (
          <div className="row" style={{ marginTop: 12 }}>
            <span className="status">Your recording</span>
            <audio controls src={recording} />
          </div>
        )}
      </section>

      <section className="card">
        <h2>Word by word</h2>
        <div className="words">
          {result.words.map((w) => (
            <div className="word" key={w.index}>
              <span className="w">{w.text}</span>
              <span className="phones">
                {w.phones.map((p) => (
                  <span
                    key={p.index}
                    className={`ph ${band(p.score)}`}
                    title={`/${p.ipa}/ ${p.realized && p.realized !== p.phone ? `heard ${p.realized} · ` : ""}score ${fmt(p.score)}`}
                  >
                    {p.ipa}
                    {p.stress === 1 ? "ˈ" : ""}
                  </span>
                ))}
              </span>
              {w.stress && !w.stress.correct && (
                <span className="stress-bad">✗ stress on syllable {w.stress.produced_syllable + 1}, expected {w.stress.expected_syllable + 1}</span>
              )}
            </div>
          ))}
        </div>
        <div className="legend">
          <span><span className="swatch" style={{ background: "var(--good)" }} />✓ good (≥75)</span>
          <span><span className="swatch" style={{ background: "var(--warning)" }} />~ close (50–74)</span>
          <span><span className="swatch" style={{ background: "var(--critical)" }} />✗ fix (&lt;50)</span>
        </div>
      </section>

      <section className="card">
        <h2>What to work on</h2>
        {result.errors.length === 0 ? (
          <p className="ok">✓ No pronunciation errors detected.</p>
        ) : (
          <ul className="errors">
            {result.errors.map((e, i) => (
              <li key={i} className={e.severity}>
                <div className="t">
                  “{e.word}” — {e.name}
                  {e.realized && e.realized !== "<del>" && e.expected ? ` (heard ${e.realized} for ${e.expected})` : ""}
                  {e.realized === "<del>" ? " (sound missing)" : ""}
                </div>
                {e.tip && <div className="tip">{e.tip}</div>}
              </li>
            ))}
          </ul>
        )}
        {result.warnings.length > 0 && <p className="warnings">{result.warnings.join(" · ")}</p>}
      </section>

      <div className="grid2">
        {result.prosody.track && (
          <section className="card">
            <h2>Intonation (pitch contour)</h2>
            <PitchChart user={result.prosody.track} reference={result.prosody.reference_track} />
          </section>
        )}
        {vref && vowels.length > 0 && (
          <section className="card">
            <h2>Vowel chart</h2>
            <VowelChart reference={vref} phones={vowels} />
          </section>
        )}
      </div>
    </>
  );
}
