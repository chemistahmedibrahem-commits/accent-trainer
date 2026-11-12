import { describe, expect, it } from "vitest";
import { Chunker, StreamingResampler, toInt16 } from "./resample";

function tone(freq: number, rate: number, seconds: number): Float32Array {
  const n = Math.round(rate * seconds);
  return Float32Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * freq * i) / rate));
}

function rms(x: Float32Array, skip = 200): number {
  let s = 0;
  for (let i = skip; i < x.length - skip; i++) s += x[i] * x[i];
  return Math.sqrt(s / (x.length - 2 * skip));
}

describe("StreamingResampler", () => {
  it("produces the right number of samples across arbitrary chunk sizes", () => {
    const r = new StreamingResampler(48000);
    const input = tone(440, 48000, 1);
    let total = 0;
    for (let i = 0; i < input.length; i += 997) total += r.push(input.subarray(i, i + 997)).length;
    expect(Math.abs(total - 16000)).toBeLessThan(40);
  });

  it("passes in-band speech frequencies", () => {
    const out = new StreamingResampler(48000).push(tone(1000, 48000, 0.5));
    expect(rms(out)).toBeGreaterThan(0.65); // sine RMS = 0.707
  });

  it("attenuates content above the 8 kHz Nyquist (anti-aliasing)", () => {
    const out = new StreamingResampler(44100).push(tone(12000, 44100, 0.5));
    expect(rms(out)).toBeLessThan(0.05);
  });

  it("preserves the waveform of a 300 Hz tone", () => {
    const out = new StreamingResampler(48000).push(tone(300, 48000, 0.2));
    const ref = tone(300, 16000, 0.2);
    // filter delay is compensated by the half-length left padding → aligned
    let err = 0;
    for (let i = 100; i < out.length - 100; i++) err += (out[i] - ref[i]) ** 2;
    expect(Math.sqrt(err / (out.length - 200))).toBeLessThan(0.05);
  });
});

describe("helpers", () => {
  it("toInt16 clips and scales", () => {
    expect(Array.from(toInt16(Float32Array.from([0, 1, -1, 2])))).toEqual([0, 32767, -32768, 32767]);
  });
  it("Chunker emits fixed frames", () => {
    const c = new Chunker(4);
    expect(c.push(new Float32Array(3))).toHaveLength(0);
    expect(c.push(new Float32Array(6))).toHaveLength(2);
  });
});
