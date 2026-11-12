/**
 * Streaming resampler: native rate (44.1/48 kHz) → 16 kHz mono.
 * Windowed-sinc low-pass (cutoff 7.2 kHz) + fractional interpolation, so sibilant
 * energy above 8 kHz doesn't alias into the band the phone model listens to.
 */
export class StreamingResampler {
  private readonly ratio: number;
  private readonly taps: Float32Array;
  private readonly half: number;
  private history: Float32Array;
  private pos = 0; // fractional read position into `history`, in input samples

  constructor(
    readonly inRate: number,
    readonly outRate = 16000,
    numTaps = 63,
  ) {
    this.ratio = inRate / outRate;
    this.half = Math.floor(numTaps / 2);
    const fc = Math.min(0.45 * outRate, 7200) / inRate;
    this.taps = new Float32Array(numTaps);
    let sum = 0;
    for (let i = 0; i < numTaps; i++) {
      const n = i - this.half;
      const sinc = n === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * n) / (Math.PI * n);
      const w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (numTaps - 1)) + 0.08 * Math.cos((4 * Math.PI * i) / (numTaps - 1));
      this.taps[i] = sinc * w;
      sum += this.taps[i];
    }
    for (let i = 0; i < numTaps; i++) this.taps[i] /= sum;
    this.history = new Float32Array(this.half); // left zero padding
    this.pos = this.half;
  }

  private filteredAt(i: number): number {
    let acc = 0;
    const base = i - this.half;
    for (let k = 0; k < this.taps.length; k++) acc += this.taps[k] * this.history[base + k];
    return acc;
  }

  push(input: Float32Array): Float32Array {
    if (this.ratio === 1) return input.slice();
    const merged = new Float32Array(this.history.length + input.length);
    merged.set(this.history);
    merged.set(input, this.history.length);
    this.history = merged;
    const out: number[] = [];
    // need `half` samples of right context for the FIR
    while (Math.floor(this.pos) + 1 + this.half < this.history.length) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      const a = this.filteredAt(i);
      const b = this.filteredAt(i + 1);
      out.push(a + (b - a) * frac);
      this.pos += this.ratio;
    }
    // drop consumed samples, keeping FIR context
    const keepFrom = Math.max(0, Math.floor(this.pos) - this.half);
    this.history = this.history.slice(keepFrom);
    this.pos -= keepFrom;
    return Float32Array.from(out);
  }
}

export function toInt16(x: Float32Array): Int16Array {
  const out = new Int16Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = Math.max(-1, Math.min(1, x[i]));
    out[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  return out;
}

/** Re-chunks a sample stream into fixed-size frames (e.g. 640 = 40 ms @16 kHz). */
export class Chunker {
  private buf: number[] = [];
  constructor(readonly size: number) {}
  push(x: Float32Array): Float32Array[] {
    for (let i = 0; i < x.length; i++) this.buf.push(x[i]);
    const out: Float32Array[] = [];
    while (this.buf.length >= this.size) out.push(Float32Array.from(this.buf.splice(0, this.size)));
    return out;
  }
}
