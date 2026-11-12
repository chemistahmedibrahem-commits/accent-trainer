import { StreamingResampler, toInt16 } from "./resample";

/** Encode 16 kHz mono float samples as a 16-bit PCM WAV blob. */
export function encodeWav(samples: Float32Array, sampleRate = 16000): Blob {
  const pcm = toInt16(samples);
  const buf = new ArrayBuffer(44 + pcm.byteLength);
  const v = new DataView(buf);
  const str = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF");
  v.setUint32(4, 36 + pcm.byteLength, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, pcm.byteLength, true);
  new Int16Array(buf, 44).set(pcm);
  return new Blob([buf], { type: "audio/wav" });
}

/** Push-to-talk recorder: mic → worklet → 16 kHz → WAV (for the conversation turn upload). */
export class WavRecorder {
  private ctx?: AudioContext;
  private stream?: MediaStream;
  private parts: Float32Array[] = [];

  async start(): Promise<void> {
    this.parts = [];
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } });
    this.ctx = new AudioContext();
    await this.ctx.audioWorklet.addModule("/pcm-worklet.js");
    const node = new AudioWorkletNode(this.ctx, "pcm-capture");
    const rs = new StreamingResampler(this.ctx.sampleRate, 16000);
    node.port.onmessage = (ev: MessageEvent<Float32Array>) => this.parts.push(rs.push(ev.data));
    this.ctx.createMediaStreamSource(this.stream).connect(node);
  }

  async stop(): Promise<Blob> {
    this.stream?.getTracks().forEach((t) => t.stop());
    if (this.ctx && this.ctx.state !== "closed") await this.ctx.close();
    const n = this.parts.reduce((a, p) => a + p.length, 0);
    const all = new Float32Array(n);
    let o = 0;
    for (const p of this.parts) {
      all.set(p, o);
      o += p.length;
    }
    return encodeWav(all);
  }
}
