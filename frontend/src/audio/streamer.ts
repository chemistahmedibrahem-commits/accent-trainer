import { Chunker, StreamingResampler, toInt16 } from "./resample";
import type { ServerMessage } from "../api";

export interface StreamerOptions {
  text: string;
  user: string;
  mode?: string;
  onMessage: (m: ServerMessage) => void;
  onRecording?: (blob: Blob) => void;
}

/**
 * Mic → AudioWorklet (native rate) → 16 kHz int16 40 ms frames → WebSocket.
 * Also keeps a local recording of the session for side-by-side playback.
 */
export class MicStreamer {
  private ctx?: AudioContext;
  private stream?: MediaStream;
  private ws?: WebSocket;
  private node?: AudioWorkletNode;
  private recorder?: MediaRecorder;

  async start(opts: StreamerOptions): Promise<void> {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    ws.onmessage = (ev) => opts.onMessage(JSON.parse(ev.data as string) as ServerMessage);
    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => resolve();
      ws.onerror = () => reject(new Error("WebSocket connection failed"));
    });
    ws.send(JSON.stringify({ type: "format", dtype: "int16" }));
    ws.send(JSON.stringify({ type: "start", text: opts.text, user: opts.user, mode: opts.mode ?? "repeat" }));

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: true },
    });
    this.ctx = new AudioContext();
    await this.ctx.audioWorklet.addModule("/pcm-worklet.js");
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "pcm-capture");
    const rs = new StreamingResampler(this.ctx.sampleRate, 16000);
    const chunker = new Chunker(640);
    this.node.port.onmessage = (ev: MessageEvent<Float32Array>) => {
      for (const frame of chunker.push(rs.push(ev.data))) {
        if (ws.readyState === WebSocket.OPEN) ws.send(toInt16(frame).buffer);
      }
    };
    src.connect(this.node);

    if (opts.onRecording && typeof MediaRecorder !== "undefined") {
      const parts: Blob[] = [];
      this.recorder = new MediaRecorder(this.stream);
      this.recorder.ondataavailable = (e) => parts.push(e.data);
      this.recorder.onstop = () => opts.onRecording?.(new Blob(parts, { type: this.recorder?.mimeType }));
      this.recorder.start();
    }
  }

  setText(text: string): void {
    this.ws?.send(JSON.stringify({ type: "text", text }));
  }

  /** Stop capturing; asks the server to close any open utterance. WS stays open for the result. */
  stop(): void {
    this.recorder?.stop();
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    void this.ctx?.close();
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: "stop" }));
  }

  close(): void {
    this.ws?.close();
  }
}
