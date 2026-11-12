import { describe, expect, it } from "vitest";
import { encodeWav } from "./wav";

describe("encodeWav", () => {
  it("writes a valid 16 kHz mono PCM header", async () => {
    const blob = encodeWav(new Float32Array([0, 0.5, -0.5, 1]));
    const b = new DataView(await blob.arrayBuffer());
    const tag = (o: number) => String.fromCharCode(...[0, 1, 2, 3].map((i) => b.getUint8(o + i)));
    expect(tag(0)).toBe("RIFF");
    expect(tag(8)).toBe("WAVE");
    expect(b.getUint32(24, true)).toBe(16000);
    expect(b.getUint32(40, true)).toBe(8);
    expect(b.getInt16(46, true)).toBe(16383);
  });
});
