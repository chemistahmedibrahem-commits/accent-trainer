import { describe, expect, it } from "vitest";
import { band, fmt, toSemitones } from "./score";

describe("score helpers", () => {
  it("bands", () => {
    expect(band(90)).toBe("good");
    expect(band(60)).toBe("warning");
    expect(band(10)).toBe("critical");
    expect(fmt(null)).toBe("–");
  });
  it("semitones re median", () => {
    const st = toSemitones([100, null, 200, 100]);
    expect(st[0]).toBeCloseTo(0);
    expect(st[1]).toBeNull();
    expect(st[2]).toBeCloseTo(12);
  });
});
