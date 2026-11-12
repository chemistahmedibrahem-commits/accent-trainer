/** Score bands. Status color always ships with a text label / icon, never alone. */
export type Band = "good" | "warning" | "critical";

export function band(score: number | null | undefined): Band {
  if (score == null || Number.isNaN(score)) return "warning";
  if (score >= 75) return "good";
  if (score >= 50) return "warning";
  return "critical";
}

export const BAND_LABEL: Record<Band, string> = { good: "✓ good", warning: "~ close", critical: "✗ fix" };

export function fmt(score: number | null | undefined): string {
  return score == null || Number.isNaN(score) ? "–" : Math.round(score).toString();
}

/** Semitones relative to the median voiced F0 (speaker-independent contour). */
export function toSemitones(f0: (number | null)[]): (number | null)[] {
  const voiced = f0.filter((v): v is number => v != null && v > 0).sort((a, b) => a - b);
  if (!voiced.length) return f0.map(() => null);
  const med = voiced[Math.floor(voiced.length / 2)];
  return f0.map((v) => (v != null && v > 0 ? 12 * Math.log2(v / med) : null));
}
