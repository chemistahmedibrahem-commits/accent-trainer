export interface PhoneResult {
  index: number;
  phone: string;
  ipa: string;
  stress: number | null;
  allophone: string | null;
  realized: string | null;
  score: number;
  lpr: number;
  start: number | null;
  end: number | null;
  formants: { f1: number; f2: number; f3: number } | null;
  vowel_score: { vowel: string; point: [number, number]; target: [number, number]; distance: number; nearest: string; score: number } | null;
}

export interface WordResult {
  index: number;
  text: string;
  accuracy: number;
  phones: PhoneResult[];
  stress: { expected_syllable: number; produced_syllable: number; correct: boolean } | null;
}

export interface ErrorItem {
  code: string;
  name: string;
  severity: "minor" | "moderate" | "major";
  tip: string;
  word_index: number;
  word: string;
  target_index: number | null;
  expected: string | null;
  realized: string | null;
  source: string;
  confidence: number;
}

export interface Track {
  times: number[];
  f0: (number | null)[];
  energy_db: number[];
}

export interface Assessment {
  text: string;
  words: WordResult[];
  errors: ErrorItem[];
  recognized: string[];
  scores: Record<"accuracy" | "completeness" | "fluency" | "prosody" | "stress" | "rhythm" | "intonation" | "total", number | null>;
  prosody: { track?: Track; reference_track?: Track; rhythm?: { npvi_v: number | null } };
  timing_ms: Record<string, number>;
  warnings: string[];
  model: string;
}

export type ServerMessage =
  | { type: "ready" }
  | { type: "level"; db: number }
  | { type: "speech_start"; t: number }
  | { type: "speech_end"; t: number; duration: number }
  | ({ type: "result"; latency_ms: number; attempt_id: number } & Assessment)
  | { type: "error"; message: string };

export interface Content {
  minimal_pairs: Record<string, [string, string][]>;
  sentences: Record<string, string[]>;
  calibration: Record<string, string>;
}

export interface VowelReference {
  sd: number;
  vowels: Record<string, { z_f1: number; z_f2: number; ipa: string }>;
}

export interface Progress {
  user: string;
  calibrated: boolean;
  phones: { phone: string; position: string; n: number; error_rate: number; mean_score: number }[];
  weaknesses: { phone: string; position: string; mean_score: number }[];
  attempts: { id: number; text: string; total: number | null; created_at: string }[];
}

async function get<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return (await r.json()) as T;
}

export const api = {
  content: () => get<Content>("/api/content"),
  vowelReference: () => get<VowelReference>("/api/vowel-reference"),
  progress: (user: string) => get<Progress>(`/api/users/${encodeURIComponent(user)}/progress`),
  health: () => get<{ backend: string; model: string }>("/api/health"),
};

export interface CoachReply {
  summary: string;
  focus: { code: string; word: string; explanation: string; articulation: string }[];
  drills: string[];
  source: "llm" | "rules";
  fallback_reason?: string;
}

export interface Drill {
  kind: "perception" | "words" | "twister" | "stress_pairs";
  skill: string;
  label: string;
  items: (string | string[])[];
  reason: string;
}

export interface Dashboard {
  user: string;
  n_attempts: number;
  skills: { skill: string; label: string; mastery: number; mastered: boolean; n_obs: number; correct_rate: number | null; due: string | null }[];
  weaknesses: { skill: string; label: string; mastery: number; n_obs: number }[];
  heatmap: { skill: string; label: string; week: string; n: number; correct_rate: number }[];
  trend: { day: string; n: number; total: number | null; accuracy: number | null; fluency: number | null; prosody: number | null }[];
}

export interface ConversationTurn {
  transcript: string;
  reply: { text: string; source: string };
  attempt_id: number | null;
  assessment: Assessment | null;
  feedback: CoachReply | null;
}

export const api2 = {
  coach: (attemptId: number) => get<CoachReply>(`/api/attempts/${attemptId}/coach`),
  plan: (user: string) => get<{ drills: Drill[] }>(`/api/users/${encodeURIComponent(user)}/plan`),
  dashboard: (user: string) => get<Dashboard>(`/api/users/${encodeURIComponent(user)}/dashboard`),
  scenarios: () => get<Record<string, { title: string; opener: string }>>("/api/conversation/scenarios"),
  async turn(scenario: string, history: { role: string; text: string }[], user: string, wav: Blob): Promise<ConversationTurn> {
    const fd = new FormData();
    fd.append("scenario", scenario);
    fd.append("history", JSON.stringify(history));
    fd.append("user", user);
    fd.append("audio", wav, "turn.wav");
    const r = await fetch("/api/conversation/turn", { method: "POST", body: fd });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    return (await r.json()) as ConversationTurn;
  },
};

/** Play American reference audio (server Piper TTS), falling back to the browser voice. */
export async function speak(text: string, rate = 1): Promise<void> {
  try {
    const audio = new Audio(`/api/tts?text=${encodeURIComponent(text)}&rate=${rate}`);
    await audio.play();
  } catch {
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US";
    u.rate = 0.9 * rate;
    speechSynthesis.cancel();
    speechSynthesis.speak(u);
  }
}
