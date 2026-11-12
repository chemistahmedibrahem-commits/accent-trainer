import { useState } from "react";
import { EarTraining } from "./components/EarTraining";
import { Practice } from "./components/Practice";
import { ProgressView } from "./components/ProgressView";

type Tab = "practice" | "ear" | "progress";

export function App() {
  const [tab, setTab] = useState<Tab>("practice");
  const [user, setUser] = useState(() => {
    try {
      return localStorage.getItem("user") ?? "me";
    } catch {
      return "me";
    }
  });
  const changeUser = (u: string) => {
    setUser(u);
    try {
      localStorage.setItem("user", u);
    } catch {
      /* storage unavailable */
    }
  };
  return (
    <main>
      <header>
        <h1>American Accent Trainer</h1>
        <nav>
          {(["practice", "ear", "progress"] as Tab[]).map((t) => (
            <button key={t} aria-pressed={tab === t} onClick={() => setTab(t)}>
              {{ practice: "Practice", ear: "Ear training", progress: "Progress" }[t]}
            </button>
          ))}
        </nav>
        <input type="text" value={user} onChange={(e) => changeUser(e.target.value)} aria-label="User name" style={{ maxWidth: 120, flex: "none" }} />
      </header>
      {tab === "practice" && <Practice user={user} />}
      {tab === "ear" && <EarTraining />}
      {tab === "progress" && <ProgressView user={user} />}
    </main>
  );
}
