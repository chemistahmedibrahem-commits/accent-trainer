import { useState } from "react";
import { Conversation } from "./components/Conversation";
import { DashboardView } from "./components/DashboardView";
import { EarTraining } from "./components/EarTraining";
import { Practice } from "./components/Practice";
import { Today } from "./components/Today";

type Tab = "today" | "practice" | "conversation" | "ear" | "progress";
const LABEL: Record<Tab, string> = {
  today: "Today",
  practice: "Practice",
  conversation: "Conversation",
  ear: "Ear training",
  progress: "Progress",
};

export function App() {
  const [tab, setTab] = useState<Tab>("practice");
  const [practiceText, setPracticeText] = useState<string>();
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
  const practise = (text: string) => {
    setPracticeText(text);
    setTab("practice");
  };
  return (
    <main>
      <header>
        <h1>American Accent Trainer</h1>
        <nav>
          {(Object.keys(LABEL) as Tab[]).map((t) => (
            <button key={t} aria-pressed={tab === t} onClick={() => setTab(t)}>
              {LABEL[t]}
            </button>
          ))}
        </nav>
        <input type="text" value={user} onChange={(e) => changeUser(e.target.value)} aria-label="User name" style={{ maxWidth: 120, flex: "none" }} />
      </header>
      {tab === "today" && <Today user={user} onPractice={practise} />}
      {tab === "practice" && <Practice user={user} initialText={practiceText} />}
      {tab === "conversation" && <Conversation user={user} />}
      {tab === "ear" && <EarTraining />}
      {tab === "progress" && <DashboardView user={user} />}
    </main>
  );
}
