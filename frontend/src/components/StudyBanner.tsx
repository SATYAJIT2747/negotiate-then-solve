import { FlaskConical } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { study, studyCode, studyTask, type StudySession } from "../lib/study";

/** A pilot participant in the portal: whose shoes they are in, what to do, and the way back. */
export function useStudySession(): StudySession | null {
  const { user } = useAuth();
  const [session, setSession] = useState<StudySession | null>(null);
  useEffect(() => {
    if (!studyCode()) return setSession(null);
    study.me().then(setSession, () => setSession(null));
  }, [user?.id]);
  return session && session.kind === "pilot" && session.consented && user?.id === session.persona ? session : null;
}

export function StudyBanner() {
  const session = useStudySession();
  const task = studyTask();
  if (!session) return null;
  return (
    <div className="mb-6 flex flex-wrap items-start gap-3 rounded-lg border border-brand/30 bg-brand/5 px-4 py-3 text-[13.5px]">
      <FlaskConical size={16} className="mt-0.5 shrink-0 text-brand" />
      <div className="min-w-0 flex-1">
        <p>
          <span className="font-medium">Study session {session.code}</span>
          <span className="text-ink-3"> · you are {session.persona_name}</span>
        </p>
        {task && <p className="mt-0.5 text-ink-2">{task}</p>}
      </div>
      <Link to="/study" className="shrink-0 font-medium text-brand hover:underline">
        Study tasks
      </Link>
    </div>
  );
}

const CHECKED_KEY = "nts.study.checked";
const checked = (): string[] => {
  try {
    return JSON.parse(localStorage.getItem(CHECKED_KEY) ?? "[]");
  } catch {
    return [];
  }
};

/** Study: under the assistant's answer to a timetable or rules question, a pilot participant says whether it was
 * right. Asked once per answer. */
export function StudyAnswerCheck({ n, step, answer }: { n: number; step: "timetable" | "rules"; answer: string }) {
  const session = useStudySession();
  const key = `${session?.code}/${n}`;
  const [saved, setSaved] = useState<string | null>(checked().includes(key) ? "saved" : null);
  const [comment, setComment] = useState("");
  if (!session) return null;
  const save = async (value: "yes" | "partly" | "no") => {
    await study.live({ kind: "answer", step, n, value, reading: answer, ...(comment.trim() ? { comment: comment.trim() } : {}) });
    try {
      localStorage.setItem(CHECKED_KEY, JSON.stringify([...checked(), key]));
    } catch {
      /* asked again after a reload */
    }
    setSaved(value);
  };
  if (saved) return <p className="ml-1 mt-1 text-[11.5px] text-ok">Study: answer saved. Thank you.</p>;
  return (
    <div className="ml-1 mt-1.5 space-y-1.5 rounded-md border border-brand/30 bg-brand/5 px-2.5 py-2 text-[12.5px]">
      <p className="flex items-center gap-1.5 font-medium">
        <FlaskConical size={12} className="text-brand" />
        {step === "rules" ? "Study: was this right, and backed by the rule it cites?" : "Study: was this answer right?"}
      </p>
      <input
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        placeholder="What was wrong? (optional)"
        className="w-full rounded border border-line bg-panel px-2 py-1 text-[12.5px] outline-none focus:border-brand/60"
      />
      <div className="flex gap-1.5">
        {(["yes", "partly", "no"] as const).map((v) => (
          <button key={v} onClick={() => void save(v)} className="rounded border border-line bg-panel px-2.5 py-0.5 hover:border-brand/60">
            {v === "yes" ? "Yes" : v === "partly" ? "Partly" : "No"}
          </button>
        ))}
      </div>
    </div>
  );
}
