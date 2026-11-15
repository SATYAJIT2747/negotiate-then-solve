import clsx from "clsx";
import { ArrowLeft, ArrowRight, BookOpen, Check, ClipboardCheck, LogOut, MessageSquareText, Play, Star, X } from "lucide-react";
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { LetterView, parsePlacement, type Placement } from "../components/LetterView";
import { RatingForm } from "../components/Rating";
import { Button, Card, CardHeader, Label, Mark, Meter, Skeleton } from "../components/ui";
import { setCurrentWorld } from "../lib/api";
import { useAuth } from "../lib/auth";
import { TIER } from "../lib/meta";
import { setStudyCode, setStudyStep, setStudyTask, study, studyCode, type LiveStep, type NextItem, type StudySession, type StudyTask } from "../lib/study";

const TASK: Record<StudyTask, { title: string; text: string; icon: typeof Star }> = {
  ratings: {
    title: "Rate messages",
    text: "Messages the timetable office sent to faculty about a clash. Rate how clear each is, and how acceptable it would be to receive.",
    icon: Star,
  },
  claims: {
    title: "Check claims",
    text: "One sentence from an explanation, and the facts it was allowed to use. Is everything it states backed by those facts?",
    icon: ClipboardCheck,
  },
  rules: {
    title: "Check rule answers",
    text: "A faculty member's question about the institute's rules, the assistant's answer, and the official rule (if there is one). What does the answer do?",
    icon: BookOpen,
  },
  replies: {
    title: "Label replies",
    text: "A faculty member's reply to a message. What does the reply do?",
    icon: MessageSquareText,
  },
};

export default function Study() {
  const [session, setSession] = useState<StudySession | null>(null);
  const [loading, setLoading] = useState(!!studyCode());
  const [task, setTask] = useState<StudyTask | null>(null);

  const refresh = useCallback(async () => {
    if (!studyCode()) return;
    try {
      setSession(await study.me());
    } catch {
      setStudyCode(null);
      setSession(null);
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => void refresh(), [refresh]);

  const leave = () => {
    setStudyCode(null);
    setStudyTask(null);
    setSession(null);
    setTask(null);
  };

  return (
    <div className="min-h-full bg-bg">
      <header className="flex h-14 items-center gap-2.5 border-b border-line bg-panel px-4 sm:px-8">
        <Mark size={22} />
        <span className="font-serif text-[15px] font-semibold">Timetable study</span>
        {session && (
          <span className="ml-auto flex items-center gap-3 text-[13px] text-ink-3">
            <span className="font-mono">{session.code}</span>
            <button onClick={leave} className="flex items-center gap-1.5 hover:text-ink" title="Leave this device (your answers are kept)">
              <LogOut size={14} /> Leave
            </button>
          </span>
        )}
      </header>
      <main className="mx-auto max-w-3xl px-4 py-8 sm:px-8">
        {loading ? (
          <Skeleton className="h-64" />
        ) : !session ? (
          <CodeEntry onIn={setSession} />
        ) : !session.consented ? (
          <Consent session={session} onDone={setSession} />
        ) : task ? (
          <Labeler task={session.kind === "pilot" && task !== "ratings" ? "ratings" : task} session={session} onBack={() => { setTask(null); void refresh(); }} />
        ) : (
          <Home session={session} onTask={setTask} onSaved={() => void refresh()} />
        )}
      </main>
    </div>
  );
}

function CodeEntry({ onIn }: { onIn: (s: StudySession) => void }) {
  const [code, setCode] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const go = async () => {
    setBusy(true);
    setError(null);
    try {
      const s = await study.me(code.trim().toUpperCase());
      setStudyCode(s.code);
      onIn(s);
    } catch {
      setError("That code is not known. Check it with the person running the study.");
    } finally {
      setBusy(false);
    }
  };
  return (
    <Card className="mx-auto max-w-md p-6">
      <h1 className="font-serif text-[22px] font-semibold">Take part</h1>
      <p className="mt-1 text-[13.5px] text-ink-3">Enter the participant code you were given. It is the only thing that identifies your answers.</p>
      <input
        value={code}
        onChange={(e) => setCode(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && code.trim() && void go()}
        placeholder="e.g. P-03"
        className="mt-5 w-full rounded-md border border-line bg-panel px-3 py-2 font-mono text-[15px] uppercase outline-none focus:border-brand/60"
        autoFocus
      />
      {error && <p className="mt-2 text-[13px] text-bad">{error}</p>}
      <Button variant="primary" className="mt-4 w-full" disabled={!code.trim()} loading={busy} onClick={go} icon={ArrowRight}>
        Continue
      </Button>
    </Card>
  );
}

function Consent({ session, onDone }: { session: StudySession; onDone: (s: StudySession) => void }) {
  const [agree, setAgree] = useState(false);
  const [busy, setBusy] = useState(false);
  return (
    <Card>
      <CardHeader title="Before you start" subtitle={`Participant ${session.code}`} />
      <div className="space-y-4 p-5 text-[14px] leading-relaxed">
        <p>
          {session.kind === "pilot"
            ? "This study looks at a timetabling assistant for faculty: answering questions about your timetable and the rules, taking requests, and writing to you when a request clashes with someone else's. You will rate some messages it wrote, then try it yourself step by step, about 25 minutes in all."
            : "Team labelling for judge validation: you will rate messages, check explanation claims against their facts, label replies, and check the assistant's answers about the rules."}
        </p>
        <ul className="list-disc space-y-1.5 pl-5 text-ink-2">
          {session.consent_text.map((t) => (
            <li key={t}>{t}</li>
          ))}
        </ul>
        <label className="flex items-start gap-2.5 rounded-md border border-line bg-panel-2 p-3">
          <input type="checkbox" checked={agree} onChange={(e) => setAgree(e.target.checked)} className="mt-1" />
          <span>I have read this and agree to take part.</span>
        </label>
        <Button
          variant="primary"
          disabled={!agree}
          loading={busy}
          onClick={async () => {
            setBusy(true);
            onDone(await study.consent());
          }}
        >
          Start
        </Button>
      </div>
    </Card>
  );
}

function Home({ session, onTask, onSaved }: { session: StudySession; onTask: (t: StudyTask) => void; onSaved: () => void }) {
  if (!session.items_ready)
    return (
      <Card className="p-6 text-[14px] text-ink-2">
        The study material is not ready yet. The coordinator builds it with <code className="font-mono">uv run python -m evaluation.study build</code>.
      </Card>
    );
  return (
    <div className="space-y-5">
      <div>
        <h1 className="font-serif text-[24px] font-semibold">{session.kind === "pilot" ? "Thank you for taking part" : "Labelling tasks"}</h1>
        <p className="mt-1 text-[13.5px] text-ink-3">
          {session.kind === "pilot" ? "Two parts: rate some messages, then try the system yourself, step by step. Either order works." : "Do them in any order; your answers are saved as you go."}
        </p>
      </div>
      {session.tasks.map((t) => {
        const p = session.progress[t];
        const Icon = TASK[t].icon;
        const done = p && p.done >= p.total;
        return (
          <Card key={t} className="p-5">
            <div className="flex items-start gap-4">
              <Icon size={18} className="mt-0.5 shrink-0 text-ink-3" />
              <div className="min-w-0 flex-1">
                <p className="text-[14.5px] font-semibold">{TASK[t].title}</p>
                <p className="mt-0.5 text-[13.5px] text-ink-3">{TASK[t].text}</p>
                {p && (
                  <div className="mt-3 flex items-center gap-3">
                    <Meter value={p.done} max={p.total || 1} tone={done ? "ok" : "brand"} className="max-w-56 flex-1" />
                    <span className="font-mono text-[12px] text-ink-3">
                      {p.done}/{p.total}
                    </span>
                  </div>
                )}
              </div>
              <Button variant={done ? "secondary" : "primary"} onClick={() => onTask(t)} disabled={done} icon={done ? Check : ArrowRight}>
                {done ? "Done" : p?.done ? "Continue" : "Start"}
              </Button>
            </div>
          </Card>
        );
      })}
      {session.kind === "pilot" && session.persona && <LiveSession session={session} onSaved={onSaved} />}
    </div>
  );
}

function LiveSession({ session, onSaved }: { session: StudySession; onSaved: () => void }) {
  const { signIn } = useAuth();
  const navigate = useNavigate();
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [survey, setSurvey] = useState(false);
  const done = new Set(session.steps_done);

  const go = async (step: LiveStep) => {
    if (step.step === "survey") return setSurvey(true);
    setBusy(step.step);
    setError(null);
    try {
      setCurrentWorld("demo"); // pilot personas and the practice clash live in the demo department
      await signIn(session.persona!);
      setStudyStep(step.step);
      if (step.step === "practice") {
        const p = await study.practice();
        setStudyTask(p.task);
      } else {
        setStudyTask(`${step.title}. ${step.text}`);
      }
      navigate("/new");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const n = session.live_steps.length;
  const finished = session.live_steps.filter((s) => done.has(s.step)).length;
  return (
    <Card className="p-5">
      <div className="flex items-start gap-4">
        <Play size={18} className="mt-0.5 shrink-0 text-ink-3" />
        <div className="min-w-0 flex-1">
          <p className="text-[14.5px] font-semibold">Try the system yourself</p>
          <p className="mt-0.5 text-[13.5px] text-ink-3">
            You use the real portal as <span className="font-medium text-ink-2">{session.persona_name}</span>, a demo faculty member, and
            write in your own words. Each step takes a minute or two; after each, a small box asks whether the system got it right.
          </p>
          <div className="mt-3 flex items-center gap-3">
            <Meter value={finished} max={n || 1} tone={finished >= n ? "ok" : "brand"} className="max-w-56 flex-1" />
            <span className="font-mono text-[12px] text-ink-3">
              {finished}/{n}
            </span>
          </div>
          <ol className="mt-4 space-y-2">
            {session.live_steps.map((s, i) => {
              const ok = done.has(s.step);
              return (
                <li key={s.step} className={clsx("flex items-start gap-3 rounded-lg border px-3.5 py-3", ok ? "border-ok/30 bg-ok/[0.04]" : "border-line")}>
                  <span className={clsx("mt-0.5 grid size-6 shrink-0 place-items-center rounded-full text-[12px] font-semibold",
                    ok ? "bg-ok/15 text-ok" : "bg-panel-2 text-ink-2")}>
                    {ok ? <Check size={13} /> : i + 1}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="text-[13.5px] font-medium">{s.title}</p>
                    <p className="mt-0.5 text-[12.5px] leading-snug text-ink-3">{s.text}</p>
                  </div>
                  <Button size="sm" variant={ok ? "secondary" : "primary"} loading={busy === s.step} onClick={() => void go(s)}>
                    {ok ? "Again" : "Start"}
                  </Button>
                </li>
              );
            })}
          </ol>
          {survey && <SurveyForm session={session} onDone={() => { setSurvey(false); onSaved(); }} />}
          {error && <p className="mt-3 text-[13px] text-bad">{error}</p>}
        </div>
      </div>
    </Card>
  );
}

/** The closing statements, each 1 (disagree) to 5 (agree), and a comment. */
function SurveyForm({ session, onDone }: { session: StudySession; onDone: () => void }) {
  const [v, setV] = useState<Record<string, number>>({});
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const complete = session.survey.every((q) => v[q.key]);
  const save = async () => {
    setBusy(true);
    setError(null);
    try {
      await study.live({ kind: "survey", step: "survey", value: v, ...(comment.trim() ? { comment: comment.trim() } : {}) });
      onDone();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="mt-4 space-y-3 rounded-lg border border-brand/30 bg-brand/[0.03] p-4">
      <p className="text-[13.5px] font-medium">How much do you agree? 1 = not at all, 5 = completely.</p>
      {session.survey.map((q) => (
        <div key={q.key} className="flex flex-wrap items-center justify-between gap-2">
          <span className="text-[13.5px] text-ink-2">{q.text}</span>
          <span className="flex gap-1">
            {[1, 2, 3, 4, 5].map((k) => (
              <button
                key={k}
                onClick={() => setV((x) => ({ ...x, [q.key]: k }))}
                className={clsx("grid size-8 place-items-center rounded-md border text-[13px] font-medium",
                  v[q.key] === k ? "border-brand bg-brand text-white" : "border-line hover:border-brand/60")}
              >
                {k}
              </button>
            ))}
          </span>
        </div>
      ))}
      <textarea
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        rows={2}
        maxLength={1000}
        placeholder="Anything else? What worked, what didn't (optional)"
        className="w-full rounded-md border border-line bg-panel px-3 py-2 text-[13.5px] outline-none focus:border-brand/60"
      />
      <Button variant="primary" disabled={!complete} loading={busy} onClick={() => void save()}>
        Save my answers
      </Button>
      {error && <p className="text-[13px] text-bad">{error}</p>}
    </div>
  );
}

const RULE_TEXT: Record<string, string> = {
  declines: "Declines: says no rule covers it, or it doesn't know",
  consistent: "Answers, and agrees with the rule",
  not_consistent: "Answers, but the rule doesn't back it (or there is no rule)",
};

function RuleItem({ item, labels, busy, onSave }: {
  item: { query: string; answer: string; rules: { id: string; title: string; text: string }[] };
  labels: string[];
  busy: boolean;
  onSave: (v: Record<string, unknown>) => void;
}) {
  return (
    <div className="space-y-5">
      <div>
        <Label>A faculty member asked</Label>
        <Quote>{item.query}</Quote>
      </div>
      <div>
        <Label>The assistant answered</Label>
        <div className="rounded-lg border border-brand/25 bg-brand/5 px-4 py-3 text-[14.5px] leading-relaxed">{item.answer}</div>
      </div>
      <div>
        <Label>The official rule</Label>
        {item.rules.length ? (
          <div className="space-y-2">
            {item.rules.map((r) => (
              <div key={r.id} className="rounded-md border border-line bg-panel-2 p-3 text-[13.5px] leading-relaxed">
                <p className="mb-1 text-[12px] font-semibold text-ink-2">
                  <span className="font-mono">{r.id}</span> · {r.title}
                </p>
                <p className="whitespace-pre-line">{r.text}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="rounded-md border border-dashed border-line px-3 py-2 text-[13.5px] text-ink-3">No rule in the handbook covers this question.</p>
        )}
      </div>
      <Card className="space-y-3 p-5">
        <p className="text-[13.5px] font-medium">What does the answer do?</p>
        <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap">
          {labels.map((l) => (
            <Button key={l} variant="secondary" disabled={busy} onClick={() => onSave({ label: l })}>
              {RULE_TEXT[l] ?? l}
            </Button>
          ))}
        </div>
      </Card>
    </div>
  );
}

function Labeler({ task, session, onBack }: { task: StudyTask; session: StudySession; onBack: () => void }) {
  const [next, setNext] = useState<NextItem | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const load = useCallback(async () => setNext(await study.next(task)), [task]);
  useEffect(() => void load(), [load]);

  const save = async (value: Record<string, unknown>) => {
    if (!next?.item) return;
    setBusy(true);
    setError(null);
    try {
      await study.label(task, next.item.id, value);
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-5">
      <button onClick={onBack} className="inline-flex items-center gap-1.5 text-[13px] text-ink-3 hover:text-ink">
        <ArrowLeft size={14} /> All tasks
      </button>
      <div className="flex items-center gap-3">
        <h1 className="font-serif text-[22px] font-semibold">{TASK[task].title}</h1>
        {next && (
          <span className="ml-auto font-mono text-[12px] text-ink-3">
            {Math.min(next.done + 1, next.total)} of {next.total}
          </span>
        )}
      </div>
      {next && <Meter value={next.done} max={next.total || 1} />}
      <p className="text-[13.5px] text-ink-3">{TASK[task].text}</p>
      {!next ? (
        <Skeleton className="h-64" />
      ) : !next.item ? (
        <Card className="p-8 text-center">
          <Check size={22} className="mx-auto text-ok" />
          <p className="mt-2 text-[15px] font-medium">All done. Thank you!</p>
          <Button className="mt-4" onClick={onBack}>
            Back to the tasks
          </Button>
        </Card>
      ) : "claim" in next.item ? (
        <ClaimItem key={next.item.id} item={next.item} busy={busy} onSave={save} />
      ) : "query" in next.item ? (
        <RuleItem key={next.item.id} item={next.item} labels={session.rule_labels} busy={busy} onSave={save} />
      ) : "reply" in next.item ? (
        <ReplyItem key={next.item.id} item={next.item} labels={session.reply_labels} busy={busy} onSave={save} />
      ) : (
        <RatingItem key={next.item.id} text={next.item.text} busy={busy} onSave={save} />
      )}
      {error && <p className="rounded-md bg-bad/10 px-3 py-2 text-sm text-bad">{error}</p>}
    </div>
  );
}

function Quote({ children }: { children: ReactNode }) {
  return <div className="whitespace-pre-line rounded-md border border-line bg-panel-2 p-4 text-[14px] leading-relaxed">{children}</div>;
}

type Fact = { id: string; text: string };

/** "Option A: the X lecture (C2-L1) on Tuesday at 3 pm in Room 101; ... The solver confirms this works." */
function parseOption(f: Fact): { key: string; placements: Placement[] } | null {
  const m = f.text.match(/^Option (\w+):\s*(.*?)\.?\s*The solver confirms this works\.?$/);
  if (!m) return null;
  const placements = m[2].split("; ").map(parsePlacement);
  return placements.every(Boolean) ? { key: m[1], placements: placements as Placement[] } : null;
}

/** "Dr. X needs ... (Tier 4, operational requirement)." -> the sentence, and the tier in plain words as a side note */
function splitTier(text: string): [string, string | null] {
  const m = text.match(/^(.*?)\s*\(Tier (\d)[^)]*\)\.?$/);
  return m ? [`${m[1]}.`, TIER[+m[2]]?.name ?? null] : [text, null];
}

function OptionCard({ option }: { option: { key: string; placements: Placement[] } }) {
  const courses = [...new Set(option.placements.map((p) => p.course))];
  return (
    <div className="rounded-md border border-line bg-panel p-3">
      <p className="text-[13px] font-semibold">Option {option.key}</p>
      {courses.map((c) => (
        <div key={c} className="mt-1.5">
          <p className="text-[12px] text-ink-3">{c}</p>
          <ul className="mt-0.5 space-y-0.5 text-[13px]">
            {option.placements
              .filter((p) => p.course === c)
              .map((p) => (
                <li key={p.session} className="flex gap-2">
                  <span className="w-12 shrink-0 font-mono text-[11.5px] leading-5 text-ink-3">{p.session}</span>
                  <span>
                    {p.day} {p.time} <span className="text-ink-3">· {p.room}</span>
                  </span>
                </li>
              ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

function ClaimItem({ item, busy, onSave }: { item: { claim: string; facts: Fact[] }; busy: boolean; onSave: (v: Record<string, unknown>) => void }) {
  const options = item.facts.map((f) => (f.id.startsWith("OPT-") ? parseOption(f) : null));
  const unparsed = item.facts.filter((f, i) => f.id.startsWith("OPT-") && !options[i]);
  const conflict = item.facts.filter((f) => !f.id.startsWith("OPT-") && !f.id.startsWith("LED-"));
  const history = item.facts.filter((f) => f.id.startsWith("LED-"));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (busy || e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "s" || e.key === "S") onSave({ supported: true });
      if (e.key === "n" || e.key === "N") onSave({ supported: false });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, onSave]);

  return (
    <Card>
      <div className="space-y-5 p-5">
        <div>
          <Label>The claim</Label>
          <Quote>{item.claim}</Quote>
        </div>

        <div className="space-y-4 rounded-md border border-line p-4">
          <p className="text-[12.5px] font-medium text-ink-2">Check it against these facts</p>
          <div>
            <Label>The conflict</Label>
            <ul className="space-y-1.5 text-[13.5px]">
              {conflict.map((f) => {
                const [text, tier] = splitTier(f.text);
                return (
                  <li key={f.id} className="flex gap-2">
                    <span className="mt-2 size-1.5 shrink-0 rounded-full bg-ink-3" />
                    <span>
                      {text}
                      {tier && <span className="ml-1.5 text-[12px] text-ink-3">{tier}</span>}
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>
          {options.some(Boolean) && (
            <div>
              <Label>Options offered (each checked by the solver)</Label>
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {options.map((o) => o && <OptionCard key={o.key} option={o} />)}
              </div>
            </div>
          )}
          {unparsed.length > 0 && (
            <ul className="space-y-1 text-[13px] text-ink-2">
              {unparsed.map((f) => (
                <li key={f.id}>{f.text}</li>
              ))}
            </ul>
          )}
          {history.length > 0 && (
            <div>
              <Label>History</Label>
              <ul className="space-y-1 text-[13.5px]">
                {history.map((f) => (
                  <li key={f.id}>{f.text}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <p className="text-[12.5px] text-ink-3">
          <span className="font-medium text-ink-2">Supported</span> means every day, time, room, name and number in the claim appears in these facts, and nothing in it contradicts them. A greeting or a
          sentence with no such detail counts as supported unless it says something false.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="success" icon={Check} disabled={busy} onClick={() => onSave({ supported: true })}>
            Supported
          </Button>
          <Button variant="danger" icon={X} disabled={busy} onClick={() => onSave({ supported: false })}>
            Not supported
          </Button>
          <span className="ml-auto hidden text-[12px] text-ink-3 sm:block">
            Keys: <kbd className="font-mono">S</kbd> supported · <kbd className="font-mono">N</kbd> not supported
          </span>
        </div>
      </div>
    </Card>
  );
}

const REPLY_TEXT: Record<string, string> = {
  counter: "Suggests other days or times",
  reject: "Declines, without another time",
  other: "Something else (a question, unclear)",
};

function ReplyItem({ item, labels, busy, onSave }: { item: { message: string; reply: string }; labels: string[]; busy: boolean; onSave: (v: Record<string, unknown>) => void }) {
  const letters = new Set([...item.message.matchAll(/^([A-Z])\) /gm)].map((m) => m[1]));
  const shown = labels.filter((l) => !l.startsWith("accept:") || letters.has(l.slice(7)));
  return (
    <div className="space-y-5">
      <div>
        <Label>The office's message</Label>
        <LetterView text={item.message} />
      </div>
      <Card className="space-y-4 p-5">
        <div>
          <Label>The faculty member's reply</Label>
          <div className="ml-4 rounded-lg rounded-tl-none border border-brand/25 bg-brand/5 px-4 py-3 text-[14.5px] leading-relaxed">“{item.reply}”</div>
        </div>
        <p className="text-[13.5px] font-medium">What does this reply do?</p>
        <div className="flex flex-wrap gap-2">
          {shown.map((l) => (
            <Button key={l} variant={l.startsWith("accept:") ? "primary" : "secondary"} disabled={busy} onClick={() => onSave({ label: l })}>
              {l.startsWith("accept:") ? `Accepts option ${l.slice(7)}` : REPLY_TEXT[l] ?? l}
            </Button>
          ))}
        </div>
      </Card>
    </div>
  );
}

function RatingItem({ text, busy, onSave }: { text: string; busy: boolean; onSave: (v: Record<string, unknown>) => void }) {
  return (
    <div className="space-y-5">
      <div>
        <Label>Imagine you received this message</Label>
        <LetterView text={text} />
      </div>
      <Card className="p-5">
        <RatingForm busy={busy} onSave={onSave} />
      </Card>
    </div>
  );
}
