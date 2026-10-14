import { Inbox, Send, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { groupOf, RequestSwitcher } from "../components/RequestSwitcher";
import { RequestThread } from "../components/RequestThread";
import { TimetableGrid } from "../components/TimetableGrid";
import { WeekStepper } from "../components/WeekStepper";
import { Button, Card, Skeleton } from "../components/ui";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useApi } from "../lib/hooks";
import type { TimetableEntry } from "../lib/types";

/** Faculty and class representatives: their week's timetable first, their requests as conversations
 * beside it, and one box to ask or request something. */
export default function Home() {
  const { user } = useAuth();
  const teaching = ["faculty", "guest_faculty", "hod"].includes(user?.role ?? "");
  const group = user?.role === "student" && user.id.startsWith("ST-") ? user.id.slice(3) : null;

  const { data: inst } = useApi(() => api.instance(), []);
  const { data: cal } = useApi(() => api.calendar(), []);
  const { data: ov } = useApi(() => api.overview(), [], 4000);
  const [week, setWeek] = useState<number | null>(null);
  useEffect(() => {
    if (cal && week === null) setWeek(cal.current_week);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cal]);
  const { data: tt } = useApi(() => (week === null ? Promise.resolve(null) : api.timetable({ week })), [week], 5000);

  const { data: cases, refresh: refreshCases } = useApi(() => api.cases("mine"), [], 3000);
  const [selected, setSelected] = useState<string | null>(null);
  const [picked, setPicked] = useState(false); // the person chose a thread (or "new"), so don't pick one for them
  useEffect(() => {
    if (picked || selected || !cases?.length) return;
    setSelected((cases.find((c) => groupOf(c) !== "done") ?? cases[0]).id); // what needs them, else the latest
  }, [cases, picked, selected]);
  const { data: detail, refresh: refreshDetail } = useApi(
    () => (selected ? api.case(selected) : Promise.resolve(null)),
    [selected],
    selected ? 1500 : undefined,
  );
  const shown = detail && detail.id === selected ? detail : null;

  const mine = useMemo(
    () => (tt?.entries ?? []).filter((e) => (teaching ? e.faculty === user?.id : group ? e.groups.includes(group) : false)),
    [tt, teaching, group, user],
  );
  // classes clicked on the grid: shown as chips in the composer and named in the message when it is sent
  const [picks, setPicks] = useState<TimetableEntry[]>([]);
  // the classes picked, else those the open request touches, else what moved in the latest change
  const outlined = useMemo(() => {
    if (picks.length) return new Set(picks.map((e) => e.session));
    const own = new Set<string>([...(shown?.proposal?.diff ?? []).map((d) => d.session), ...(shown?.extra?.session ? [shown.extra.session] : [])]);
    return own.size ? own : new Set(tt?.changed ?? []);
  }, [picks, shown, tt]);
  // each day's date in the week shown, for the grid's headings
  const dates = useMemo(() => {
    const w = week ? cal?.weeks[week - 1] : null;
    if (!w || !cal) return undefined;
    const today = new Date().toDateString();
    return Object.fromEntries(
      cal.days.map((d, i) => {
        const at = new Date(`${w.monday}T00:00:00`);
        at.setDate(at.getDate() + i);
        return [d, { label: at.toLocaleDateString(undefined, { day: "numeric", month: "short" }), today: at.toDateString() === today }];
      }),
    );
  }, [cal, week]);
  const cancelled = (tt?.cancelled ?? []).filter((c) => mine.some((e) => e.session === c.session) || (group && c.groups.includes(group)));

  // the composer answers the open thread's question, or starts a new request
  const [text, setText] = useState("");
  const [fresh, setFresh] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const box = useRef<HTMLTextAreaElement>(null);
  const answering = !fresh && shown?.status === "clarification_requested" && shown.sender === user?.id;
  useEffect(() => setFresh(false), [selected]);

  const dayName = (d: string) => inst?.day_names[d] ?? d;
  const label = (slot: number) => inst?.slots.find((s) => s.slot === slot)?.label ?? `slot ${slot}`;
  const phrase = (e: TimetableEntry) => `${teaching ? "my" : "the"} ${e.title} ${e.kind} on ${dayName(e.day)} at ${label(e.slot)}`;
  /** A click attaches that one class (or takes it off again); Ctrl or Shift + click adds it to the others. */
  const toggle = (e: TimetableEntry, ev?: { ctrlKey: boolean; metaKey: boolean; shiftKey: boolean }) =>
    setPicks((p) => {
      const has = p.some((x) => x.session === e.session);
      if (ev && (ev.ctrlKey || ev.metaKey || ev.shiftKey)) return has ? p.filter((x) => x.session !== e.session) : [...p, e];
      return has && p.length === 1 ? [] : [e];
    });
  /** "a", "a and b", "a, b and c" */
  const listed = (xs: string[]) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);

  const send = async () => {
    const typed = text.trim();
    if (!typed) return;
    const t = picks.length ? `${typed.replace(/[.!?]$/, "")} (${listed(picks.map(phrase))}).` : typed;
    setSending(true);
    setError(null);
    try {
      if (answering && shown) {
        await api.clarify(shown.id, t);
        void refreshDetail();
      } else {
        const { id } = await api.submit(t);
        setPicked(true);
        setSelected(id);
      }
      setText("");
      setPicks([]);
      void refreshCases();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  };


  return (
    <div className="grid gap-4 lg:h-full lg:grid-cols-[minmax(0,1fr)_minmax(300px,27%)]">
      <div className="flex min-h-0 min-w-0 flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h1 className="font-serif text-[22px] font-semibold">{week ? `Week ${week}` : "Timetable"}</h1>
          {cal && <WeekStepper cal={cal} week={week} onChange={setWeek} compact />}
        </div>
        {cancelled.length > 0 && (
          <p className="rounded-lg border border-bad/40 bg-bad/5 px-4 py-2.5 text-[13px]">
            Not held this week: {cancelled.map((c) => c.session_name.replace(/^the /, "")).join(" · ")}
          </p>
        )}
        <Card className="h-[560px] min-h-0 p-3 lg:h-auto lg:flex-1">
          {!inst || !tt ? (
            <Skeleton className="h-full" />
          ) : !mine.length ? (
            <p className="px-3 py-16 text-center text-[13.5px] text-ink-3">No classes in the published timetable for this week.</p>
          ) : (
            <TimetableGrid instance={inst} entries={mine} changed={outlined} onPick={toggle} fill dates={dates} />
          )}
        </Card>
        <p className="text-[12px] text-ink-3">Click a class to attach it to a request (Ctrl or Shift + click for more than one). Outlined: the classes attached, or the ones the open request changes.</p>
      </div>

      <Card className="flex h-[560px] min-h-0 flex-col lg:h-full">
        <div className="flex items-center gap-2 border-b border-line px-2 py-1.5">
          <RequestSwitcher
            cases={cases ?? []}
            selected={selected}
            onSelect={(id) => {
              setPicked(true);
              setSelected(id);
            }}
          />
          <Link to="/inbox" title="Inbox" className="relative grid size-8 shrink-0 place-items-center rounded-md text-ink-3 hover:bg-panel-2 hover:text-ink">
            <Inbox size={16} />
            {(ov?.my_inbox ?? 0) > 0 && (
              <span className="absolute -right-0.5 -top-0.5 rounded-full bg-brand px-1 text-[10px] font-semibold leading-4 text-panel">{ov!.my_inbox}</span>
            )}
          </Link>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {selected && !shown ? (
            <Skeleton className="h-40" />
          ) : shown ? (
            <RequestThread
              c={shown}
              onChanged={() => {
                void refreshDetail();
                void refreshCases();
              }}
            />
          ) : (
            <div className="grid h-full place-items-center text-center">
              <div className="max-w-[260px] space-y-2">
                <p className="text-[14px] font-medium">Ask or request something</p>
                <p className="text-[12.5px] text-ink-3">
                  Write it the way you would write to the timetable office: an absence, a class to move, or a question about the rules.
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="border-t border-line p-3">
          {answering && (
            <p className="mb-2 flex items-center gap-2 text-[12px] text-warn">
              <span className="min-w-0 flex-1 truncate">Answering: {shown?.reply}</span>
              <button onClick={() => setFresh(true)} className="flex shrink-0 items-center gap-1 text-ink-3 hover:text-ink" title="Write a new request instead">
                <X size={12} /> New request instead
              </button>
            </p>
          )}
          {picks.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1">
              {picks.map((e) => (
                <span key={e.session} className="flex items-center gap-1 rounded border border-brand/40 bg-brand/10 py-0.5 pl-1.5 pr-1 text-[11.5px]">
                  {e.title} {e.kind} · {dayName(e.day)} {label(e.slot)}
                  <button onClick={() => toggle(e)} className="text-ink-3 hover:text-ink" aria-label={`Remove ${e.title}`}>
                    <X size={11} />
                  </button>
                </span>
              ))}
            </div>
          )}
          <div className="flex items-end gap-2 rounded-lg border border-line bg-panel px-3 py-2 focus-within:border-brand/60">
            <textarea
              ref={box}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                }
              }}
              rows={2}
              placeholder={answering ? "Your answer…" : picks.length ? "What should happen to these classes?" : "Ask or request something…"}
              className="max-h-40 flex-1 resize-none bg-transparent text-[13.5px] leading-relaxed outline-none placeholder:text-ink-3"
            />
            <Button variant="primary" size="sm" icon={Send} loading={sending} disabled={!text.trim()} onClick={send} aria-label="Send" />
          </div>
          <p className="mt-1.5 text-[11px] text-ink-3">Enter to send · Shift+Enter for a new line</p>
          {error && <p className="mt-1 text-[12.5px] text-bad">{error}</p>}
        </div>
      </Card>
    </div>
  );
}
