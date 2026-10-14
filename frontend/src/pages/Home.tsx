import { ArrowLeft, Send, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { buildConversations, ColleagueThread, ConversationSwitcher, groupOf } from "../components/Conversations";
import { RequestThread } from "../components/RequestThread";
import { TimetableGrid } from "../components/TimetableGrid";
import { WeekStepper } from "../components/WeekStepper";
import { Button, Card, Skeleton } from "../components/ui";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useApi } from "../lib/hooks";
import type { TimetableEntry } from "../lib/types";

/** Faculty and class representatives: their week's timetable first; beside it, their requests and what
 * colleagues' requests asked of them, as conversations (this replaces the Inbox for them); and one box to
 * ask or request something. */
export default function Home() {
  const { user } = useAuth();
  const teaching = ["faculty", "guest_faculty", "hod"].includes(user?.role ?? "");
  const group = user?.role === "student" && user.id.startsWith("ST-") ? user.id.slice(3) : null;

  const { data: inst } = useApi(() => api.instance(), []);
  const { data: cal } = useApi(() => api.calendar(), []);
  const [week, setWeek] = useState<number | null>(null); // null until the calendar is in
  const [scope, setScope] = useState<"mine" | "all">("mine"); // this person's classes, or every teacher's
  useEffect(() => {
    if (cal && week === null) setWeek(cal.current_week);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cal]);
  const { data: tt } = useApi(() => (week === null ? Promise.resolve(null) : api.timetable({ week })), [week], 5000);
  const shownWeek = week;

  const { data: cases, refresh: refreshCases } = useApi(() => api.cases("mine"), [], 3000);
  const { data: feed, refresh: refreshFeed } = useApi(() => api.feed(), [], 3000);
  const convs = useMemo(() => buildConversations(cases ?? [], feed ?? []), [cases, feed]);
  const [selected, setSelected] = useState<string | null>(null); // a conversation: a case id, or "other:…"
  const [picked, setPicked] = useState(false); // the person chose a thread (or "new"), so don't pick one for them
  useEffect(() => {
    if (picked || selected || !convs.length) return;
    const first = convs.find((c) => groupOf(c) === "needs") ?? convs.find((c) => groupOf(c) === "progress") ?? convs.find((c) => c.own);
    if (first) setSelected(first.key); // what needs them, else what is under way, else their latest request
  }, [convs, picked, selected]);
  const conv = convs.find((c) => c.key === selected) ?? null;
  const caseId = selected && !selected.startsWith("other:") ? selected : null;
  const { data: detail, refresh: refreshDetail } = useApi(
    () => (caseId ? api.case(caseId) : Promise.resolve(null)),
    [caseId],
    caseId ? 1500 : undefined,
  );
  const shown = detail && detail.id === caseId ? detail : null;
  // opening a conversation reads it (things waiting on an answer stay unread until answered)
  useEffect(() => {
    const unread = (conv?.feed ?? []).filter((f) => f.unread && !f.needs_you);
    if (unread.length) void api.feedSeen(Object.fromEntries(unread.map((f) => [f.id, f.n]))).then(refreshFeed);
  }, [conv?.key, conv?.feed.map((f) => `${f.id}:${f.n}`).join()]); // eslint-disable-line react-hooks/exhaustive-deps
  const changed = () => {
    void refreshDetail();
    void refreshCases();
    void refreshFeed();
  };

  const mine = useMemo(
    () => (tt?.entries ?? []).filter((e) => (teaching ? e.faculty === user?.id : group ? e.groups.includes(group) : false)),
    [tt, teaching, group, user],
  );
  // classes clicked on the grid: shown as chips in the composer and named in the message when it is sent
  const [picks, setPicks] = useState<TimetableEntry[]>([]);
  // what the open request is changing, while it is still under way (once published the class is simply
  // in its new place; declined or withdrawn, nothing changes)
  const pending = !!shown && !!conv && (groupOf(conv) === "progress" || groupOf(conv) === "needs") && (shown.proposal?.week == null || shown.proposal.week === shownWeek);
  const diff = pending ? (shown?.proposal?.diff ?? []) : [];
  // outlined: the classes attached to the message being written, else those the open request moves
  const outlined = useMemo(
    () => new Set(picks.length ? picks.map((e) => e.session) : diff.filter((d) => d.before).map((d) => d.session)),
    [picks, diff],
  );
  const ghosts = diff
    .filter((d) => d.after && (d.after.day !== d.before?.day || d.after.slot !== d.before?.slot))
    .map((d) => ({
      session: d.session,
      day: d.after!.day,
      slot: d.after!.slot,
      title: d.session_name.replace(/^(the|my) /, "").replace(/\s*\([A-Z]-[\w-]+\)$/, ""), // "Data Structures lecture"
      note: shown?.status === "awaiting_approval" ? "moving here · waiting for approval" : "proposed",
    }));
  // each day's date in the week shown, for the grid's headings
  const dates = useMemo(() => {
    const w = shownWeek ? cal?.weeks[shownWeek - 1] : null;
    if (!w || !cal) return undefined;
    const today = new Date().toDateString();
    return Object.fromEntries(
      cal.days.map((d, i) => {
        const at = new Date(`${w.monday}T00:00:00`);
        at.setDate(at.getDate() + i);
        return [d, { label: at.toLocaleDateString(undefined, { day: "numeric", month: "short" }), today: at.toDateString() === today }];
      }),
    );
  }, [cal, shownWeek]);
  const cancelled = (tt?.cancelled ?? []).filter((c) => mine.some((e) => e.session === c.session) || (group && c.groups.includes(group)));

  // The composer belongs to what the panel shows: it answers the open request's question, and otherwise
  // writes a new request, so starting to write (or attaching a class) swaps the open thread for a blank
  // new request, with a way back to it.
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [back, setBack] = useState<string | null>(null); // the thread to go back to from a new request
  const box = useRef<HTMLTextAreaElement>(null);
  const answering = shown?.status === "clarification_requested" && shown.sender === user?.id;
  const startNew = () => {
    if (!selected) return;
    setBack(selected);
    setPicked(true);
    setSelected(null);
  };

  const dayName = (d: string) => inst?.day_names[d] ?? d;
  const label = (slot: number) => inst?.slots.find((s) => s.slot === slot)?.label ?? `slot ${slot}`;
  const phrase = (e: TimetableEntry) => {
    const whose = teaching && e.faculty === user?.id ? "my" : !teaching && group && e.groups.includes(group) ? "the" : `${e.faculty_name}'s`;
    return `${whose} ${e.title} ${e.kind} on ${dayName(e.day)} at ${label(e.slot)}`;
  };
  /** A click attaches that one class (or takes it off again); Ctrl or Shift + click adds it to the others. */
  const toggle = (e: TimetableEntry, ev?: { ctrlKey: boolean; metaKey: boolean; shiftKey: boolean }) => {
    if (!answering) startNew();
    setPicks((p) => {
      const has = p.some((x) => x.session === e.session);
      if (ev && (ev.ctrlKey || ev.metaKey || ev.shiftKey)) return has ? p.filter((x) => x.session !== e.session) : [...p, e];
      return has && p.length === 1 ? [] : [e];
    });
  };
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
        setBack(null);
        setSelected(id);
      }
      setText("");
      setPicks([]);
      void refreshCases();
      void refreshFeed();
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
          {cal ? (
            <WeekStepper cal={cal} week={shownWeek} onChange={(w) => w !== null && setWeek(w)} compact />
          ) : (
            <Skeleton className="h-8 w-80" />
          )}
          <select
            value={scope}
            onChange={(e) => setScope(e.target.value as "mine" | "all")}
            title="My timetable: your own classes. All classes: every teacher's classes in the department, for the same week."
            className="h-8 w-40 rounded-md border border-line bg-panel px-2.5 text-[13px] outline-none hover:bg-panel-2 focus:border-brand/60"
          >
            <option value="mine">My timetable</option>
            <option value="all">All classes</option>
          </select>
        </div>
        {cancelled.length > 0 && (
          <p className="rounded-lg border border-bad/40 bg-bad/5 px-4 py-2.5 text-[13px]">
            Not held this week: {cancelled.map((c) => c.session_name.replace(/^the /, "")).join(" · ")}
          </p>
        )}
        <Card className="h-[560px] min-h-0 p-3 lg:h-auto lg:flex-1">
          {!inst || !tt ? (
            <Skeleton className="h-full" />
          ) : scope === "all" ? (
            <div className="h-full overflow-auto">
              <TimetableGrid instance={inst} entries={tt.entries} changed={outlined} onPick={toggle} dates={dates} ghosts={ghosts} />
            </div>
          ) : !mine.length ? (
            <p className="px-3 py-16 text-center text-[13.5px] text-ink-3">No classes in the published timetable for this week.</p>
          ) : (
            <TimetableGrid instance={inst} entries={mine} changed={outlined} onPick={toggle} fill dates={dates} ghosts={ghosts} />
          )}
        </Card>
        <p className="text-[12px] text-ink-3">Click a class to attach it to a request (Ctrl or Shift + click for more than one). Outlined: the classes attached, or the one the open request is moving; dashed: where it would go.</p>
      </div>

      <Card className="flex h-[560px] min-h-0 flex-col lg:h-full">
        <div className="flex items-center gap-2 border-b border-line px-2 py-1.5">
          <ConversationSwitcher
            items={convs}
            selected={selected}
            onSelect={(key) => {
              setPicked(true);
              setBack(null);
              setSelected(key);
            }}
          />
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {conv && !conv.own ? (
            <ColleagueThread c={conv} onChanged={changed} />
          ) : caseId && !shown ? (
            <Skeleton className="h-40" />
          ) : shown ? (
            <RequestThread c={shown} onChanged={changed} />
          ) : (
            <div className="grid h-full place-items-center text-center">
              <div className="max-w-[260px] space-y-2">
                {back && (
                  <button
                    onClick={() => {
                      setSelected(back);
                      setBack(null);
                    }}
                    className="mx-auto mb-3 flex max-w-full items-center gap-1 text-[12px] text-ink-3 hover:text-brand"
                  >
                    <ArrowLeft size={12} className="shrink-0" />
                    <span className="truncate">Back to: {convs.find((c) => c.key === back)?.title ?? back}</span>
                  </button>
                )}
                <p className="text-[14px] font-medium">New request</p>
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
              <button onClick={startNew} className="flex shrink-0 items-center gap-1 text-ink-3 hover:text-ink" title="Write a new request instead">
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
              onFocus={() => !answering && startNew()}
              rows={2}
              placeholder={
                answering ? "Your answer…" : picks.length ? "What should happen to these classes?" : selected ? "Start a new request…" : "Ask or request something…"
              }
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
