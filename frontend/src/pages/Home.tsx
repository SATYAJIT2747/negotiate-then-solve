import { useEffect, useMemo, useState } from "react";
import { ConversationPane, useConversation } from "../components/ConversationPane";
import { ConversationSwitcher, groupOf, useConversations } from "../components/Conversations";
import { TimetableGrid } from "../components/TimetableGrid";
import { WeekStepper } from "../components/WeekStepper";
import { Card, Skeleton } from "../components/ui";
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

  const { convs, refresh } = useConversations();
  const s = useConversation(convs, refresh);
  const { conv, shown } = s;

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

  const dayName = (d: string) => inst?.day_names[d] ?? d;
  const label = (slot: number) => inst?.slots.find((s) => s.slot === slot)?.label ?? `slot ${slot}`;
  const phrase = (e: TimetableEntry) => {
    const whose = teaching && e.faculty === user?.id ? "my" : !teaching && group && e.groups.includes(group) ? "the" : `${e.faculty_name}'s`;
    return `${whose} ${e.title} ${e.kind} on ${dayName(e.day)} at ${label(e.slot)}`;
  };
  /** A click attaches that one class (or takes it off again); Ctrl or Shift + click adds it to the others. */
  const toggle = (e: TimetableEntry, ev?: { ctrlKey: boolean; metaKey: boolean; shiftKey: boolean }) => {
    if (!s.answering) s.startNew();
    if (!s.answering && s.replying) return; // options are waiting: a class click would start a new request
    setPicks((p) => {
      const has = p.some((x) => x.session === e.session);
      if (ev && (ev.ctrlKey || ev.metaKey || ev.shiftKey)) return has ? p.filter((x) => x.session !== e.session) : [...p, e];
      return has && p.length === 1 ? [] : [e];
    });
  };
  /** "a", "a and b", "a, b and c" */
  const listed = (xs: string[]) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);

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
          <ConversationSwitcher items={convs} selected={s.selected} onSelect={s.open} />
        </div>
        <ConversationPane
          s={s}
          convs={convs}
          attachments={{
            items: picks.map((e) => ({ key: e.session, label: `${e.title} ${e.kind} · ${dayName(e.day)} ${label(e.slot)}` })),
            remove: (key) => setPicks((p) => p.filter((x) => x.session !== key)),
            compose: (t) => (picks.length ? `${t.replace(/[.!?]$/, "")} (${listed(picks.map(phrase))}).` : t),
            clear: () => setPicks([]),
          }}
        />
      </Card>
    </div>
  );
}
