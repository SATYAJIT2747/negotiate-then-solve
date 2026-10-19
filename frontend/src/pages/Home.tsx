import { Info } from "lucide-react";
import { useEffect, useMemo, useState, type MouseEvent } from "react";
import { ActionCard, ClassMenu, intent, sentence, type ActionKind, type Draft } from "../components/ClassActions";
import { ConversationPane, useConversation } from "../components/ConversationPane";
import { ConversationSwitcher, useConversations } from "../components/Conversations";
import { DetailsDrawer } from "../components/DetailsDrawer";
import { TimetableGrid } from "../components/TimetableGrid";
import { WeekStepper } from "../components/WeekStepper";
import { Card, Skeleton } from "../components/ui";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useApi } from "../lib/hooks";
import type { TimetableEntry } from "../lib/types";

/** The request workspace, the same for every role: the week's timetable first (click a class to cancel, move
 * or swap it, with the free slots shown, or to name it in a request); beside it, the person's requests and
 * what colleagues' requests asked of them, as conversations; and with none open, the full new-request
 * composer (examples, the week and days on the timetable, Undo). Faculty and class representatives have it
 * as their home (it replaces the Inbox for them); everyone else opens it as New request. */
export default function Home() {
  const { user } = useAuth();
  const teaching = ["faculty", "guest_faculty", "hod"].includes(user?.role ?? "");
  const group = user?.role === "student" && user.id.startsWith("ST-") ? user.id.slice(3) : null;

  const { data: inst } = useApi(() => api.instance(), []);
  const { data: cal } = useApi(() => api.calendar(), []);
  const [week, setWeek] = useState<number | null>(null); // null until the calendar is in
  // this person's classes, or every teacher's: someone with no classes of their own (the office, the lab
  // in-charge, the exam cell) starts on every class
  const [scope, setScope] = useState<"mine" | "all">(teaching || group ? "mine" : "all");
  useEffect(() => {
    if (!cal || week !== null) return;
    // at the weekend this week's classes are all over, so open on the next one (nothing can move into the past)
    const w = cal.weeks[cal.current_week - 1];
    const friday = w ? new Date(`${w.monday}T00:00:00`) : null;
    friday?.setDate(friday.getDate() + cal.days.length - 1);
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    setWeek(friday && friday < today && cal.current_week < cal.weeks.length ? cal.current_week + 1 : cal.current_week);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cal]);
  const { data: tt } = useApi(() => (week === null ? Promise.resolve(null) : api.timetable({ week })), [week], 5000);
  const shownWeek = week;

  const { convs, refresh } = useConversations();
  const s = useConversation(convs, refresh, { autoOpen: false }); // opens on a new request
  const [details, setDetails] = useState<string | null>(null); // the request whose details are open

  const mine = useMemo(
    () => (tt?.entries ?? []).filter((e) => (teaching ? e.faculty === user?.id : group ? e.groups.includes(group) : false)),
    [tt, teaching, group, user],
  );
  // classes clicked on the grid: shown as chips in the composer and named in the message when it is sent
  const [picks, setPicks] = useState<TimetableEntry[]>([]);
  // outlined: the classes attached to the message being written, else those the open request moves
  // a request being put together by clicks (cancel, move or swap a class), and the menu that starts one
  const [menu, setMenu] = useState<{ x: number; y: number; cls: TimetableEntry } | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [sending, setSending] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const outlined = useMemo(
    () =>
      new Set(
        draft
          ? [draft.cls.session, ...(draft.theirs ? [draft.theirs.session] : [])]
          : picks.map((e) => e.session),
      ),
    [draft, picks],
  );
  const target = draft?.kind === "move" && draft.to?.slot != null && draft.to.day ? { day: draft.to.day, slot: draft.to.slot } : null;
  // while a move is being put together, where it would go (a request's change is shown in its conversation)
  const ghosts = target && draft ? [{ session: draft.cls.session, day: target.day, slot: target.slot, title: `${draft.cls.title} ${draft.cls.kind}`, note: "new time · not sent yet" }] : [];

  // Where a class may go, and which classes it may be swapped with, judged on this week's published
  // timetable: its teacher and its sections must be free, and the lunch hour stays free. Only a hint;
  // the solver checks the rest (rooms, pending changes).
  const all = useMemo(() => tt?.entries ?? [], [tt]);
  const cellsOf = (day: string, slot: number, duration: number) => Array.from({ length: duration }, (_, i) => `${day}:${slot + i}`);
  const busy = (who: (o: TimetableEntry) => boolean, except: string[]) =>
    new Set(all.filter((o) => who(o) && !except.includes(o.session)).flatMap((o) => cellsOf(o.day, o.slot, o.duration)));
  const freeFor = (e: TimetableEntry, cells: string[], except: string[]) => {
    const taken = busy((o) => o.faculty === e.faculty || o.groups.some((g) => e.groups.includes(g)), except);
    return !cells.some((c) => taken.has(c));
  };
  // a day of the shown week that is already over: nothing can be moved there any more
  const dayPassed = (day: string) => {
    const w = week && cal ? cal.weeks[week - 1] : null;
    if (!w || !cal) return false;
    const at = new Date(`${w.monday}T00:00:00`);
    at.setDate(at.getDate() + cal.days.indexOf(day));
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    return at < today;
  };
  const weekOver = !!cal && cal.days.every(dayPassed);
  const slotState = (day: string, slot: number): "free" | "blocked" | null => {
    if (draft?.kind !== "move" || !inst) return null;
    const e = draft.cls;
    if (day === e.day && slot === e.slot) return null; // where it is now
    if (dayPassed(day)) return "blocked";
    const lunch = inst.calendar.lunch_slot;
    const cells = cellsOf(day, slot, e.duration);
    if (slot + e.duration > inst.calendar.slots_per_day || (lunch != null && slot <= lunch && lunch < slot + e.duration)) return "blocked";
    return freeFor(e, cells, [e.session]) ? "free" : "blocked";
  };
  const swappable = useMemo(() => {
    if (draft?.kind !== "swap") return null;
    const e = draft.cls;
    return new Set(
      all
        .filter(
          (o) =>
            o.session !== e.session &&
            o.faculty !== e.faculty &&
            o.duration === e.duration &&
            freeFor(e, cellsOf(o.day, o.slot, o.duration), [e.session, o.session]) &&
            freeFor(o, cellsOf(e.day, e.slot, e.duration), [e.session, o.session]),
        )
        .map((o) => o.session),
    );
  }, [draft, all]); // eslint-disable-line react-hooks/exhaustive-deps
  const dim = swappable ? new Set(all.filter((o) => o.session !== draft!.cls.session && !swappable.has(o.session)).map((o) => o.session)) : undefined;
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
  // clicking a class: in a swap it is the colleague's class; with Ctrl or Shift it is attached; otherwise the menu
  const pickClass = (e: TimetableEntry, ev: MouseEvent) => {
    if (draft && e.session === draft.cls.session) return stopDraft(); // the class again: leave the action
    if (draft?.kind === "swap") {
      if (swappable?.has(e.session)) setDraft({ ...draft, theirs: e });
      return;
    }
    if (draft) return; // choosing a time: classes are not targets
    if (ev.ctrlKey || ev.metaKey || ev.shiftKey) return toggle(e, ev);
    setMenu({ x: ev.clientX, y: ev.clientY, cls: e });
  };
  /** Leave a cancel, move or swap being put together: the Cancel button, Esc, or clicking its class again. */
  const stopDraft = () => {
    if (draft?.kind === "swap") setScope("mine");
    setDraft(null);
    setActionError(null);
  };
  useEffect(() => {
    if (!draft) return;
    const esc = (ev: KeyboardEvent) => ev.key === "Escape" && stopDraft();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [draft]); // eslint-disable-line react-hooks/exhaustive-deps
  const startAction = (kind: ActionKind | "write") => {
    const cls = menu!.cls;
    setMenu(null);
    if (kind === "write") return toggle(cls);
    setPicks([]);
    setActionError(null);
    const passed = !!cal && week !== null && week < cal.current_week; // a past week: pick the weeks instead
    setDraft({ kind, cls, to: null, theirs: null, when: { mode: passed ? "pick" : "this", weeks: [] }, hard: true, reason: "" });
    if (kind === "swap") setScope("all"); // a colleague's class is picked among everyone's
  };
  const sendAction = async () => {
    if (!draft || !inst || week === null) return;
    setSending(true);
    setActionError(null);
    try {
      const { id } = await api.submitForm(sentence(draft, inst, week), intent(draft, week));
      setDraft(null);
      if (draft.kind === "swap") setScope("mine");
      s.open(id);
      s.changed();
    } catch (e) {
      setActionError((e as Error).message);
    } finally {
      setSending(false);
    }
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
            {(teaching || group) && <option value="mine">My timetable</option>}
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
              <TimetableGrid instance={inst} entries={tt.entries} changed={outlined} onPick={pickClass} dates={dates} ghosts={ghosts} slotState={slotState} onSlot={(day, slot) => draft && setDraft({ ...draft, to: { day, slot } })} dim={dim} />
            </div>
          ) : !mine.length ? (
            <p className="px-3 py-16 text-center text-[13.5px] text-ink-3">No classes in the published timetable for this week.</p>
          ) : (
            <TimetableGrid instance={inst} entries={mine} changed={outlined} onPick={pickClass} fill dates={dates} ghosts={ghosts} slotState={slotState} onSlot={(day, slot) => draft && setDraft({ ...draft, to: { day, slot } })} />
          )}
        </Card>
        <p className="flex flex-wrap items-center gap-x-2 text-[12px] text-ink-3">
          {draft?.kind === "move"
            ? weekOver
              ? `Week ${week} is over, so there is no free slot left in it: go to a later week to pick the new time.`
              : "Click a green slot for the new time."
            : draft?.kind === "swap"
              ? "Click the colleague's class to swap with. Classes that cannot be swapped are faded."
              : draft
                ? "Fill in the details on the right."
                : "Click a class to cancel, reschedule or swap it."}
          {draft && (
            <>
              <span>Esc or click the class again to stop.</span>
              <button onClick={stopDraft} className="font-medium text-brand hover:underline">
                Cancel {draft.kind === "move" ? "rescheduling" : draft.kind === "swap" ? "the swap" : "this"}
              </button>
            </>
          )}
        </p>
      </div>

      <Card className="flex h-[560px] min-h-0 flex-col lg:h-full">
        <div className="flex items-center gap-2 border-b border-line px-2 py-1.5">
          <ConversationSwitcher items={convs} selected={s.selected} onSelect={s.open} />
          {s.caseId && (
            <button
              onClick={() => setDetails(s.caseId)}
              title="Details: how the system handled this request"
              aria-label="Details"
              className="grid size-8 shrink-0 place-items-center rounded-md text-ink-3 hover:bg-panel-2 hover:text-ink"
            >
              <Info size={16} />
            </button>
          )}
        </div>
        {draft && inst && week !== null ? (
          <ActionCard
            d={draft}
            set={setDraft}
            inst={inst}
            week={week}
            weeks={inst.calendar.weeks}
            current={cal?.current_week ?? 1}
            sending={sending}
            error={actionError}
            onCancel={stopDraft}
            onSend={sendAction}
          />
        ) : (
          <ConversationPane
            s={s}
            convs={convs}
            when={
              cal && shownWeek
                ? { week: shownWeek, days: cal.days.map((d) => ({ code: d, name: cal.day_names[d], date: dates?.[d]?.label })) }
                : undefined
            }
            attachments={{
              items: picks.map((e) => ({ key: e.session, label: `${e.title} ${e.kind} · ${dayName(e.day)} ${label(e.slot)}` })),
              remove: (key) => setPicks((p) => p.filter((x) => x.session !== key)),
              compose: (t) => (picks.length ? `${t.replace(/[.!?]$/, "")} (${listed(picks.map(phrase))}).` : t),
              clear: () => setPicks([]),
            }}
          />
        )}
      </Card>
      {menu && (
        <ClassMenu at={menu} cls={menu.cls} own={teaching && menu.cls.faculty === user?.id} onPick={startAction} onClose={() => setMenu(null)} />
      )}
      <DetailsDrawer id={details} onClose={() => setDetails(null)} />
    </div>
  );
}
