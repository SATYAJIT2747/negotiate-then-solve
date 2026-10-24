import { X } from "lucide-react";
import { useEffect, useMemo, useState, type MouseEvent } from "react";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useApi } from "../lib/hooks";
import { canMoveTo, dayPassed, startWeek, swappableWith } from "../lib/scheduling";
import type { CalendarView, InstanceView, SliceView, TimetableEntry } from "../lib/types";
import { ActionCard, ClassMenu, intent, sentence, type ActionKind, type Draft } from "./ClassActions";
import { TimetableGrid } from "./TimetableGrid";
import { WeekStepper } from "./WeekStepper";
import { Skeleton } from "./ui";

/** What a chat answer asks the popup to show: its week, and its cells highlighted ("View in Timetable"). */
export interface Highlight {
  week: number;
  /** "Mon:2": free hours (green) */
  free: Set<string>;
  /** session ids: the classes the answer is about (outlined) */
  classes: Set<string>;
}

/** A view's cells as a highlight: free hours green, the classes it names outlined. */
export function highlightOf(view: SliceView): Highlight {
  if (view.layout === "rows") {
    return { week: view.week, free: new Set(view.rows.flatMap((r) => r.free.map((s) => `${view.day}:${s}`))), classes: new Set() };
  }
  const free = new Set<string>();
  for (const c of view.cells) if (c.kind === "free") for (let i = 0; i < c.span; i++) free.add(`${c.day}:${c.slot + i}`);
  return { week: view.week, free, classes: new Set(view.cells.filter((c) => c.session).map((c) => c.session!)) };
}

/** The week's timetable over the chat (CHAT-PLAN step 5): the week stepper, "My timetable / All classes", and point
 * and click as on the home page before: a class → cancel, move, swap it, or name it in the message. A request made
 * here is sent at once (it was put together with clicks) and appears in the chat. */
export function TimetablePopup({ inst, cal, highlight, onClose, onSent, onName }: {
  inst: InstanceView | null;
  cal: CalendarView | null;
  highlight: Highlight | null;
  onClose: () => void;
  /** a request was sent from here (the chat shows its card) */
  onSent: () => void;
  /** "Name it in the message": the class in words, for the chat's input box */
  onName: (phrase: string) => void;
}) {
  const { user } = useAuth();
  const teaching = ["faculty", "guest_faculty", "hod"].includes(user?.role ?? "");
  const group = user?.role === "student" && user.id.startsWith("ST-") ? user.id.slice(3) : null;

  // the answer's week, else this week (at the weekend, the coming one: this week's classes are all over)
  const [week, setWeek] = useState<number | null>(highlight?.week ?? null);
  useEffect(() => {
    if (cal && week === null) setWeek(startWeek(cal));
  }, [cal, week]);
  const [scope, setScope] = useState<"mine" | "all">(teaching || group ? "mine" : "all");
  const { data: tt } = useApi(() => (week === null ? Promise.resolve(null) : api.timetable({ week })), [week], 5000);
  const mine = useMemo(
    () => (tt?.entries ?? []).filter((e) => (teaching ? e.faculty === user?.id : group ? e.groups.includes(group) : false)),
    [tt, teaching, group, user],
  );
  const all = useMemo(() => tt?.entries ?? [], [tt]);

  // a request being put together by clicks, and the menu that starts one
  const [menu, setMenu] = useState<{ x: number; y: number; cls: TimetableEntry } | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shade, setShade] = useState<Highlight | null>(highlight);
  const shaded = !draft && !!shade && shade.week === week;

  const outlined = useMemo(
    () => new Set(draft ? [draft.cls.session, ...(draft.theirs ? [draft.theirs.session] : [])] : shaded ? [...shade!.classes] : []),
    [draft, shaded, shade],
  );
  const target = draft?.kind === "move" && draft.to?.slot != null && draft.to.day ? { day: draft.to.day, slot: draft.to.slot } : null;
  const ghosts = target && draft ? [{ session: draft.cls.session, day: target.day, slot: target.slot, title: `${draft.cls.title} ${draft.cls.kind}`, note: "new time · not sent yet" }] : [];

  // where a class may go and what it may be swapped with: hints from this week's timetable (lib/scheduling)
  const weekOver = !!cal && cal.days.every((d) => dayPassed(cal, week, d));
  const slotState = (day: string, slot: number): "free" | "blocked" | null => {
    if (shaded) return shade!.free.has(`${day}:${slot}`) ? "free" : null;
    if (draft?.kind !== "move" || !inst) return null;
    if (day === draft.cls.day && slot === draft.cls.slot) return null; // where it is now
    return canMoveTo(inst, cal, week, all, draft.cls, day, slot) ? "free" : "blocked";
  };
  const swappable = useMemo(
    () => (draft?.kind === "swap" ? new Set(swappableWith(all, draft.cls).map((o) => o.session)) : null),
    [draft, all],
  );
  const dim = swappable ? new Set(all.filter((o) => o.session !== draft!.cls.session && !swappable.has(o.session)).map((o) => o.session)) : undefined;
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

  const dayName = (d: string) => inst?.day_names[d] ?? d;
  const label = (slot: number) => inst?.slots.find((s) => s.slot === slot)?.label ?? `slot ${slot}`;
  const phrase = (e: TimetableEntry) => {
    const whose = teaching && e.faculty === user?.id ? "my" : !teaching && group && e.groups.includes(group) ? "the" : `${e.faculty_name}'s`;
    return `${whose} ${e.title} ${e.kind} on ${dayName(e.day)} at ${label(e.slot)}`;
  };

  const stopDraft = () => {
    if (draft?.kind === "swap") setScope("mine");
    setDraft(null);
    setError(null);
  };
  // Esc leaves a request being put together, else closes the popup
  useEffect(() => {
    const esc = (ev: KeyboardEvent) => {
      if (ev.key !== "Escape" || menu) return;
      if (draft) stopDraft();
      else onClose();
    };
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [draft, menu]); // eslint-disable-line react-hooks/exhaustive-deps

  const pickClass = (e: TimetableEntry, ev: MouseEvent) => {
    if (draft && e.session === draft.cls.session) return stopDraft();
    if (draft?.kind === "swap") {
      if (swappable?.has(e.session)) setDraft({ ...draft, theirs: e });
      return;
    }
    if (draft) return;
    setMenu({ x: ev.clientX, y: ev.clientY, cls: e });
  };
  const startAction = (kind: ActionKind | "write") => {
    const cls = menu!.cls;
    setMenu(null);
    if (kind === "write") {
      onName(phrase(cls));
      return onClose();
    }
    setShade(null);
    setError(null);
    const passed = !!cal && week !== null && week < cal.current_week;
    setDraft({ kind, cls, to: null, theirs: null, when: { mode: passed ? "pick" : "this", weeks: [] }, hard: true, reason: "" });
    if (kind === "swap") setScope("all");
  };
  const sendAction = async () => {
    if (!draft || !inst || week === null) return;
    setSending(true);
    setError(null);
    try {
      await api.submitForm(sentence(draft, inst, week), intent(draft, week));
      setDraft(null);
      onSent();
      onClose();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  };

  const hint = draft?.kind === "move"
    ? weekOver
      ? `Week ${week} is over, so there is no free slot left in it: go to a later week to pick the new time.`
      : "Click a green slot for the new time."
    : draft?.kind === "swap"
      ? "Click the colleague's class to swap with. Classes that cannot be swapped are faded."
      : draft
        ? "Fill in the details on the right."
        : shaded
          ? "Green: the hours from the answer; outlined: its classes."
          : "Click a class to cancel, reschedule or swap it, or to name it in your message.";

  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/50 p-2 sm:p-6" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="flex h-full max-h-[52rem] w-full max-w-6xl flex-col overflow-hidden rounded-xl border border-line bg-bg shadow-2xl" role="dialog" aria-label="Timetable">
        <div className="flex flex-wrap items-center gap-3 border-b border-line px-4 py-2.5">
          {cal ? <WeekStepper cal={cal} week={week} onChange={(w) => w !== null && setWeek(w)} compact /> : <Skeleton className="h-8 w-72" />}
          <select
            value={scope}
            onChange={(e) => setScope(e.target.value as "mine" | "all")}
            className="h-8 rounded-md border border-line bg-panel px-2.5 text-[13px] outline-none hover:bg-panel-2 focus:border-brand/60"
          >
            {(teaching || group) && <option value="mine">My timetable</option>}
            <option value="all">All classes</option>
          </select>
          <button onClick={onClose} className="ml-auto grid size-8 place-items-center rounded-md text-ink-3 hover:bg-panel-2 hover:text-ink" aria-label="Close">
            <X size={16} />
          </button>
        </div>

        <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
          <div className="min-h-0 flex-1 overflow-auto p-3">
            {!inst || !tt ? (
              <Skeleton className="h-full min-h-60" />
            ) : scope === "mine" && !mine.length ? (
              <p className="px-3 py-16 text-center text-[13.5px] text-ink-3">No classes in the published timetable for this week.</p>
            ) : (
              <TimetableGrid
                instance={inst}
                entries={scope === "all" ? all : mine}
                changed={outlined}
                onPick={pickClass}
                fill={scope === "mine"}
                dates={dates}
                ghosts={ghosts}
                slotState={slotState}
                onSlot={draft?.kind === "move" ? (day, slot) => setDraft({ ...draft, to: { day, slot } }) : undefined}
                dim={dim}
              />
            )}
          </div>
          {draft && inst && week !== null && (
            <div className="max-h-[45%] border-t border-line lg:max-h-none lg:w-80 lg:border-l lg:border-t-0">
              <ActionCard d={draft} set={setDraft} inst={inst} week={week} weeks={inst.calendar.weeks} current={cal?.current_week ?? 1}
                          sending={sending} error={error} onCancel={stopDraft} onSend={sendAction} />
            </div>
          )}
        </div>

        <p className="flex flex-wrap items-center gap-x-2 border-t border-line px-4 py-2 text-[12px] text-ink-3">
          {hint}
          {shaded && (
            <button onClick={() => setShade(null)} className="font-medium text-brand hover:underline">
              Clear
            </button>
          )}
          {draft && (
            <button onClick={stopDraft} className="font-medium text-brand hover:underline">
              Cancel {draft.kind === "move" ? "rescheduling" : draft.kind === "swap" ? "the swap" : "this"}
            </button>
          )}
        </p>
      </div>
      {menu && <ClassMenu at={menu} cls={menu.cls} own={teaching && menu.cls.faculty === user?.id} onPick={startAction} onClose={() => setMenu(null)} />}
    </div>
  );
}
