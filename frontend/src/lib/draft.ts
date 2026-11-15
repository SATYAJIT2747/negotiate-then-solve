import type { InstanceView, TimetableEntry } from "./types";

/** A request made by clicking instead of typing: cancel a class, move it, or swap it with a colleague's. It is
 * sent as data (``intent``, read by no parser) together with the same request in words (``sentence``), which is
 * what people see in the conversation. Shared by the timetable's action card and the New request builder. */

export type ActionKind = "cancel" | "move" | "swap";
export interface Target {
  day?: string;
  slot?: number;
  part?: "morning" | "afternoon";
}
export interface WeekChoice {
  mode: "this" | "from" | "every" | "pick";
  weeks: number[];
}
export interface Draft {
  kind: ActionKind;
  cls: TimetableEntry;
  to: Target | null;
  theirs: TimetableEntry | null;
  when: WeekChoice;
  hard: boolean;
  reason: string;
}

/** Which "when" choices each kind of request offers. */
export const WHEN: Record<ActionKind, WeekChoice["mode"][]> = { cancel: ["this", "pick"], move: ["this", "from", "pick"], swap: ["this", "every", "pick"] };

const listed = (xs: (string | number)[]) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);

/** The request in words: what the card sends. */
export function sentence(d: Draft, inst: InstanceView, week: number): string {
  const day = (x: string) => inst.day_names[x] ?? x;
  const time = (s: number) => inst.slots.find((x) => x.slot === s)?.label ?? `slot ${s}`;
  const cls = `my ${d.cls.title} ${d.cls.kind} on ${day(d.cls.day)} at ${time(d.cls.slot)}`;
  const weeks = [...d.when.weeks].sort((a, b) => a - b);
  const when =
    d.when.mode === "this"
      ? `in week ${week} only`
      : d.when.mode === "from"
        ? `from week ${week} onwards`
        : d.when.mode === "every"
          ? "every week"
          : weeks.length === 1
            ? `in week ${weeks[0]}`
            : `in weeks ${listed(weeks)}`;
  const reason = d.reason.trim() ? ` The reason: ${d.reason.trim().replace(/[.]$/, "")}.` : "";
  if (d.kind === "cancel") return `Please cancel ${cls} ${when}.${reason}`;
  if (d.kind === "swap" && d.theirs) {
    const o = d.theirs;
    return `Please swap ${cls} with ${o.faculty_name}'s ${o.title} ${o.kind} on ${day(o.day)} at ${time(o.slot)}, ${when}.${reason}`;
  }
  const t = d.to ?? {};
  const to =
    t.slot != null && t.day
      ? `to ${day(t.day)} at ${time(t.slot)}`
      : t.part
        ? `to any time ${t.part === "morning" ? "before" : "after"} lunch${t.day ? ` on ${day(t.day)}` : ""}`
        : `to any time on ${day(t.day!)}`;
  return `Please move ${cls} ${to}, ${when}${d.hard ? "; it has to be then" : ", if possible"}.${reason}`;
}

const FIRST_HOUR = 9; // slot 0 starts at 9:00 (language.corpus.FIRST_SLOT_HOUR)
export const clock = (slot: number) => `${String(FIRST_HOUR + slot).padStart(2, "0")}:00`;

/** The request as the pipeline takes it: one intent of ``language.intents`` (cancel, move or swap). */
export function intent(d: Draft, week: number): Record<string, unknown> {
  const weeks = [...d.when.weeks].sort((a, b) => a - b);
  const period = d.when.mode === "this" ? { weeks: [week] } : d.when.mode === "from" ? { from_week: week } : d.when.mode === "every" ? {} : { weeks };
  const justification = d.reason.trim() ? "stated" : "none";
  const which = { session: d.cls.session };
  if (d.kind === "cancel") return { kind: "cancel", which, period, justification };
  if (d.kind === "swap") return { kind: "swap", mine: which, theirs: { session: d.theirs!.session, owner: d.theirs!.faculty }, period, justification };
  const t = d.to ?? {};
  const to = t.slot != null && t.day ? { days: [t.day], at: [clock(t.slot)] } : { ...(t.part ? { part: t.part } : {}), ...(t.day ? { days: [t.day] } : {}) };
  return { kind: "move", which, to, period, hard: d.hard, justification };
}

export function ready(d: Draft): boolean {
  if (d.when.mode === "pick" && !d.when.weeks.length) return false;
  if (d.kind === "move") return !!d.to && (d.to.slot != null || !!d.to.part || !!d.to.day);
  if (d.kind === "swap") return !!d.theirs;
  return true;
}

