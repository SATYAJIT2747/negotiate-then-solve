import type { CalendarView, InstanceView, SliceView } from "./types";

/** One line of a lookup answer in the chat: a stretch of time (free hours merged), a class, or a change. */
export interface Slot {
  /** "9 am–1 pm", "3 pm" */
  time: string;
  /** a class or a room / person: "Data Structures lecture", "Lab 1" */
  title?: string;
  /** under the title: "Room 102 · Section 1" */
  sub?: string;
  tone: "free" | "busy" | "class" | "cancelled" | "moved" | "extra";
  /** for busy hours and changes: "taken", "not held", "moved from here", "moved here", "extra class" */
  note?: string;
  start: number;
}

/** The slots of one day, under its date ("Oct 26, Mon"). */
export interface SlotGroup {
  day: string;
  date: string;
  slots: Slot[];
}

/** The 24-hour clock hour a slot starts at (slot 0 is the first teaching hour, e.g. 9 am). */
function hourOf(inst: InstanceView, slot: number): number {
  const m = /^(\d{1,2})(?::\d\d)?\s*(am|pm)$/i.exec(inst.slots[0]?.label.trim() ?? "9 am");
  const first = m ? (Number(m[1]) % 12) + (m[2].toLowerCase() === "pm" ? 12 : 0) : 9;
  return first + slot;
}

function clock(h: number): { n: number; half: string } {
  return { n: h % 12 || 12, half: h < 12 ? "am" : "pm" };
}

/** "9 am" for slot 0. */
export function at(inst: InstanceView, slot: number): string {
  const c = clock(hourOf(inst, slot));
  return `${c.n} ${c.half}`;
}

/** "9 am–1 pm", "2–3 pm": from the start of ``start`` to the start of ``end`` (am/pm once when both share it). */
export function range(inst: InstanceView, start: number, end: number): string {
  const a = clock(hourOf(inst, start));
  const b = clock(hourOf(inst, end));
  return a.half === b.half ? `${a.n}–${b.n} ${b.half}` : `${a.n} ${a.half}–${b.n} ${b.half}`;
}

/** Consecutive slots merged: [0, 1, 2, 3, 5] → [[0, 4], [5, 6]] (start, end). */
function runs(slots: number[]): [number, number][] {
  const out: [number, number][] = [];
  for (const s of [...slots].sort((x, y) => x - y)) {
    const last = out[out.length - 1];
    if (last && last[1] === s) last[1] = s + 1;
    else out.push([s, s + 1]);
  }
  return out;
}

/** "Oct 26, Mon": the date of ``day`` in ``week``, from the calendar's Mondays. */
export function dateOf(cal: CalendarView, week: number, day: string): string {
  const w = cal.weeks[week - 1];
  const name = (cal.day_names[day] ?? day).slice(0, 3);
  if (!w) return name;
  const d = new Date(`${w.monday}T00:00:00`);
  d.setDate(d.getDate() + cal.days.indexOf(day));
  return `${d.toLocaleDateString(undefined, { month: "short", day: "numeric" })}, ${name}`;
}

const CHANGE: Record<string, { tone: Slot["tone"]; note: string }> = {
  cancelled: { tone: "cancelled", note: "not held" },
  moved_from: { tone: "moved", note: "moved from here" },
  moved_to: { tone: "moved", note: "moved here" },
  extra: { tone: "extra", note: "extra class" },
};

/** A lookup answer's view as the chat lists it: per day (in calendar order, under its date), free hours merged
 * into stretches, an hour asked about "free" or "taken", each class or change, each room or person. */
export function slotsOf(view: SliceView, inst: InstanceView, cal: CalendarView): SlotGroup[] {
  if (view.layout === "rows") {
    const slots = view.rows.map((r) => ({
      title: r.name,
      time: runs(r.free).map(([a, b]) => range(inst, a, b)).join(", "),
      tone: "free" as const,
      start: Math.min(...r.free),
    }));
    return slots.length ? [{ day: view.day, date: dateOf(cal, view.week, view.day), slots }] : [];
  }
  const groups: SlotGroup[] = [];
  for (const day of view.days) {
    const cells = view.cells.filter((c) => c.day === day);
    const free = cells.filter((c) => c.kind === "free").map((c) => c.slot);
    const slots: Slot[] = [
      ...runs(free).map(([a, b]) => ({ time: range(inst, a, b), tone: "free" as const, start: a })),
      ...cells.filter((c) => c.kind === "busy").map((c) => ({ time: at(inst, c.slot), tone: "busy" as const, note: "taken", start: c.slot })),
      ...cells
        .filter((c) => c.kind !== "free" && c.kind !== "busy")
        .map((c) => ({
          time: range(inst, c.slot, c.slot + c.span),
          title: c.label,
          sub: c.sub,
          tone: c.kind === "class" ? ("class" as const) : CHANGE[c.kind].tone,
          note: c.kind === "class" ? undefined : CHANGE[c.kind].note,
          start: c.slot,
        })),
    ].sort((x, y) => x.start - y.start);
    if (slots.length) groups.push({ day, date: dateOf(cal, view.week, day), slots });
  }
  return groups;
}

/** The first ``n`` slots, in order, still grouped by date; and how many were left out. */
export function firstSlots(groups: SlotGroup[], n: number): { shown: SlotGroup[]; more: number } {
  const shown: SlotGroup[] = [];
  let left = n;
  let total = 0;
  for (const g of groups) {
    total += g.slots.length;
    if (left > 0) {
      shown.push({ ...g, slots: g.slots.slice(0, left) });
      left -= Math.min(left, g.slots.length);
    }
  }
  return { shown, more: Math.max(0, total - n) };
}
