import type { CalendarView, InstanceView, TimetableEntry } from "./types";

/** Hints for requests made by clicking, judged on one week's published timetable: where a class may go and what it
 * may be swapped with. Its teacher and its sections must be free, the lunch hour stays free, a day that is over is
 * out. Only hints; the solver checks the rest (rooms, pending changes). Shared by the timetable popup and the
 * request builder. */

/** The hour cells a class takes: "Mon:2", "Mon:3" for a 2-hour class at slot 2. */
export const cellsOf = (day: string, slot: number, duration: number) => Array.from({ length: duration }, (_, i) => `${day}:${slot + i}`);

/** Whether ``cells`` are free for ``e``'s teacher and sections, among the week's classes ``all`` (``except``: classes
 * left out, e.g. the one being moved). */
export function freeFor(all: TimetableEntry[], e: TimetableEntry, cells: string[], except: string[]): boolean {
  const taken = new Set(
    all
      .filter((o) => (o.faculty === e.faculty || o.groups.some((g) => e.groups.includes(g))) && !except.includes(o.session))
      .flatMap((o) => cellsOf(o.day, o.slot, o.duration)),
  );
  return !cells.some((c) => taken.has(c));
}

/** Whether ``day`` of ``week`` is already over (before today). */
export function dayPassed(cal: CalendarView | null, week: number | null, day: string): boolean {
  const w = week && cal ? cal.weeks[week - 1] : null;
  if (!w || !cal) return false;
  const at = new Date(`${w.monday}T00:00:00`);
  at.setDate(at.getDate() + cal.days.indexOf(day));
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  return at < today;
}

/** Whether class ``e`` may start at (``day``, ``slot``) in ``week``: inside the day, not over lunch, not on a day
 * that is over, its teacher and sections free (its own place does not count). */
export function canMoveTo(inst: InstanceView, cal: CalendarView | null, week: number | null, all: TimetableEntry[],
                          e: TimetableEntry, day: string, slot: number): boolean {
  if (day === e.day && slot === e.slot) return false;
  if (dayPassed(cal, week, day)) return false;
  const lunch = inst.calendar.lunch_slot;
  if (slot + e.duration > inst.calendar.slots_per_day) return false;
  if (lunch != null && slot <= lunch && lunch < slot + e.duration) return false;
  return freeFor(all, e, cellsOf(day, slot, e.duration), [e.session]);
}

/** Every (day, slot) class ``e`` may move to in ``week``, in time order. */
export function freeSlotsFor(inst: InstanceView, cal: CalendarView | null, week: number | null, all: TimetableEntry[],
                             e: TimetableEntry): { day: string; slot: number }[] {
  const out: { day: string; slot: number }[] = [];
  for (const day of inst.calendar.days)
    for (let slot = 0; slot < inst.calendar.slots_per_day; slot++)
      if (canMoveTo(inst, cal, week, all, e, day, slot)) out.push({ day, slot });
  return out;
}

/** Colleagues' classes ``e`` can be swapped with: same length, and each teacher and section free at the other's time. */
export function swappableWith(all: TimetableEntry[], e: TimetableEntry): TimetableEntry[] {
  return all.filter(
    (o) =>
      o.session !== e.session &&
      o.faculty !== e.faculty &&
      o.duration === e.duration &&
      freeFor(all, e, cellsOf(o.day, o.slot, o.duration), [e.session, o.session]) &&
      freeFor(all, o, cellsOf(e.day, e.slot, e.duration), [e.session, o.session]),
  );
}

/** The week to open on: this one, or at the weekend the next (this week's classes are all over). */
export function startWeek(cal: CalendarView): number {
  const w = cal.weeks[cal.current_week - 1];
  const friday = w ? new Date(`${w.monday}T00:00:00`) : null;
  friday?.setDate(friday.getDate() + cal.days.length - 1);
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  return friday && friday < today && cal.current_week < cal.weeks.length ? cal.current_week + 1 : cal.current_week;
}
