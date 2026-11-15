import type { DiffRow } from "./types";

/** A request's change to the timetable, week by week: a request over several weeks (an absence in weeks 10-12) has a
 * version for each week, the first one in ``proposal`` and the others in ``proposal.more``. */
export interface WeekChange {
  version: number;
  week: number | null;
  diff: DiffRow[];
}

/** Every week a proposal changes, in order. */
export function changedWeeks(p: { version: number; week: number | null; diff: DiffRow[]; more?: WeekChange[] } | null | undefined): WeekChange[] {
  if (!p) return [];
  return [{ version: p.version, week: p.week, diff: p.diff }, ...(p.more ?? []).map((m) => ({ version: m.version, week: m.week, diff: m.diff }))];
}

const where = (x: DiffRow["before"]) => (x ? `${x.day}:${x.time}:${x.room}` : "-");
const keyOf = (diff: DiffRow[]) =>
  diff
    .map((r) => `${r.session}|${where(r.before)}|${where(r.after)}`)
    .sort()
    .join(";");

/** Weeks that change the same classes the same way, together: the usual absence over weeks 10-12 is one group. */
export function groupWeeks(changes: WeekChange[]): { weeks: (number | null)[]; versions: number[]; diff: DiffRow[] }[] {
  const out: { key: string; weeks: (number | null)[]; versions: number[]; diff: DiffRow[] }[] = [];
  for (const c of changes) {
    const key = keyOf(c.diff);
    const g = out.find((x) => x.key === key);
    if (g) {
      g.weeks.push(c.week);
      g.versions.push(c.version);
    } else out.push({ key, weeks: [c.week], versions: [c.version], diff: c.diff });
  }
  return out.map(({ weeks, versions, diff }) => ({ weeks, versions, diff }));
}

/** [10, 11, 12] -> "weeks 10–12"; [10, 12] -> "weeks 10 and 12"; [10] -> "week 10"; [null] -> "every week". */
export function weeksLabel(weeks: (number | null)[]): string {
  const ws = [...new Set(weeks.filter((w): w is number => w != null))].sort((a, b) => a - b);
  if (!ws.length) return "every week";
  if (ws.length === 1) return `week ${ws[0]}`;
  if (ws[ws.length - 1] - ws[0] === ws.length - 1) return `weeks ${ws[0]}–${ws[ws.length - 1]}`;
  return `weeks ${ws.slice(0, -1).join(", ")} and ${ws[ws.length - 1]}`;
}

/** [35, 36, 37] -> "versions 35–37"; [35] -> "version 35". */
export function versionsLabel(vs: number[]): string {
  const s = [...vs].sort((a, b) => a - b);
  if (s.length === 1) return `version ${s[0]}`;
  return s[s.length - 1] - s[0] === s.length - 1 ? `versions ${s[0]}–${s[s.length - 1]}` : `versions ${s.join(", ")}`;
}
