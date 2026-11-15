import clsx from "clsx";
import { classTile } from "./classTile";
import type { SemCalendar, SemMeeting } from "../lib/types";

const ROW = 92; // px per lecture slot
const LUNCH = 30;

function mins(t: string): number {
  const [h, m] = t.split(":").map(Number);
  return h * 60 + m;
}

function hhmm(m: number): string {
  return `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
}

interface Row {
  kind: "slot" | "lunch" | "evening";
  a: number;
  b: number;
  label: string;
  end?: string;
  sub?: string;
  top: number;
  height: number;
}

/** The institute's grid: one row per lecture slot, lunch between slots 3 and 4, and the evening. */
function rows(calendar: SemCalendar): Row[] {
  const out: Row[] = [];
  let top = 0;
  const slots = calendar.lecture_slots.map(([a, b]) => [mins(a), mins(b)]);
  slots.forEach(([a, b], i) => {
    const next = slots[i + 1]?.[0];
    if (next !== undefined && next - b > 30) {
      out.push({ kind: "slot", a, b, label: calendar.lecture_slots[i][0], end: calendar.lecture_slots[i][1], sub: `slot ${i + 1}`, top, height: ROW });
      top += ROW;
      out.push({ kind: "lunch", a: b, b: next, label: "", top, height: LUNCH });
      top += LUNCH;
    } else {
      out.push({ kind: "slot", a, b, label: calendar.lecture_slots[i][0], end: calendar.lecture_slots[i][1], sub: `slot ${i + 1}`, top, height: ROW });
      top += ROW;
    }
  });
  const last = slots.at(-1)?.[1] ?? 18 * 60 + 40;
  out.push({ kind: "evening", a: last, b: 24 * 60, label: calendar.lecture_slots.at(-1)?.[1] ?? "18:40", sub: "evening", top, height: ROW });
  return out;
}

/** The rows a meeting occupies: those it covers for at least half an hour (a 09:00-12:00
 * lab is slots 1-2, not 1-3, so the 12:00 tutorial gets slot 3 to itself); failing that,
 * the row it overlaps most. */
function span(rs: Row[], m: SemMeeting): [number, number] {
  const a = mins(m.start);
  const b = mins(m.end);
  const overlap = rs.map((r) => (r.kind === "lunch" ? 0 : Math.max(0, Math.min(b, r.b) - Math.max(a, r.a))));
  const hit = overlap.map((o, i) => (o >= 30 ? i : -1)).filter((i) => i >= 0);
  if (hit.length) return [hit[0], hit.at(-1)!];
  const best = overlap.indexOf(Math.max(...overlap));
  const i = overlap[best] > 0 ? best : rs.length - 1;
  return [i, i];
}

/** Side-by-side lanes for meetings whose rows overlap. */
function lanes(items: { m: SemMeeting; r0: number; r1: number }[]) {
  const sorted = [...items].sort((x, y) => x.r0 - y.r0 || y.r1 - x.r1);
  const out: { m: SemMeeting; r0: number; r1: number; lane: number; of: number }[] = [];
  let cluster: typeof out = [];
  let end = -1;
  const ends: number[] = [];
  const flush = () => {
    const n = Math.max(1, ...cluster.map((c) => c.lane + 1));
    cluster.forEach((c) => (c.of = n));
    out.push(...cluster);
    cluster = [];
    ends.length = 0;
  };
  for (const it of sorted) {
    if (it.r0 > end && cluster.length) flush();
    let lane = ends.findIndex((e) => e < it.r0);
    if (lane < 0) lane = ends.length;
    ends[lane] = it.r1;
    cluster.push({ ...it, lane, of: 1 });
    end = Math.max(end, it.r1);
  }
  if (cluster.length) flush();
  return out;
}

export function SemesterGrid({
  calendar,
  meetings,
  changed,
  optional,
  onPick,
}: {
  calendar: SemCalendar;
  meetings: SemMeeting[];
  changed?: Set<string>;
  /** Components drawn as options (dashed, lighter): electives a student may or may not take. */
  optional?: Set<string>;
  onPick?: (m: SemMeeting) => void;
}) {
  const rs = rows(calendar);
  const height = rs.at(-1)!.top + rs.at(-1)!.height;
  const blockedRows = (d: string) =>
    calendar.blocked
      .filter((w) => w.days.includes(d))
      .map((w) => {
        const inside = rs.map((r, i) => (r.kind === "slot" && mins(w.start) <= r.a && r.a < mins(w.end) ? i : -1)).filter((i) => i >= 0);
        return inside.length ? { w, r0: inside[0], r1: inside.at(-1)! } : null;
      })
      .filter((x) => x !== null);

  return (
    <div className="overflow-x-auto">
      <div className="grid min-w-[900px]" style={{ gridTemplateColumns: `100px repeat(${calendar.days.length}, minmax(0, 1fr))` }}>
        <div />
        {calendar.days.map((d) => (
          <div key={d} className="pb-2 text-center text-[12.5px] font-medium text-ink-2">
            {d}
            {calendar.mirror[d] && <span className="ml-1 text-[11px] font-normal text-ink-3">↔ {calendar.mirror[d]}</span>}
          </div>
        ))}
        <div className="relative" style={{ height }}>
          {rs.map((r, i) =>
            r.kind === "lunch" ? (
              <div key={i} className="absolute right-2 flex items-center whitespace-nowrap font-mono text-[10.5px] text-ink-3" style={{ top: r.top, height: r.height }}>
                {hhmm(r.a)} – {hhmm(r.b)}
              </div>
            ) : (
              <div key={i} className="absolute right-2 text-right font-mono text-[10.5px] leading-tight text-ink-3" style={{ top: r.top + 4 }}>
                <div className="whitespace-nowrap text-ink-2">{r.end ? `${r.label} – ${r.end}` : `from ${r.label}`}</div>
                {r.sub && <div className="mt-0.5 text-[9.5px]">{r.sub}</div>}
              </div>
            ),
          )}
        </div>
        {calendar.days.map((d) => {
          const day = meetings.filter((m) => m.day === d).map((m) => {
            const [r0, r1] = span(rs, m);
            return { m, r0, r1 };
          });
          return (
            <div key={d} className="relative border-l border-line" style={{ height }}>
              {rs.map((r, i) =>
                r.kind === "lunch" ? (
                  <div
                    key={i}
                    className="absolute inset-x-0 grid place-items-center border-t border-line/70 bg-panel-2/60 text-[10px] text-ink-3"
                    style={{ top: r.top, height: r.height }}
                  >
                    lunch
                  </div>
                ) : (
                  <div key={i} className={clsx("absolute inset-x-0 border-t border-line/70", r.kind === "evening" && "bg-panel-2/30")} style={{ top: r.top, height: r.height }} />
                ),
              )}
              {blockedRows(d).map(({ w, r0, r1 }, i) => (
                <div
                  key={i}
                  className="hatch absolute inset-x-0 grid place-items-center text-[10.5px] text-ink-3"
                  style={{ top: rs[r0].top, height: rs[r1].top + rs[r1].height - rs[r0].top }}
                  title={w.reason}
                >
                  free afternoon
                </div>
              ))}
              {lanes(day).map(({ m, r0, r1, lane, of }) => {
                const top = rs[r0].top + 2;
                const h = rs[r1].top + rs[r1].height - rs[r0].top - 4;
                const opt = optional?.has(m.component);
                const tile = classTile(m.course, { changed: changed?.has(m.component), dashed: opt, tint: m.kind === "lecture" ? 12 : 7 });
                const exact = m.kind !== "lecture";
                return (
                  <button
                    key={`${m.component}-${m.day}`}
                    onClick={() => onPick?.(m)}
                    title={`${m.course} ${m.name} · ${m.label} · ${m.start}-${m.end} · ${m.rooms.join(", ")}${m.half !== "full" ? ` · ${m.half}` : ""}`}
                    className={clsx("absolute flex flex-col justify-start px-1.5 py-1", tile.className)}
                    style={{
                      top,
                      height: h,
                      left: `calc(${(lane / of) * 100}% + 2px)`,
                      width: `calc(${100 / of}% - 4px)`,
                      ...tile.style,
                    }}
                  >
                    <p className="truncate text-[11px] font-semibold leading-tight">
                      {m.course}
                      {m.half !== "full" && <span className="ml-1 font-normal text-ink-3">{m.half}</span>}
                    </p>
                    <p className="truncate text-[10.5px] leading-tight text-ink-2">{m.kind === "lecture" ? m.name : m.label}</p>
                    {exact && <p className="truncate font-mono text-[10px] leading-tight text-ink-3">{m.start}–{m.end}</p>}
                    {h > 56 && <p className="truncate text-[10px] leading-tight text-ink-3">{m.rooms.join(", ")}</p>}
                  </button>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}
