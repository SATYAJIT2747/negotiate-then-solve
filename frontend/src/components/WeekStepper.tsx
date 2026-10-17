import clsx from "clsx";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useEffect } from "react";
import type { CalendarView } from "../lib/types";

const at = (mondayIso: string, plus = 0) => {
  const d = new Date(`${mondayIso}T00:00:00`);
  d.setDate(d.getDate() + plus);
  return d;
};
const fmt = (d: Date, month = true) => d.toLocaleDateString(undefined, month ? { day: "numeric", month: "short" } : { day: "numeric" });

/** Moving between the semester's weeks: arrows (and the arrow keys), a strip of week numbers to
 * jump with, back to this week, and the semester timetable (``week`` null). */
export function WeekStepper({ cal, week, onChange, compact = false }: {
  cal: CalendarView;
  week: number | null;
  onChange: (w: number | null) => void;
  /** Just the week number, without the "This week" and "Semester" buttons (the page offers the semester
   * view itself): the dates are in the grid's day headings, the current week is marked in the strip, and
   * the bar keeps its width whichever week is shown. */
  compact?: boolean;
}) {
  const n = cal.weeks.length;
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (week === null || /INPUT|TEXTAREA|SELECT/.test(t.tagName) || t.isContentEditable) return;
      if (e.key === "ArrowLeft" && week > 1) onChange(week - 1);
      if (e.key === "ArrowRight" && week < n) onChange(week + 1);
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [week, n, onChange]);

  const w = week ? cal.weeks[week - 1] : null;
  const last = cal.days.length - 1;
  const sameMonth = w && at(w.monday).getMonth() === at(w.monday, last).getMonth();

  return (
    <div className="flex flex-wrap items-center gap-2">
      <div className="flex items-center rounded-md border border-line">
        <button
          onClick={() => week && week > 1 && onChange(week - 1)}
          disabled={!week || week <= 1}
          title="Previous week (←)"
          className="grid size-8 place-items-center text-ink-3 hover:text-ink disabled:opacity-30"
        >
          <ChevronLeft size={15} />
        </button>
        <div className={clsx("px-1 text-center leading-tight", compact ? "min-w-[72px]" : "min-w-[132px]")}>
          {compact ? (
            <p className="text-[13px] font-medium">{w ? `Week ${w.week}` : "Semester"}</p>
          ) : w ? (
            <>
              <p className="text-[13px] font-medium">Week {w.week}</p>
              <p className="text-[11px] text-ink-3">
                {fmt(at(w.monday))} – {fmt(at(w.monday, last), !sameMonth)}
                {w.week === cal.current_week ? " · this week" : ""}
              </p>
            </>
          ) : (
            <>
              <p className="text-[13px] font-medium">Semester</p>
              <p className="text-[11px] text-ink-3">no one-week changes</p>
            </>
          )}
        </div>
        <button
          onClick={() => onChange(week ? Math.min(n, week + 1) : cal.current_week)}
          disabled={!!week && week >= n}
          title="Next week (→)"
          className="grid size-8 place-items-center text-ink-3 hover:text-ink disabled:opacity-30"
        >
          <ChevronRight size={15} />
        </button>
      </div>

      <div className="flex items-center gap-0.5" role="group" aria-label="Weeks">
        {cal.weeks.map((x) => {
          const selected = x.week === week;
          const now = x.week === cal.current_week;
          return (
            <button
              key={x.week}
              onClick={() => onChange(x.week)}
              title={`Week ${x.week} · ${fmt(at(x.monday))}${now ? " · this week" : ""}${x.own_changes ? " · has changes of its own" : ""}`}
              className={clsx(
                "relative grid h-7 w-6 place-items-center rounded text-[11.5px] tabular-nums transition-colors",
                selected ? "bg-brand font-semibold text-white" : now ? "font-semibold text-brand ring-1 ring-brand/60" : "hover:bg-panel-2",
                !selected && x.week < cal.current_week && "text-ink-3/70",
              )}
            >
              {x.week}
              {x.own_changes && <span className={clsx("absolute bottom-0.5 size-1 rounded-full", selected ? "bg-white" : "bg-warn")} />}
            </button>
          );
        })}
      </div>

      {!compact && week !== cal.current_week && (
        <button onClick={() => onChange(cal.current_week)} className="h-7 rounded-md border border-line px-2.5 text-[12px] hover:bg-panel-2">
          This week
        </button>
      )}
      {!compact && (
      <button
        onClick={() => onChange(week === null ? cal.current_week : null)}
        title="The semester timetable, without changes made for one week only"
        className={clsx("h-7 rounded-md border px-2.5 text-[12px]", week === null ? "border-brand bg-brand/10 text-brand" : "border-line text-ink-3 hover:bg-panel-2")}
      >
        Semester
      </button>
      )}
    </div>
  );
}
