import clsx from "clsx";
import { WHEN, type ActionKind, type WeekChoice } from "../../lib/draft";

/** Which weeks a click-made request is for: the week in view only, from it on, every week, or weeks picked one by
 * one; weeks already over cannot be chosen. ``weeks``: how many the semester has; ``current``: the teaching week
 * now. Used by the timetable's action card and the New request builder. */
export function WhenField({ kind, value, onChange, week, weeks, current }: {
  kind: ActionKind;
  value: WeekChoice;
  onChange: (w: WeekChoice) => void;
  week: number;
  weeks: number;
  current: number;
}) {
  const label = { this: `Week ${week} only`, from: `From week ${week} on`, every: "Every week", pick: "Pick weeks" };
  return (
    <>
      <div className="flex flex-wrap gap-1.5">
        {WHEN[kind].map((m) => {
          const passed = (m === "this" || m === "from") && week < current;
          return (
            <button
              key={m}
              disabled={passed}
              title={passed ? `Week ${week} has passed` : undefined}
              onClick={() => onChange({ mode: m, weeks: m === "pick" ? (value.weeks.length ? value.weeks : week >= current ? [week] : []) : [] })}
              className={clsx(segment(value.mode === m), passed && "cursor-not-allowed opacity-40")}
            >
              {label[m]}
            </button>
          );
        })}
      </div>
      {value.mode === "pick" && (
        <div className="mt-2 flex flex-wrap gap-1">
          {Array.from({ length: weeks }, (_, i) => i + 1).map((w) => {
            const on = value.weeks.includes(w);
            const passed = w < current;
            return (
              <button
                key={w}
                disabled={passed}
                title={passed ? `Week ${w} has passed` : undefined}
                onClick={() => onChange({ ...value, weeks: on ? value.weeks.filter((x) => x !== w) : [...value.weeks, w] })}
                className={clsx(
                  "grid h-7 w-7 place-items-center rounded text-[12px] tabular-nums",
                  on ? "bg-brand font-semibold text-panel" : passed ? "cursor-not-allowed text-ink-3/40" : "border border-line hover:bg-panel-2",
                )}
              >
                {w}
              </button>
            );
          })}
        </div>
      )}
    </>
  );
}

/** A choice button in a row of choices, selected or not. */
export const segment = (on: boolean) =>
  clsx("rounded-md border px-2.5 py-1 text-[12.5px]", on ? "border-brand bg-brand/10 text-brand" : "border-line text-ink-2 hover:bg-panel-2");
