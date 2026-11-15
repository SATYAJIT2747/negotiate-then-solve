import clsx from "clsx";
import { CalendarDays, Check, Plus } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { Card } from "./ui";

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
/** Whole-phrase match, so "in week 1" is not found inside "in week 10". */
const has = (text: string, phrase: string) => new RegExp(`\\b${escape(phrase)}\\b`, "i").test(text);
const strip = (text: string, phrase: string) => {
  const out = text.replace(new RegExp(`\\s*\\b${escape(phrase)}\\b`, "i"), "").replace(/\s+([.,;:!?])/g, "$1").trim();
  return out ? out.charAt(0).toUpperCase() + out.slice(1) : out;
};
const dayOf = (mondayIso: string, plus: number) => {
  const d = new Date(`${mondayIso}T00:00:00`);
  d.setDate(d.getDate() + plus);
  return d;
};
const fmt = (d: Date) => d.toLocaleDateString(undefined, { day: "numeric", month: "short" });

/** "When is this for?": the semester's weeks, and for the week in view the signed-in person's classes
 * day by day (that week's actual timetable). A week or a day can be written into the request. */
export function WeekCalendar({ text, onPick, onRemove }: { text: string; onPick?: (phrase: string) => void; onRemove?: (next: string) => void }) {
  const { data: cal } = useApi(() => api.calendar(), [], 15000);
  const [view, setView] = useState<number | null>(null);
  useEffect(() => {
    if (cal && view === null) setView(cal.current_week);
  }, [cal, view]);
  if (!cal || view === null) return null;

  const editable = !!onPick && !!onRemove;
  const toggle = (phrase: string) => {
    if (!editable) return;
    if (has(text, phrase)) onRemove!(strip(text, phrase));
    else onPick!(phrase);
  };
  const w = cal.weeks[view - 1];
  const weekPhrase = `in week ${w.week}`;
  const mentioned = (n: number) => has(text, `in week ${n}`) || cal.days.some((d) => has(text, `on ${cal.day_names[d]} in week ${n}`));

  return (
    <Card className="@container mt-4 p-4">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <p className="flex items-center gap-2 text-[13.5px] font-medium">
          <CalendarDays size={15} className="text-ink-3" /> When is this for?
        </p>
        <p className="text-[11.5px] text-ink-3">Pick a week to see your classes in it</p>
      </div>

      {/* weeks: numbers in a narrow panel (the date on hover and in the line below), number and date when there is room */}
      <div className="grid grid-cols-8 gap-1">
        {cal.weeks.map((x) => {
          const past = x.week < cal.current_week;
          return (
            <button
              key={x.week}
              onClick={() => setView(x.week)}
              title={`Week ${x.week} · from ${fmt(dayOf(x.monday, 0))}`}
              className={clsx(
                "relative rounded-md border px-1 py-1 text-center transition-colors @2xl:px-2 @2xl:py-1.5 @2xl:text-left",
                x.week === view ? "border-brand bg-brand/10" : "border-line hover:bg-panel-2",
                past && x.week !== view && "opacity-50",
              )}
            >
              <span className={clsx("block whitespace-nowrap text-[12.5px] font-medium tabular-nums", x.week === cal.current_week && "text-brand")}>
                <span className="hidden @2xl:inline">Week </span>
                {x.week}
                {mentioned(x.week) && <Check size={10} className="ml-0.5 inline text-brand" />}
              </span>
              <span className="hidden text-[11px] text-ink-3 @2xl:block">{fmt(dayOf(x.monday, 0))}</span>
              {x.own_changes && <span title="This week has changes of its own" className="absolute right-1.5 top-1.5 size-1.5 rounded-full bg-warn" />}
            </button>
          );
        })}
      </div>

      <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-line pt-3">
        <p className="text-[13px]">
          <span className="font-medium">Week {w.week}</span>
          <span className="text-ink-3">
            {" "}
            · {fmt(dayOf(w.monday, 0))} – {fmt(dayOf(w.monday, cal.days.length - 1))}
            {w.week === cal.current_week ? " · this week" : w.week < cal.current_week ? " · already past" : ""}
            {w.own_changes ? " · has changes of its own" : ""}
          </span>
        </p>
        {editable && (
          <button
            onClick={() => toggle(weekPhrase)}
            className={clsx(
              "flex items-center gap-1 rounded-md border px-2.5 py-1 text-[12.5px]",
              has(text, weekPhrase) ? "border-brand bg-brand text-white" : "border-line hover:bg-panel-2",
            )}
          >
            {has(text, weekPhrase) ? <Check size={13} /> : <Plus size={13} />} {has(text, weekPhrase) ? `Week ${w.week} added` : `Use week ${w.week}`}
          </button>
        )}
      </div>

      {/* days: one row each in a narrow panel, side by side when there is room */}
      <div className="mt-3 grid grid-cols-1 gap-1.5 @2xl:grid-cols-5 @2xl:gap-2">
        {cal.days.map((d, i) => {
          const phrase = `on ${cal.day_names[d]} in week ${w.week}`;
          const picked = has(text, phrase);
          const mine = w.classes.filter((c) => c.day === d);
          return (
            <button
              key={d}
              onClick={() => toggle(phrase)}
              disabled={!editable}
              title={editable ? (picked ? "Take this day out of the request" : `Add "${phrase}" to the request`) : undefined}
              className={clsx(
                "flex gap-3 rounded-md border px-2.5 py-2 text-left transition-colors @2xl:min-h-[92px] @2xl:flex-col @2xl:gap-0 @2xl:p-2",
                picked ? "border-brand bg-brand/10" : "border-line",
                editable && !picked && "hover:bg-panel-2",
              )}
            >
              <span className="flex w-20 shrink-0 flex-col @2xl:w-auto @2xl:flex-row @2xl:items-baseline @2xl:justify-between @2xl:gap-1">
                <span className="text-[12.5px] font-medium">{cal.day_names[d]}</span>
                <span className="text-[11px] text-ink-3">{fmt(dayOf(w.monday, i))}</span>
              </span>
              <span className="min-w-0 flex-1 space-y-1 @2xl:mt-1.5">
                {mine.length ? (
                  mine.map((c) => (
                    <span key={c.session} className={clsx("block truncate text-[11.5px] leading-snug @2xl:whitespace-normal", c.cancelled ? "text-bad line-through" : "text-ink-2")}>
                      <span className="font-mono text-[10.5px] text-ink-3">{c.time}</span> {c.title}
                      {c.kind === "practical" ? " (lab)" : ""}
                    </span>
                  ))
                ) : (
                  <span className="block text-[11.5px] text-ink-3">No classes</span>
                )}
              </span>
            </button>
          );
        })}
      </div>
      {editable && <p className="mt-2 text-[11.5px] text-ink-3">Click a day to add it to your request; click again to take it out.</p>}
    </Card>
  );
}
