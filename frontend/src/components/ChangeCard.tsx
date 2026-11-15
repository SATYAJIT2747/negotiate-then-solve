import { ArrowRight, CalendarClock, Info } from "lucide-react";
import { Link } from "react-router-dom";
import { useAuth } from "../lib/auth";
import { ago } from "../lib/meta";
import type { ChangeView, FeedItem, PlacementView } from "../lib/types";

const DAY: Record<string, string> = { Mon: "Mon", Tue: "Tue", Wed: "Wed", Thu: "Thu", Fri: "Fri", Sat: "Sat" };

/** "Week 9", "Weeks 10 and 11", "Every week". */
export function weeksLabel(c: ChangeView): string {
  const ws = c.weeks.map((w) => w.week).filter((w): w is number => w !== null);
  if (!ws.length) return "Every week";
  if (ws.length === 1) return `Week ${ws[0]}`;
  return `Weeks ${ws.slice(0, -1).join(", ")} and ${ws[ws.length - 1]}`;
}

const when = (p: PlacementView) => `${DAY[p.day] ?? p.day} ${p.time} · ${p.room_name ?? p.room}`;

/** A change to this person's timetable: each class before and after, week by week, why (when anyone may
 * know), who approved it, and a link to the week on their timetable. */
export function ChangeCard({ item, change }: { item: FeedItem; change: ChangeView }) {
  const { user } = useAuth();
  const firstWeek = change.weeks.find((w) => w.week !== null)?.week;
  return (
    <div className="max-w-[640px] overflow-hidden rounded-lg border border-line bg-panel">
      <div className="flex items-center gap-2 border-b border-line bg-info/[0.06] px-4 py-2.5">
        <CalendarClock size={15} className="shrink-0 text-info" />
        <p className="text-[13.5px] font-semibold">{change.title}</p>
        <span className="ml-auto shrink-0 text-[12px] text-ink-3">{weeksLabel(change)}</span>
      </div>
      <div className="divide-y divide-line">
        {change.weeks.map((w) => (
          <div key={w.week ?? "all"} className="px-4 py-2.5">
            {change.weeks.length > 1 && <p className="pb-1.5 text-[11.5px] font-medium uppercase tracking-wide text-ink-3">{w.week !== null ? `Week ${w.week}` : "Every week"}</p>}
            <ul className="space-y-2">
              {w.rows.map((r) => (
                <li key={r.session} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-0.5 sm:grid-cols-[minmax(0,1.1fr)_minmax(0,1.4fr)]">
                  <div className="min-w-0">
                    <p className="truncate text-[13.5px] font-medium">
                      {r.name}
                      {r.extra && <span className="ml-1.5 rounded bg-brand/15 px-1 text-[10.5px] font-medium text-brand">extra</span>}
                    </p>
                    {r.faculty && r.faculty !== user?.id && <p className="truncate text-[11.5px] text-ink-3">{r.faculty_name}</p>}
                  </div>
                  <div className="flex flex-wrap items-center gap-x-2 text-[12.5px]">
                    {r.before ? <span className="text-ink-3 line-through decoration-ink-3/60">{when(r.before)}</span> : <span className="text-ink-3">new</span>}
                    <ArrowRight size={12} className="shrink-0 text-ink-3" />
                    {r.after ? <span className="font-medium text-ok">{when(r.after)}</span> : <span className="font-medium text-bad">Not held this week</span>}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      {change.reason && (
        <p className="flex items-start gap-2 border-t border-line px-4 py-2 text-[12.5px] text-ink-2">
          <Info size={13} className="mt-0.5 shrink-0 text-ink-3" />
          <span>Why: {change.reason.charAt(0).toUpperCase() + change.reason.slice(1)}.</span>
        </p>
      )}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line bg-panel-2/40 px-4 py-2 text-[11.5px] text-ink-3">
        <span>
          Approved by {item.from_name ?? "the timetable office"} · {ago(item.at)}
          {item.version != null ? ` · timetable version ${item.version}` : ""}
        </span>
        {firstWeek != null && (
          <Link to={`/?week=${firstWeek}`} className="ml-auto font-medium text-brand hover:underline">
            See week {firstWeek} on your timetable →
          </Link>
        )}
      </div>
    </div>
  );
}
