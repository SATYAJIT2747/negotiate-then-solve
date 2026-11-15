import { CalendarDays, Check, Plus } from "lucide-react";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useApi } from "../lib/hooks";
import { mentions } from "../lib/mention";
import type { TimetableEntry } from "../lib/types";
import { Card, CardHeader, Skeleton } from "./ui";

/** The classes a person teaches (faculty) or attends (a class rep's group), from the published
 * timetable, so a request can name them the way the timetable does. Clicking one adds it to the text. */
export function MyClasses({ onPick, text = "" }: { onPick?: (phrase: string) => void; text?: string }) {
  const { user } = useAuth();
  const { data: tt } = useApi(() => api.timetable(), [], 10_000); // approvals elsewhere move classes
  const { data: inst } = useApi(() => api.instance(), []);
  if (!user) return null;
  const teaching = ["faculty", "hod", "guest_faculty"].includes(user.role);
  const group = user.role === "student" && user.id.startsWith("ST-") ? user.id.slice(3) : null;
  if (!teaching && !group) return null;

  const days = inst?.calendar.days ?? [];
  const dayName = (d: string) => inst?.day_names[d] ?? d;
  const label = (slot: number) => inst?.slots.find((s) => s.slot === slot)?.label ?? `slot ${slot}`;
  const mine = (tt?.entries ?? [])
    .filter((e) => (teaching ? e.faculty === user.id : e.groups.includes(group!)))
    .sort((a, b) => days.indexOf(a.day) - days.indexOf(b.day) || a.slot - b.slot);

  const phrase = (e: TimetableEntry) =>
    teaching
      ? `my ${e.title} ${e.kind} on ${dayName(e.day)} at ${label(e.slot)}`
      : `the ${e.title} ${e.kind} on ${dayName(e.day)} at ${label(e.slot)}`;

  return (
    <Card>
      <CardHeader
        icon={CalendarDays}
        title={teaching ? "Classes you teach" : "Your group's classes"}
        subtitle={tt ? `Published timetable, version ${tt.version}${onPick ? " · click one to mention it" : ""}` : undefined}
      />
      {!tt || !inst ? (
        <div className="p-4">
          <Skeleton className="h-40" />
        </div>
      ) : !mine.length ? (
        <p className="p-5 text-[13px] text-ink-3">No classes in the published timetable.</p>
      ) : (
        <ul className="max-h-[420px] divide-y divide-line overflow-y-auto">
          {mine.map((e) => (
            <li key={e.session}>
              <button
                onClick={onPick ? () => onPick(phrase(e)) : undefined}
                disabled={!onPick}
                className="group flex w-full items-start gap-3 px-4 py-2.5 text-left enabled:hover:bg-panel-2/60"
              >
                <div className="w-20 shrink-0 leading-tight">
                  <p className="text-[13px] font-semibold">{dayName(e.day)}</p>
                  <p className="text-[12px] text-ink-3">
                    {label(e.slot)}
                    {e.duration > 1 ? ` · ${e.duration} h` : ""}
                  </p>
                </div>
                <div className="min-w-0 flex-1 leading-tight">
                  <p className="truncate text-[13.5px] font-medium">
                    {e.title} <span className="font-normal text-ink-3">{e.kind}</span>
                  </p>
                  <p className="truncate text-[12px] text-ink-3">
                    {e.room_name} · {teaching ? e.group_names.join(", ") : e.faculty_name}
                  </p>
                </div>
                {onPick &&
                  (mentions(text, phrase(e)) ? (
                    <Check size={14} className="mt-1 shrink-0 text-ok" aria-label="mentioned" />
                  ) : (
                    <Plus size={14} className="mt-1 shrink-0 text-ink-3 opacity-0 transition-opacity group-hover:opacity-100" />
                  ))}
              </button>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
