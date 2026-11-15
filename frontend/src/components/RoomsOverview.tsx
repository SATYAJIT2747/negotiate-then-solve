import clsx from "clsx";
import { colorFor } from "../lib/meta";
import type { InstanceView, Timetable, TimetableEntry } from "../lib/types";

type Room = InstanceView["rooms"][number];
type Closed = NonNullable<Timetable["closed"]>[number];

/** Every room someone looks after (the lab in-charge's labs, the exam cell's rooms) for one week, at a glance:
 * a row per room, each day split into its hours, each hour booked (the class's colour), closed or free.
 * Clicking a room opens its own week. */
export function RoomsOverview({
  instance,
  rooms,
  entries,
  closed,
  dates,
  onOpen,
}: {
  instance: InstanceView;
  rooms: Room[];
  entries: TimetableEntry[];
  closed: Closed[];
  dates?: Record<string, { label: string; today: boolean }>;
  onOpen: (room: string) => void;
}) {
  const { days, slots_per_day, lunch_slot } = instance.calendar;
  const at = new Map<string, TimetableEntry>(); // "room:day:slot" -> the class there
  for (const e of entries) for (let q = e.slot; q < e.slot + e.duration; q++) at.set(`${e.room}:${e.day}:${q}`, e);
  const isClosed = (room: string, day: string, slot: number) =>
    closed.find((c) => c.room === room && (!c.days || c.days.includes(day)) && (!c.slots || c.slots.includes(slot)));
  const teaching = Array.from({ length: slots_per_day }, (_, s) => s).filter((s) => s !== lunch_slot);

  return (
    <div className="h-full overflow-auto">
      <table className="w-full min-w-[760px] table-fixed border-separate border-spacing-0 text-[12.5px]">
        <thead>
          <tr>
            <th className="w-44 pb-3 pl-2 text-left text-[12px] font-semibold text-ink-3">Room</th>
            {days.map((d) => (
              <th key={d} className={clsx("px-2 pb-3 text-center text-[12px] font-semibold", dates?.[d]?.today ? "text-brand" : "text-ink-3")}>
                {dates?.[d] && <span className="block text-[11px] font-normal">{dates[d].label}</span>}
                {instance.day_names[d] ?? d}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rooms.map((r) => {
            const booked = days.reduce((n, d) => n + teaching.filter((s) => at.has(`${r.id}:${d}:${s}`)).length, 0);
            const shut = closed.filter((c) => c.room === r.id);
            return (
              <tr key={r.id} className="group">
                <td className="border-t border-line py-2.5 pl-2 pr-3 align-middle">
                  <button onClick={() => onOpen(r.id)} className="text-left">
                    <span className="block font-semibold text-ink group-hover:text-brand">{r.name}</span>
                    <span className="block text-[11.5px] text-ink-3">
                      {shut.length ? (
                        <span className="text-bad">closed{shut.some((c) => !c.days && !c.slots) ? " all week" : " in part"}</span>
                      ) : (
                        <>
                          {booked} of {days.length * teaching.length} hours booked
                        </>
                      )}
                      {" · "}
                      {r.type === "lab" ? "lab" : `${r.capacity} seats`}
                    </span>
                  </button>
                </td>
                {days.map((d) => (
                  <td key={d} className="border-l border-t border-line px-1.5 py-2.5 align-middle">
                    <div className="flex gap-[3px]">
                      {Array.from({ length: slots_per_day }, (_, s) => {
                        const e = at.get(`${r.id}:${d}:${s}`);
                        const c = isClosed(r.id, d, s);
                        const time = instance.slots[s]?.label ?? `${s}`;
                        const tip = s === lunch_slot && !e ? `${time}: lunch` : c ? `${time}: closed${c.text ? ` (“${c.text}”)` : ""}` : e
                          ? `${time}: ${e.title} · ${e.faculty_name} · ${e.group_names.join(", ")}` : `${time}: free`;
                        return (
                          <span
                            key={s}
                            title={tip}
                            onClick={() => onOpen(r.id)}
                            className={clsx(
                              "h-6 flex-1 cursor-pointer rounded-[3px]",
                              s === lunch_slot && !e ? "hatch opacity-60" : c ? "hatch-bad" : !e && "bg-panel-2",
                            )}
                            style={e && !c ? { background: colorFor(e.course) } : undefined}
                          />
                        );
                      })}
                    </div>
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 px-2 text-[11.5px] text-ink-3">
        <span className="flex items-center gap-1.5"><span className="inline-block size-3 rounded-[3px] bg-brand" /> booked (hover for the class)</span>
        <span className="flex items-center gap-1.5"><span className="inline-block size-3 rounded-[3px] bg-panel-2" /> free</span>
        <span className="flex items-center gap-1.5"><span className="hatch inline-block size-3 rounded-[3px]" /> lunch</span>
        <span className="flex items-center gap-1.5"><span className="hatch-bad inline-block size-3 rounded-[3px]" /> closed</span>
        <span>Click a room for its week.</span>
      </p>
    </div>
  );
}
