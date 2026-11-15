import clsx from "clsx";
import { FlaskConical, GraduationCap, MapPin } from "lucide-react";
import { Marker } from "./ui";
import type { MouseEvent, ReactNode } from "react";
import { classTile } from "./classTile";
import type { InstanceView, TimetableEntry } from "../lib/types";

export function TimetableGrid({
  instance,
  entries,
  changed,
  onPick,
  fill = false,
  dates,
  ghosts = [],
  slotState,
  onSlot,
  dim,
  marks,
}: {
  instance: InstanceView;
  entries: TimetableEntry[];
  changed: Set<string>;
  onPick?: (e: TimetableEntry, ev: MouseEvent) => void;
  /** Stretch to the parent's size (rows share the height, no minimum width), so it keeps its proportions at any zoom. */
  fill?: boolean;
  /** Each day's date ("12 Oct"), shown above its name; ``today`` is highlighted. */
  dates?: Record<string, { label: string; today: boolean }>;
  /** Where classes would go if a pending change goes through: drawn as dashed placeholders. */
  ghosts?: { session: string; day: string; slot: number; title: string; note: string; tone?: "new" | "cancelled" | "from"; course?: string; kind?: string; detail?: string }[];
  /** Classes not where they usually are this week: a small tag on the class, and where it usually is in its tooltip. */
  marks?: Map<string, { tag: string; usually: string }>;
  /** Picking a time: which empty slots are free (green, clickable with ``onSlot``) or blocked (greyed). */
  slotState?: (day: string, slot: number) => "free" | "blocked" | null;
  onSlot?: (day: string, slot: number) => void;
  /** Classes that cannot be picked right now: faded and not clickable. */
  dim?: Set<string>;
}) {
  const { days, slots_per_day, lunch_slot } = instance.calendar;
  const cell = new Map<string, TimetableEntry[]>();
  for (const e of entries) {
    const k = `${e.day}:${e.slot}`;
    cell.set(k, [...(cell.get(k) ?? []), e]);
  }
  /** What every class shown has in common (one teacher's week, one room's) is not repeated on each card. */
  const same = (f: (e: TimetableEntry) => string) => entries.length > 1 && new Set(entries.map(f)).size === 1;
  // only the teacher of a one-teacher grid (their own timetable) is left out: rooms and sections stay on every card,
  // so a card looks the same whichever week it is in (a week whose classes happen to share a room is no reason)
  const hide = { teacher: same((e) => e.faculty), room: false, groups: false };
  const covered = new Set<string>();
  for (const e of entries) for (let q = e.slot + 1; q < e.slot + e.duration; q++) covered.add(`${e.day}:${q}`);
  /** A class longer than one slot is drawn as one card over all its rows, when the rows it runs into hold
   * nothing else (no class, no proposed move, no slot to pick): "day:slot" of the first row -> rows spanned. */
  const span = new Map<string, number>();
  const hidden = new Set<string>();
  for (const [k, list] of cell) {
    if (list.length !== 1 || list[0].duration < 2) continue;
    const e = list[0];
    const rest = Array.from({ length: e.duration - 1 }, (_, i) => e.slot + 1 + i);
    const clear = rest.every(
      (q) => q < slots_per_day && q !== lunch_slot && !cell.has(`${e.day}:${q}`)
        && !ghosts.some((g) => g.day === e.day && g.slot === q) && !slotState?.(e.day, q),
    );
    if (!clear) continue;
    span.set(k, e.duration);
    for (const q of rest) hidden.add(`${e.day}:${q}`);
  }

  return (
    <div className={fill ? "h-full" : "overflow-x-auto"}>
      <div
        className={clsx("grid", fill ? "h-full" : "min-w-[860px]")}
        style={{
          gridTemplateColumns: `${fill ? "56px" : "72px"} repeat(${days.length}, minmax(0, 1fr))`,
          gridTemplateRows: fill ? `auto repeat(${slots_per_day}, minmax(0, 1fr))` : undefined,
        }}
      >
        <div />
        {days.map((d) => (
          <div key={d} className={clsx("px-2 pb-3 text-center text-[12px] font-semibold", dates?.[d]?.today ? "text-brand" : "text-ink-3")}>
            {dates?.[d] && <span className="block text-[11px] font-normal">{dates[d].label}</span>}
            {instance.day_names[d] ?? d}
          </div>
        ))}
        {Array.from({ length: slots_per_day }, (_, s) => (
          <Row key={s}>
            <div className="pr-3 pt-2 text-right font-mono text-[11px] text-ink-3">{instance.slots[s]?.label}</div>
            {days.map((d) => {
              if (hidden.has(`${d}:${s}`)) return null; // drawn by the card above it
              const rows = span.get(`${d}:${s}`) ?? 1;
              const list = cell.get(`${d}:${s}`) ?? [];
              const ghost = ghosts.filter((g) => g.day === d && g.slot === s);
              const lunch = s === lunch_slot;
              const state = slotState?.(d, s) ?? null;
              const pickable = state === "free" && !!onSlot;
              return (
                <div
                  key={d}
                  onClick={pickable ? () => onSlot!(d, s) : undefined}
                  role={pickable ? "button" : undefined}
                  title={pickable ? `${instance.day_names[d] ?? d} ${instance.slots[s]?.label}` : undefined}
                  style={rows > 1 ? { gridRow: `span ${rows}` } : undefined}
                  className={clsx(
                    "relative border-l border-t border-line p-1",
                    fill ? "min-h-0 overflow-hidden" : "min-h-[64px]",
                    lunch && "hatch",
                    covered.has(`${d}:${s}`) && !list.length && "bg-panel-2/40",
                    state === "free" && (pickable ? "free-slot cursor-pointer" : "free-slot"),
                    state === "blocked" && !lunch && "bg-ink-3/10",
                  )}
                >
                  {lunch && !list.length && !ghost.length && <span className="absolute inset-0 grid place-items-center text-[11px] text-ink-3">Lunch</span>}
                  <div className={clsx("space-y-1", rows > 1 && "h-full")}>
                      {list.map((e, i) => (
                        <Entry key={e.session} e={e} changed={changed.has(e.session)} index={i} onPick={onPick} faded={dim?.has(e.session)} tall={rows > 1} hide={hide} mark={marks?.get(e.session)} />
                      ))}
                      {ghost.map((g) => (
                        <StateCard key={g.session} g={g} />
                      ))}
                  </div>
                </div>
              );
            })}
          </Row>
        ))}
      </div>
    </div>
  );
}

function Row({ children }: { children: ReactNode }) {
  return <>{children}</>;
}

type Hide = { teacher: boolean; room: boolean; groups: boolean };

/** A class not held this week, in its usual place: the card it always is (its icon and colour, room and section),
 * dimmed with its name struck through, and why in one short red line ("Not held · you are away"). */
type Ghost = { session: string; title: string; note: string; tone?: "new" | "cancelled" | "from"; course?: string; kind?: string };

/** A class in a state of its own this week, in the same card a class always is (its icon, colour, two lines):
 * not held (dashed red, its name struck through, why on the second line), its usual place when it moved (dashed and
 * quiet, where it went), or the new time a move being put together would give it (dashed, brand). The tag in the
 * title row says which, the same tag as everywhere else (ui/Marker). */
function StateCard({ g }: { g: Ghost }) {
  const tile = classTile(g.course ?? g.session, { faded: false });
  const Icon = g.kind === "practical" ? FlaskConical : GraduationCap;
  const state = g.tone === "cancelled" ? "notHeld" : g.tone === "from" ? "moved" : "newTime";
  const detail = g.tone === "cancelled" ? g.note.replace(/^Not held\s*·?\s*/, "") : g.note;
  const look = {
    notHeld: { border: "border-bad/60", bg: "color-mix(in oklab, var(--bad) 8%, var(--panel))", title: "text-ink-2 line-through decoration-bad decoration-[1.5px]" },
    moved: { border: "border-line", bg: "transparent", title: "text-ink-3" },
    newTime: { border: "border-brand/70", bg: "color-mix(in oklab, var(--brand) 6%, var(--panel))", title: "text-ink" },
  }[state];
  return (
    <div
      className={clsx("relative block w-full overflow-hidden rounded border border-dashed px-2 py-1.5 text-left", look.border)}
      style={{ borderLeft: `3px ${state === "moved" ? "dashed" : "solid"} ${tile.color}`, background: look.bg }}
      title={`${g.title}: ${g.note}`}
    >
      <p className="flex min-w-0 items-center gap-1 text-[11.5px] font-semibold">
        <Icon size={11} className="shrink-0" style={{ color: tile.color }} />
        <span className={clsx("min-w-0 truncate", look.title)}>{g.title}</span>
        <span className="ml-auto pl-1">
          <Marker kind={state} title={state === "notHeld" ? g.note : state === "moved" ? `Usually here; this week ${g.note.replace(/^→\s*/, "")}` : "The new time, not sent yet"} />
        </span>
      </p>
      {detail && <p className={clsx("mt-0.5 truncate text-[10.5px]", state === "notHeld" ? "text-bad" : "text-ink-3")}>{detail}</p>}
    </div>
  );
}

function Entry({ e, changed, index, onPick, faded, tall, hide, mark }: { e: TimetableEntry; changed: boolean; index: number; onPick?: (e: TimetableEntry, ev: MouseEvent) => void; faded?: boolean; tall?: boolean; hide: Hide; mark?: { tag: string; usually: string } }) {
  const tile = classTile(e.course, { changed, faded });
  const c = tile.color;
  const Icon = e.kind === "practical" ? FlaskConical : GraduationCap;
  // one line under the title: who, where and for whom, less what every card on the grid shares
  const detail = [!hide.teacher && e.faculty_name, !hide.room && e.room_name, !hide.groups && e.group_names.join(", ")]
    .filter(Boolean).join(" · ");
  return (
    <button
      data-index={index}
      onClick={(ev) => {
        ev.stopPropagation();
        onPick?.(e, ev);
      }}
      disabled={faded}
      className={clsx(
        "relative block w-full px-2 py-1.5 align-top",
        tall && "flex h-full flex-col justify-start",
        tile.className,
        mark && !changed && "outline-1 outline-offset-1 outline-info/70",
      )}
      style={tile.style}
      title={`${e.title} · ${e.faculty_name} · ${e.group_names.join(", ")} · ${e.room_name}${mark ? `\n${mark.tag === "room" ? "Another room this week" : "Moved this week"}: usually ${mark.usually}` : ""}`}
    >
      {/* the title truncates, never the markers: they sit after it at a fixed size (an icon each, said in the
          tooltip), with the length of a long class */}
      <p className="flex min-w-0 items-center gap-1 text-[11.5px] font-semibold text-ink">
        <Icon size={11} className="shrink-0" style={{ color: c }} />
        <span className="min-w-0 truncate">{e.title}</span>
        <span className="ml-auto flex shrink-0 items-center gap-1 pl-1">
          {e.extra && <Marker kind="extra" title="An extra class, this week only" />}
          {mark && (
            <Marker kind={mark.tag === "room" ? "room" : "moved"} title={`${mark.tag === "room" ? "Another room this week" : "Moved this week"}: usually ${mark.usually}`} />
          )}
          {e.duration > 1 && <span className="text-[10px] font-normal text-ink-3">{e.duration}h</span>}
        </span>
      </p>
      {detail && (
        <p className="mt-0.5 flex items-center gap-1 truncate text-[10.5px] text-ink-3">
          {!hide.room && <MapPin size={9} className="shrink-0" />}
          <span className="truncate">{detail}</span>
        </p>
      )}
    </button>
  );
}
