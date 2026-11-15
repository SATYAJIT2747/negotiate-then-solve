import clsx from "clsx";
import { ArrowLeftRight, CalendarClock, CircleSlash, DoorOpen, Plus, type LucideIcon } from "lucide-react";
import { TONE_CLASS, type Tone } from "../../lib/meta";

/** What can be different about a class this week, one look for all of them wherever they are shown (the timetable's
 * cards, the chat's cards): a tone and an icon each, the words in the tooltip or, with ``label``, beside the icon. */
export type MarkerKind = "notHeld" | "moved" | "room" | "extra" | "newTime";

export const MARKERS: Record<MarkerKind, { tone: Tone; icon: LucideIcon; label: string }> = {
  notHeld: { tone: "bad", icon: CircleSlash, label: "Not held" },
  moved: { tone: "info", icon: ArrowLeftRight, label: "Moved" },
  room: { tone: "info", icon: DoorOpen, label: "Other room" },
  extra: { tone: "brand", icon: Plus, label: "Extra" },
  newTime: { tone: "brand", icon: CalendarClock, label: "New time" },
};

/** A small tag: the icon in its tone, ``label`` to say it in words too; ``title``: the tooltip (what it means here). */
export function Marker({ kind, label = false, title }: { kind: MarkerKind; label?: boolean; title?: string }) {
  const m = MARKERS[kind];
  const t = TONE_CLASS[m.tone];
  return (
    <span
      title={title ?? m.label}
      className={clsx("inline-flex h-4 shrink-0 items-center gap-0.5 rounded font-semibold", t.bg, t.text, label ? "px-1 text-[9.5px]" : "w-4 justify-center")}
    >
      <m.icon size={10} strokeWidth={2.5} />
      {label && m.label}
    </span>
  );
}
