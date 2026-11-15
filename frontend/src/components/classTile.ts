import clsx from "clsx";
import type { CSSProperties } from "react";
import { colorFor } from "../lib/meta";

/** How a class looks wherever a grid draws it (the week's timetable, the semester plan): its course's colour as
 * a bar on the left and a tint behind, outlined when ``changed``, faded when it cannot be picked, dashed when it
 * is only optional. ``tint``: how strong the tint is, in percent. */
export function classTile(course: string, { changed = false, faded = false, dashed = false, tint = 10 } = {}): { className: string; style: CSSProperties; color: string } {
  const color = colorFor(course);
  return {
    color,
    className: clsx(
      "overflow-hidden rounded text-left transition-[colors,opacity]",
      dashed ? "border border-dashed" : "border-l-[3px]",
      changed && "outline-2 outline-offset-1 outline-brand",
      faded && "opacity-30",
    ),
    style: { borderColor: color, background: dashed ? "var(--panel)" : `color-mix(in oklab, ${color} ${tint}%, var(--panel))` },
  };
}
