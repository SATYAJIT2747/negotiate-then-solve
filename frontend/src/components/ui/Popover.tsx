import clsx from "clsx";
import { useEffect, useRef, type ReactNode } from "react";

/** Something floating at a point on the screen (a menu where a class was clicked): a click outside it or Esc
 * closes it. Kept inside the window: ``width`` and ``height`` are how much room it needs. */
export function Popover({ at, onClose, width = 240, height = 200, className, children }: {
  at: { x: number; y: number };
  onClose: () => void;
  width?: number;
  height?: number;
  className?: string;
  children: ReactNode;
}) {
  const box = useOutside<HTMLDivElement>(onClose);
  const left = Math.max(8, Math.min(at.x, window.innerWidth - width - 8));
  const top = Math.max(8, Math.min(at.y, window.innerHeight - height));
  return (
    <div ref={box} style={{ left, top, width }} className={clsx("fixed z-40 rounded-lg border border-line bg-panel p-1 shadow-lg", className)} role="menu">
      {children}
    </div>
  );
}

/** A ref for an element that ``onClose`` should follow a click outside of, or Esc (while ``active``). */
export function useOutside<T extends HTMLElement>(onClose: () => void, active = true) {
  const box = useRef<T>(null);
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    if (!active) return;
    const click = (e: MouseEvent) => !box.current?.contains(e.target as Node) && close.current();
    const key = (e: KeyboardEvent) => e.key === "Escape" && close.current();
    window.addEventListener("mousedown", click);
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("mousedown", click);
      window.removeEventListener("keydown", key);
    };
  }, [active]);
  return box;
}
