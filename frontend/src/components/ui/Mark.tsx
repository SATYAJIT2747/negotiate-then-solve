import clsx from "clsx";

/** The mark: a timetable grid with one cell settled. */
export function Mark({ size = 22, className }: { size?: number; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" width={size} height={size} className={clsx("shrink-0 text-brand", className)} aria-hidden>
      <rect x="2.5" y="3.5" width="19" height="17" rx="2.5" fill="none" stroke="currentColor" strokeWidth="1.7" />
      <path d="M2.5 8.5h19M8.8 8.5v12M15.2 8.5v12" stroke="currentColor" strokeWidth="1.4" />
      <rect x="9.6" y="9.3" width="4.8" height="5.1" rx="0.8" fill="currentColor" />
    </svg>
  );
}
