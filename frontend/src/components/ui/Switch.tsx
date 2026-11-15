import clsx from "clsx";

/** An on/off switch; ``label`` is read by screen readers when the text beside it is not its label. */
export function Switch({ on, onChange, label, disabled }: { on: boolean; onChange: (on: boolean) => void; label?: string; disabled?: boolean }) {
  return (
    <button
      role="switch"
      aria-checked={on}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!on)}
      className={clsx("relative h-[18px] w-8 shrink-0 rounded-full transition-colors disabled:opacity-40", on ? "bg-ok" : "bg-line")}
    >
      <span className="absolute top-[3px] size-3 rounded-full bg-panel transition-[left]" style={{ left: on ? 17 : 3 }} />
    </button>
  );
}
