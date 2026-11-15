import clsx from "clsx";
import { type LucideIcon } from "lucide-react";
import { type ReactNode } from "react";
import { TONE_CLASS, type Tone } from "../../lib/meta";

export function Stat({ label, value, format, hint }: { label: string; value: number; format?: (v: number) => string; hint?: ReactNode; icon?: LucideIcon; tone?: Tone; delay?: number }) {
  return (
    <div className="min-w-0">
      <p className="text-[12.5px] text-ink-3">{label}</p>
      <p className="mt-1 font-serif text-[28px] leading-none font-semibold tabular-nums">{format ? format(value) : value}</p>
      {hint && <p className="mt-1.5 text-[12px] text-ink-3">{hint}</p>}
    </div>
  );
}

export function Meter({ value, max = 1, tone = "brand", className }: { value: number; max?: number; tone?: Tone; className?: string }) {
  const p = Math.max(0, Math.min(1, max ? value / max : 0));
  return (
    <div className={clsx("h-1.5 overflow-hidden rounded-full bg-panel-2", className)}>
      <div className={clsx("h-full rounded-full transition-[width] duration-500", TONE_CLASS[tone].dot)} style={{ width: `${p * 100}%` }} />
    </div>
  );
}
