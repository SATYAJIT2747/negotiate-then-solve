import clsx from "clsx";
import { type LucideIcon } from "lucide-react";

export function Tabs<T extends string>({
  tabs,
  value,
  onChange,
}: {
  tabs: { id: T; label: string; icon?: LucideIcon; count?: number }[];
  value: T;
  onChange: (t: T) => void;
}) {
  return (
    <div className="inline-flex max-w-full gap-0.5 overflow-x-auto rounded-md border border-line bg-panel-2 p-0.5">
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={clsx(
            "flex shrink-0 items-center gap-1.5 rounded px-2.5 py-1 text-[13px] font-medium transition-colors",
            value === t.id ? "bg-panel text-ink shadow-[0_0_0_1px_var(--line)]" : "text-ink-3 hover:text-ink",
          )}
        >
          {t.icon && <t.icon size={14} />}
          {t.label}
          {t.count !== undefined && t.count > 0 && <span className="text-[11.5px] tabular-nums text-ink-3">{t.count}</span>}
        </button>
      ))}
    </div>
  );
}
