import clsx from "clsx";
import { type ReactNode } from "react";

export function PageHeader({ title, subtitle, actions }: { title: ReactNode; subtitle?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4 border-b border-line pb-5">
      <div className="min-w-0">
        <h1 className="font-serif text-[28px] leading-tight font-semibold">{title}</h1>
        {subtitle && <p className="mt-1 max-w-2xl text-[13.5px] text-ink-3">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

/** A small uppercase label above a group of fields. */
export function Label({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={clsx("mb-2 text-[11.5px] font-medium uppercase tracking-wide text-ink-3", className)}>{children}</p>;
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded border border-line bg-panel-2 px-1 py-px font-mono text-[10.5px] text-ink-3">{children}</kbd>;
}
