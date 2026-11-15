import clsx from "clsx";
import { type LucideIcon } from "lucide-react";
import { type ReactNode } from "react";

export function Card({ className, children, hover }: { className?: string; children: ReactNode; hover?: boolean }) {
  return (
    <div className={clsx("min-w-0 rounded-lg border border-line bg-panel", hover && "transition-colors hover:border-ink-3/60", className)}>
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  subtitle,
  icon: Icon,
  action,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  icon?: LucideIcon;
  action?: ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-line px-5 py-3.5">
      <div className="min-w-0">
        <h3 className="flex items-center gap-2 text-[14px] font-semibold">
          {Icon && <Icon size={15} className="shrink-0 text-ink-3" />}
          {title}
        </h3>
        {subtitle && <p className="mt-0.5 text-[12.5px] text-ink-3">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}
