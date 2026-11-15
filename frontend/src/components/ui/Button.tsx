import clsx from "clsx";
import { Loader2, type LucideIcon } from "lucide-react";
import { type ButtonHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger" | "success";

export function Button({
  variant = "secondary",
  size = "md",
  icon: Icon,
  loading,
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: "sm" | "md"; icon?: LucideIcon; loading?: boolean }) {
  return (
    <button
      {...rest}
      disabled={rest.disabled || loading}
      className={clsx(
        "inline-flex select-none items-center justify-center gap-2 rounded-md font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        size === "sm" ? "h-8 px-3 text-[13px]" : "h-9 px-3.5 text-[13.5px]",
        variant === "primary" && "bg-brand text-panel hover:bg-brand/90",
        variant === "secondary" && "border border-line bg-panel text-ink hover:bg-panel-2",
        variant === "ghost" && "text-ink-2 hover:bg-panel-2 hover:text-ink",
        variant === "danger" && "border border-bad/30 text-bad hover:bg-bad/8",
        variant === "success" && "bg-ok text-panel hover:bg-ok/90",
        className,
      )}
    >
      {loading ? <Loader2 size={15} className="animate-spin" /> : Icon ? <Icon size={15} /> : null}
      {children}
    </button>
  );
}
