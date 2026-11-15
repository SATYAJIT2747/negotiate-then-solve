import clsx from "clsx";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { OUTCOME_DOT, OUTCOME_TEXT, type Outcome } from "../../lib/meta";

/** The pieces every conversation is drawn with (the assistant chat, a request's thread, a colleague's request):
 * what the person wrote, what came back, the steps in between, and the frame of a card. */

/** What the person wrote, on the right. ``note``: a line under it ("Read as: …", the time). */
export function YouBubble({ text, note, compact = false }: { text: ReactNode; note?: ReactNode; compact?: boolean }) {
  return (
    <div className="flex flex-col items-end gap-0.5">
      <div className={clsx("max-w-[88%] whitespace-pre-wrap rounded-lg rounded-br-sm bg-brand/12 px-3 py-2 leading-relaxed", compact ? "text-[13px]" : "text-[13.5px]")}>{text}</div>
      {note && <p className="max-w-[88%] text-[11px] text-ink-3">{note}</p>}
    </div>
  );
}

const REPLY_TONE = {
  plain: "border-line bg-panel-2/60",
  warn: "border-warn/40 bg-warn/5",
  info: "border-info/30 bg-info/5",
  ok: "border-ok/40 bg-ok/5",
  bad: "border-bad/40 bg-bad/5",
} as const;

const LABEL_TONE: Record<keyof typeof REPLY_TONE, string> = {
  plain: "text-ink-3",
  warn: "text-warn",
  info: "text-info",
  ok: "text-ok",
  bad: "text-bad",
};

/** What came back, on the left: an answer, a question back, a notice. */
export function ReplyBubble({ tone = "plain", label, icon: Icon, children, footer }: {
  tone?: keyof typeof REPLY_TONE;
  label?: ReactNode;
  icon?: LucideIcon;
  children: ReactNode;
  footer?: ReactNode;
}) {
  return (
    <div className={clsx("max-w-[92%] rounded-lg rounded-bl-sm border px-3 py-2", REPLY_TONE[tone])}>
      {label && (
        <p className={clsx("mb-1 flex items-center gap-1.5 text-[11.5px] font-medium", LABEL_TONE[tone])}>
          {Icon && <Icon size={13} />} {label}
        </p>
      )}
      <div className="whitespace-pre-wrap text-[13.5px] leading-relaxed">{children}</div>
      {footer && <div className="mt-1 text-[11px] text-ink-3">{footer}</div>}
    </div>
  );
}

/** The steps a request went through, one short line each. */
export function StepList({ steps }: { steps: { key: string | number; text: string; outcome?: Outcome | null }[] }) {
  if (!steps.length) return null;
  return (
    <ul className="space-y-1 pl-1">
      {steps.map((s) => (
        <li key={s.key} className={clsx("flex items-center gap-2 text-[12px]", s.outcome ? clsx("font-medium", OUTCOME_TEXT[s.outcome]) : "text-ink-3")}>
          <span className={clsx("size-1.5 shrink-0 rounded-full", s.outcome ? OUTCOME_DOT[s.outcome] : "bg-ink-3/50")} />
          {s.text}
        </li>
      ))}
    </ul>
  );
}

/** An assistant card in the chat: a small label, then the content; ``tone`` warn: waiting on the person. */
export function CardFrame({ label, icon: Icon, tone, action, children }: { label: string; icon?: LucideIcon; tone?: "warn"; action?: ReactNode; children: ReactNode }) {
  return (
    <div className={clsx(tone === "warn" && "rounded-lg border border-warn/40 px-3 py-2")}>
      <p className="mb-1 flex items-center gap-1 text-[11px] text-ink-3">
        {Icon && <Icon size={12} />} {label}
        {action && <span className="ml-auto">{action}</span>}
      </p>
      <div className="text-[13px] leading-relaxed">{children}</div>
    </div>
  );
}

/** A small rounded tag, clickable when ``onClick`` is given (a rule a reply cites). */
export function Tag({ children, title, onClick }: { children: ReactNode; title?: string; onClick?: () => void }) {
  const cls = "rounded border border-line bg-panel px-1.5 py-0.5 text-[11px] text-ink-2";
  return onClick ? (
    <button onClick={onClick} title={title} className={clsx(cls, "hover:border-brand/50 hover:text-brand")}>
      {children}
    </button>
  ) : (
    <span title={title} className={cls}>
      {children}
    </span>
  );
}
