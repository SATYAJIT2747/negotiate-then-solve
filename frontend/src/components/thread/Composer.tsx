import clsx from "clsx";
import { Send, X } from "lucide-react";
import type { ReactNode, RefObject } from "react";
import { Button } from "../ui";

/** The box under every conversation: Enter sends, Shift+Enter starts a new line. What sits above it (classes
 * attached, a reading to confirm, what is being answered) and the small controls under it are the caller's. */
export function Composer({ value, onChange, onSend, sending, inputRef, placeholder, rows = 2, onFocus, above, footerLeft, footerRight, error }: {
  value: string;
  onChange: (text: string) => void;
  onSend: () => void;
  sending: boolean;
  inputRef?: RefObject<HTMLTextAreaElement | null>;
  placeholder: string;
  rows?: number;
  onFocus?: () => void;
  above?: ReactNode;
  footerLeft?: ReactNode;
  footerRight?: ReactNode;
  error?: string | null;
}) {
  return (
    <div>
      {above}
      <div className="flex items-end gap-2 rounded-xl border border-line bg-panel px-3 py-2 focus-within:border-brand/60">
        <textarea
          ref={inputRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              if (value.trim() && !sending) onSend();
            }
          }}
          onFocus={onFocus}
          rows={rows}
          placeholder={placeholder}
          title="Enter to send · Shift+Enter for a new line"
          className="max-h-40 min-h-[2.5rem] flex-1 resize-none bg-transparent py-1 text-[13.5px] leading-relaxed outline-none placeholder:text-ink-3"
        />
        <Button variant="primary" size="sm" icon={Send} loading={sending} disabled={!value.trim()} onClick={onSend} aria-label="Send" />
      </div>
      {(footerLeft || footerRight) && (
        <div className="mt-1 flex flex-wrap items-center justify-between gap-2 px-1 text-[11px] text-ink-3">
          <div className="flex items-center gap-2">{footerLeft}</div>
          {footerRight}
        </div>
      )}
      {error && <p className="mt-1 text-[12.5px] text-bad">{error}</p>}
    </div>
  );
}

/** Classes (or anything else) attached to the message being written, each with a way to take it off. */
export function Chips({ items, onRemove }: { items: { key: string; label: string }[]; onRemove: (key: string) => void }) {
  if (!items.length) return null;
  return (
    <div className="mb-2 flex flex-wrap gap-1">
      {items.map((a) => (
        <span key={a.key} className="flex items-center gap-1 rounded border border-brand/40 bg-brand/10 py-0.5 pl-1.5 pr-1 text-[11.5px]">
          {a.label}
          <button onClick={() => onRemove(a.key)} className="text-ink-3 hover:text-ink" aria-label={`Remove ${a.label}`}>
            <X size={11} />
          </button>
        </span>
      ))}
    </div>
  );
}

/** "Hold make-up classes": for a teacher's absence, whether the classes missed move or are cancelled. */
export function MakeupToggle({ checked, onChange, className }: { checked: boolean; onChange: (on: boolean) => void; className?: string }) {
  return (
    <label
      className={clsx("flex cursor-pointer items-center gap-1.5", className)}
      title="If you are away in some weeks: ticked, the classes you miss move to other times that week; left empty, they are cancelled"
    >
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="accent-brand" />
      Hold make-up classes
    </label>
  );
}
