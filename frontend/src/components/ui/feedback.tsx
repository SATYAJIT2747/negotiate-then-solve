import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { type LucideIcon } from "lucide-react";
import { useEffect, useRef, type ReactNode } from "react";
import { TONE_CLASS, type Tone } from "../../lib/meta";

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("skeleton rounded-md", className)} />;
}

export function EmptyState({ icon: Icon, title, text, action }: { icon: LucideIcon; title: string; text?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center px-6 py-12 text-center">
      <Icon size={22} className="text-ink-3" />
      <p className="mt-3 font-medium">{title}</p>
      {text && <p className="mt-1 max-w-sm text-[13px] text-ink-3">{text}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function Toast({ message, tone = "ok", onDone }: { message: string | null; tone?: Tone; onDone: () => void }) {
  // Callers pass an inline onDone, which changes on every render; keying the timer on it
  // restarted the countdown each time a polling page re-rendered, so the toast never left.
  const done = useRef(onDone);
  done.current = onDone;
  useEffect(() => {
    if (!message) return;
    const id = window.setTimeout(() => done.current(), 3200);
    return () => window.clearTimeout(id);
  }, [message]);
  return (
    <AnimatePresence>
      {message && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="fixed bottom-6 left-1/2 z-50 -translate-x-1/2">
          <div className="flex items-center gap-2.5 rounded-md bg-ink px-4 py-2.5 text-[13.5px] text-bg shadow-lg">
            <span className={clsx("size-1.5 rounded-full", TONE_CLASS[tone].dot)} />
            {message}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
