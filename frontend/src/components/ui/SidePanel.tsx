import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { useEffect, type ReactNode } from "react";

/** A panel that slides in from the right over the page (request details, a rule's text): Esc, the backdrop
 * or the close button shut it. ``open`` false leaves it out of the page. */
export function SidePanel({ open, title, subtitle, onClose, width = 860, children }: {
  open: boolean;
  title: ReactNode;
  subtitle?: ReactNode;
  onClose: () => void;
  /** the widest it gets, in px */
  width?: number;
  children: ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open && (
        <motion.div className="fixed inset-0 z-40 bg-ink/25" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }} onClick={onClose}>
          <motion.aside
            initial={{ x: 40, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 40, opacity: 0 }}
            transition={{ type: "tween", duration: 0.18 }}
            onClick={(e) => e.stopPropagation()}
            style={{ maxWidth: width }}
            className="absolute inset-y-0 right-0 flex w-full flex-col border-l border-line bg-bg shadow-xl"
            role="dialog"
            aria-label={typeof title === "string" ? title : undefined}
          >
            <div className="flex items-center justify-between gap-3 border-b border-line bg-panel px-5 py-3">
              <div className="min-w-0">
                <p className="truncate text-[14px] font-semibold">{title}</p>
                {subtitle && <p className="text-[12px] text-ink-3">{subtitle}</p>}
              </div>
              <button onClick={onClose} className="grid size-8 shrink-0 place-items-center rounded-md text-ink-3 hover:bg-panel-2 hover:text-ink" aria-label="Close">
                <X size={16} />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-5">{children}</div>
          </motion.aside>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
