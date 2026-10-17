import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { useEffect } from "react";
import CaseDetail from "../pages/CaseDetail";

/** How the system handled a request (lifecycle, parse, rules, solver, negotiation, fairness), in a panel
 * that slides over the conversation instead of a page of its own. */
export function DetailsDrawer({ id, onClose }: { id: string | null; onClose: () => void }) {
  useEffect(() => {
    if (!id) return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [id, onClose]);

  return (
    <AnimatePresence>
      {id && (
        <motion.div className="fixed inset-0 z-40 bg-ink/25" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }} onClick={onClose}>
          <motion.aside
            initial={{ x: 40, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 40, opacity: 0 }}
            transition={{ type: "tween", duration: 0.18 }}
            onClick={(e) => e.stopPropagation()}
            className="absolute inset-y-0 right-0 flex w-full max-w-[860px] flex-col border-l border-line bg-bg shadow-xl"
            role="dialog"
            aria-label={`Details of ${id}`}
          >
            <div className="flex items-center justify-between gap-3 border-b border-line bg-panel px-5 py-3">
              <div>
                <p className="text-[14px] font-semibold">Details</p>
                <p className="text-[12px] text-ink-3">How the system handled {id}: the rules it checked, how it was solved and negotiated</p>
              </div>
              <button onClick={onClose} className="grid size-8 place-items-center rounded-md text-ink-3 hover:bg-panel-2 hover:text-ink" aria-label="Close details">
                <X size={16} />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-5">
              <CaseDetail id={id} embedded />
            </div>
          </motion.aside>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
