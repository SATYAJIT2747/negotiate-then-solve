import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronRight } from "lucide-react";
import { useState, type ReactNode } from "react";

export function Collapse({ title, children, defaultOpen = false }: { title: ReactNode; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-lg border border-line bg-panel">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 px-4 py-3 text-left text-[13.5px] font-medium">
        <ChevronRight size={15} className={clsx("text-ink-3 transition-transform", open && "rotate-90")} />
        {title}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0 }} animate={{ height: "auto" }} exit={{ height: 0 }} transition={{ duration: 0.18 }} className="overflow-hidden">
            <div className="border-t border-line px-4 py-4">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
