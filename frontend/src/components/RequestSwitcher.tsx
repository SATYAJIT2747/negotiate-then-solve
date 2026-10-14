import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, ChevronDown, Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ago, STATUS, TONE_CLASS } from "../lib/meta";
import type { CaseSummary, Status } from "../lib/types";

/** Still on its way: being worked on, or waiting on someone other than the sender. */
const WAITING: Status[] = ["received", "classified", "policy_checked", "compiled", "solved", "negotiating", "fairness_audited", "awaiting_approval", "escalated"];

export type Group = "needs" | "progress" | "done";

export function groupOf(c: CaseSummary): Group {
  if (c.status === "clarification_requested") return "needs";
  if (c.running || WAITING.includes(c.status) || (c.status === "forwarded" && !c.handled)) return "progress";
  return "done";
}

const GROUP_LABEL: Record<Group, string> = { needs: "Needs you", progress: "In progress", done: "Done" };

/** The sender's requests in a dropdown, grouped by what they are waiting on; picking one opens its thread. */
export function RequestSwitcher({ cases, selected, onSelect }: { cases: CaseSummary[]; selected: string | null; onSelect: (id: string | null) => void }) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const click = (e: MouseEvent) => !box.current?.contains(e.target as Node) && setOpen(false);
    const key = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("mousedown", click);
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("mousedown", click);
      window.removeEventListener("keydown", key);
    };
  }, [open]);

  const groups = useMemo(() => {
    const ql = q.trim().toLowerCase();
    const out: Record<Group, CaseSummary[]> = { needs: [], progress: [], done: [] };
    for (const c of cases) if (!ql || `${c.id} ${c.text}`.toLowerCase().includes(ql)) out[groupOf(c)].push(c);
    return out;
  }, [cases, q]);
  const needs = cases.filter((c) => groupOf(c) === "needs").length;
  const current = cases.find((c) => c.id === selected);

  return (
    <div className="relative min-w-0 flex-1" ref={box}>
      <button
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full min-w-0 items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-panel-2"
      >
        <span className="min-w-0 flex-1 truncate text-[13.5px] font-semibold">{current ? current.text : "Requests"}</span>
        {needs > 0 && <span className="shrink-0 rounded bg-warn/15 px-1.5 text-[11px] font-semibold text-warn">{needs} needs you</span>}
        <ChevronDown size={15} className={clsx("shrink-0 text-ink-3 transition-transform", open && "rotate-180")} />
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.12 }}
            className="absolute left-0 right-0 top-full z-30 mt-1 overflow-hidden rounded-lg border border-line bg-panel shadow-lg"
          >
            <div className="flex items-center gap-2 border-b border-line px-3">
              <Search size={14} className="text-ink-3" />
              <input
                autoFocus
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Find a request"
                className="h-9 flex-1 bg-transparent text-[13px] outline-none placeholder:text-ink-3"
              />
            </div>
            <div className="max-h-[360px] overflow-y-auto py-1">
              {(["needs", "progress", "done"] as Group[]).map(
                (g) =>
                  groups[g].length > 0 && (
                    <div key={g}>
                      <p className="px-3 pb-1 pt-2 text-[11px] font-medium uppercase tracking-wide text-ink-3">{GROUP_LABEL[g]}</p>
                      {groups[g].slice(0, g === "done" ? 8 : 20).map((c) => {
                        const s = STATUS[c.status];
                        return (
                          <button
                            key={c.id}
                            onClick={() => {
                              onSelect(c.id);
                              setOpen(false);
                            }}
                            className={clsx("flex w-full items-start gap-2.5 px-3 py-2 text-left hover:bg-panel-2", c.id === selected && "bg-panel-2")}
                          >
                            <s.icon size={14} className={clsx("mt-0.5 shrink-0", TONE_CLASS[s.tone].text)} />
                            <span className="min-w-0 flex-1">
                              <span className="block truncate text-[13px]">{c.text}</span>
                              <span className="block text-[11.5px] text-ink-3">
                                {c.running ? "Working on it…" : s.label} · {ago(c.received_at)}
                              </span>
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  ),
              )}
              {!cases.length && <p className="px-3 py-6 text-center text-[13px] text-ink-3">No requests yet. Write one below.</p>}
              {cases.length > 0 && !groups.needs.length && !groups.progress.length && !groups.done.length && (
                <p className="px-3 py-6 text-center text-[13px] text-ink-3">Nothing matches “{q}”.</p>
              )}
            </div>
            <div className="flex items-center justify-between border-t border-line px-3 py-2 text-[12.5px]">
              <button
                onClick={() => {
                  onSelect(null);
                  setOpen(false);
                }}
                className="text-ink-2 hover:text-ink"
              >
                Start a new one
              </button>
              <Link to="/requests" className="flex items-center gap-1 text-brand hover:underline">
                See all requests <ArrowRight size={13} />
              </Link>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
