import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, CalendarClock, ChevronDown, MessagesSquare, Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ago, STATUS, TONE_CLASS } from "../lib/meta";
import type { CaseSummary, FeedItem, Status } from "../lib/types";
import { Options } from "./RequestThread";

/** Everything one person is told, as conversations: one per request of theirs, and one per colleague's
 * request that asks something of them or changed their timetable. Built from their requests and the
 * inbox feed, so the home panel replaces the Inbox for faculty and class representatives. */
export interface Conversation {
  key: string;
  own: boolean;
  case: string | null;
  title: string;
  /** The latest thing that happened, in a few words. */
  latest: string;
  at: string;
  needs: boolean;
  unread: boolean;
  status?: Status;
  running?: boolean;
  handled?: boolean;
  feed: FeedItem[];
}

export type Group = "needs" | "progress" | "updates" | "done";

/** Still on its way: being worked on, or waiting on someone other than the sender. */
const WAITING: Status[] = ["received", "classified", "policy_checked", "compiled", "solved", "negotiating", "fairness_audited", "awaiting_approval", "escalated"];

export function groupOf(c: Conversation): Group {
  if (c.needs) return "needs";
  if (!c.own) return "updates";
  if (c.running || (c.status && WAITING.includes(c.status)) || (c.status === "forwarded" && !c.handled)) return "progress";
  return "done";
}

const GROUP_LABEL: Record<Group, string> = { needs: "Needs you", progress: "Your requests: in progress", updates: "From colleagues", done: "Your requests: done" };

export function buildConversations(cases: CaseSummary[], feed: FeedItem[]): Conversation[] {
  const byCase = new Map<string, FeedItem[]>();
  for (const f of feed) if (f.case) byCase.set(f.case, [...(byCase.get(f.case) ?? []), f]);
  const out: Conversation[] = cases.map((c) => {
    const items = byCase.get(c.id) ?? [];
    return {
      key: c.id,
      own: true,
      case: c.id,
      title: c.text,
      latest: c.running ? "Working on it…" : STATUS[c.status].label,
      at: items[0]?.at ?? c.received_at,
      needs: c.status === "clarification_requested" || items.some((f) => f.needs_you),
      unread: items.some((f) => f.unread),
      status: c.status,
      running: c.running,
      handled: c.handled,
      feed: items,
    };
  });
  const own = new Set(cases.map((c) => c.id));
  const others = new Map<string, FeedItem[]>();
  for (const f of feed) {
    if (f.kind === "action" || (f.case && own.has(f.case))) continue; // approvals and decisions stay on their pages
    const key = f.case ?? f.id;
    others.set(key, [...(others.get(key) ?? []), f]);
  }
  for (const [key, items] of others) {
    const top = items[0]; // the feed is newest first
    out.push({
      key: `other:${key}`,
      own: false,
      case: top.case,
      title: top.request ?? top.title,
      latest: top.title,
      at: top.at,
      needs: items.some((f) => f.needs_you),
      unread: items.some((f) => f.unread),
      feed: items,
    });
  }
  return out.sort((a, b) => b.at.localeCompare(a.at));
}

/** The conversations in a dropdown, grouped by what they wait on; picking one opens it. */
export function ConversationSwitcher({ items, selected, onSelect }: { items: Conversation[]; selected: string | null; onSelect: (key: string | null) => void }) {
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
    const out: Record<Group, Conversation[]> = { needs: [], progress: [], updates: [], done: [] };
    for (const c of items) if (!ql || `${c.case ?? ""} ${c.title} ${c.latest}`.toLowerCase().includes(ql)) out[groupOf(c)].push(c);
    return out;
  }, [items, q]);
  const needs = items.filter((c) => c.needs).length;
  const unread = items.filter((c) => c.unread && !c.needs).length;
  const current = items.find((c) => c.key === selected);

  return (
    <div className="relative min-w-0 flex-1" ref={box}>
      <button onClick={() => setOpen(!open)} aria-expanded={open} className="flex w-full min-w-0 items-center gap-2 rounded-md px-2 py-1.5 text-left hover:bg-panel-2">
        <span className="min-w-0 flex-1 truncate text-[13.5px] font-semibold">{current ? current.title : "Requests and messages"}</span>
        {needs > 0 ? (
          <span className="shrink-0 rounded bg-warn/15 px-1.5 text-[11px] font-semibold text-warn">{needs} needs you</span>
        ) : (
          unread > 0 && <span className="shrink-0 rounded bg-brand px-1.5 text-[11px] font-semibold text-panel">{unread} new</span>
        )}
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
              <input autoFocus value={q} onChange={(e) => setQ(e.target.value)} placeholder="Find a request or message" className="h-9 flex-1 bg-transparent text-[13px] outline-none placeholder:text-ink-3" />
            </div>
            <div className="max-h-[400px] overflow-y-auto py-1">
              {(["needs", "progress", "updates", "done"] as Group[]).map(
                (g) =>
                  groups[g].length > 0 && (
                    <div key={g}>
                      <p className="px-3 pb-1 pt-2 text-[11px] font-medium uppercase tracking-wide text-ink-3">{GROUP_LABEL[g]}</p>
                      {groups[g].slice(0, g === "done" || g === "updates" ? 8 : 20).map((c) => {
                        const s = c.status ? STATUS[c.status] : null;
                        const Icon = s ? s.icon : c.feed.some((f) => f.kind === "negotiation") ? MessagesSquare : CalendarClock;
                        return (
                          <button
                            key={c.key}
                            onClick={() => {
                              onSelect(c.key);
                              setOpen(false);
                            }}
                            className={clsx("flex w-full items-start gap-2.5 px-3 py-2 text-left hover:bg-panel-2", c.key === selected && "bg-panel-2")}
                          >
                            <Icon size={14} className={clsx("mt-0.5 shrink-0", s ? TONE_CLASS[s.tone].text : c.needs ? "text-warn" : "text-info")} />
                            <span className="min-w-0 flex-1">
                              <span className={clsx("block truncate text-[13px]", c.unread && "font-semibold")}>{c.title}</span>
                              <span className="block truncate text-[11.5px] text-ink-3">
                                {c.latest} · {ago(c.at)}
                              </span>
                            </span>
                            {c.unread && <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-brand" />}
                          </button>
                        );
                      })}
                    </div>
                  ),
              )}
              {!items.length && <p className="px-3 py-6 text-center text-[13px] text-ink-3">Nothing yet. Write a request below.</p>}
              {items.length > 0 && !Object.values(groups).some((g) => g.length) && <p className="px-3 py-6 text-center text-[13px] text-ink-3">Nothing matches “{q}”.</p>}
            </div>
            <div className="flex items-center justify-between border-t border-line px-3 py-2 text-[12.5px]">
              <button
                onClick={() => {
                  onSelect(null);
                  setOpen(false);
                }}
                className="text-ink-2 hover:text-ink"
              >
                Start a new request
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

/** A colleague's request as it reached this person: what was asked, the options sent to them, and how
 * it changed their timetable. */
export function ColleagueThread({ c, onChanged }: { c: Conversation; onChanged: () => void }) {
  const items = [...c.feed].reverse(); // oldest first, like a conversation
  return (
    <div className="space-y-3">
      {c.feed[0]?.request && (
        <div className="max-w-[85%] rounded-lg rounded-bl-sm border border-line bg-panel-2/60 px-3 py-2">
          <p className="text-[11px] text-ink-3">A colleague's request</p>
          <p className="mt-0.5 whitespace-pre-wrap text-[13.5px] leading-relaxed">“{c.feed[0].request}”</p>
        </div>
      )}
      {items.map((f) =>
        f.kind === "negotiation" && f.message ? (
          <Options key={f.id} item={f.message} onDone={onChanged} />
        ) : (
          <div key={f.id} className="max-w-[92%] rounded-lg rounded-bl-sm border border-info/30 bg-info/5 px-3 py-2">
            <p className="text-[11.5px] font-medium text-info">{f.title}</p>
            {f.text && <p className="mt-0.5 whitespace-pre-line text-[13.5px] leading-relaxed">{f.text}</p>}
            <p className="mt-1 text-[11px] text-ink-3">
              {f.from_name ? `${f.from_name} · ` : ""}
              {ago(f.at)}
              {f.version != null ? ` · timetable version ${f.version}` : ""}
            </p>
          </div>
        ),
      )}
    </div>
  );
}
