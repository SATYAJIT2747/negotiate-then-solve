import clsx from "clsx";
import { CalendarClock, MessagesSquare } from "lucide-react";
import { useMemo } from "react";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useApi } from "../lib/hooks";
import { ago, OUTCOME_DOT, OUTCOME_TEXT, outcomeOf, PLAIN_STATUS, splitLabel, splitOutcome, STATUS, TONE_CLASS, type Outcome } from "../lib/meta";
import type { CaseSummary, ChangeView, FeedItem, Status } from "../lib/types";
import { ChangeCard, weeksLabel } from "./ChangeCard";
import { Options } from "./RequestThread";
import { ReplyBubble } from "./thread/Bubble";

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
  /** How it stands, for its colour: a message split into requests stands where its parts stand. */
  outcome?: Outcome;
  /** Only changes to this person's timetable: no request of theirs, nothing to answer. */
  changeOnly?: boolean;
  /** A message split into several requests: the keys of those requests, listed under it. */
  parts?: string[];
  /** One of the requests a message was split into: the message's key (listed under it, not on its own). */
  partOf?: string;
  feed: FeedItem[];
}

export type Group = "needs" | "progress" | "updates" | "done";

/** Still on its way: being worked on, or waiting on someone other than the sender. */
const WAITING: Status[] = ["received", "classified", "policy_checked", "compiled", "solved", "negotiating", "fairness_audited", "awaiting_approval", "escalated"];

export function groupOf(c: Conversation): Group {
  if (c.needs) return "needs";
  if (!c.own) return "updates";
  if (c.running || c.outcome === "pending" || (c.status && WAITING.includes(c.status)) || (c.status === "forwarded" && !c.handled)) return "progress";
  return "done";
}

const GROUP_LABEL: Record<Group, string> = { needs: "Needs you", progress: "Your requests: in progress", updates: "From colleagues", done: "Your requests: done" };

export function buildConversations(cases: CaseSummary[], feed: FeedItem[]): Conversation[] {
  const byCase = new Map<string, FeedItem[]>();
  for (const f of feed) if (f.case) byCase.set(f.case, [...(byCase.get(f.case) ?? []), f]);
  const mine = new Set(cases.map((c) => c.id));
  const partOf = new Map<string, string>();
  for (const c of cases) for (const p of c.parts ?? []) if (mine.has(p)) partOf.set(p, c.id);
  const out: Conversation[] = cases.map((c) => {
    const items = byCase.get(c.id) ?? [];
    return {
      key: c.id,
      own: true,
      case: c.id,
      title: c.text,
      latest: c.running ? "Working on it…" : c.parts_status?.length ? splitLabel(c.parts_status) : PLAIN_STATUS[c.status].label,
      outcome: c.running ? "progress" : c.parts_status?.length ? splitOutcome(c.parts_status) : outcomeOf(c.status),
      at: items[0]?.at ?? c.received_at,
      needs: c.status === "clarification_requested" || items.some((f) => f.needs_you),
      unread: items.some((f) => f.unread),
      status: c.status,
      running: c.running,
      handled: c.handled,
      parts: (c.parts ?? []).filter((p) => mine.has(p)),
      partOf: partOf.get(c.id),
      feed: items,
    };
  });
  // a split message stands for its parts in the list: it needs you, is unread and is as recent as any of them
  const byKey = new Map(out.map((c) => [c.key, c]));
  for (const c of out) {
    const parts = (c.parts ?? []).map((p) => byKey.get(p)).filter((p): p is Conversation => !!p);
    if (!parts.length) continue;
    c.needs ||= parts.some((p) => p.needs);
    c.unread ||= parts.some((p) => p.unread);
    for (const p of parts) if (p.at > c.at) c.at = p.at;
  }
  const others = new Map<string, FeedItem[]>();
  for (const f of feed) {
    if (f.kind === "action" || (f.case && mine.has(f.case))) continue; // approvals and decisions stay on their pages
    const key = f.case ?? f.id;
    others.set(key, [...(others.get(key) ?? []), f]);
  }
  for (const [key, items] of others) {
    const top = items[0]; // the feed is newest first
    const change: ChangeView | null | undefined = items.find((f) => f.change)?.change;
    out.push({
      key: `other:${key}`,
      own: false,
      case: top.case,
      // a change is named by what it did ("Information Security practical cancelled"), not by the request
      title: change?.title ?? top.request ?? top.title,
      latest: change ? weeksLabel(change) : top.title,
      changeOnly: items.every((f) => f.kind === "change"),
      at: top.at,
      needs: items.some((f) => f.needs_you),
      unread: items.some((f) => f.unread),
      feed: items,
    });
  }
  return out.sort((a, b) => b.at.localeCompare(a.at));
}

/** This person's conversations, kept fresh. */
export function useConversations() {
  const { data: cases, refresh: refreshCases } = useApi(() => api.cases("mine"), [], 3000);
  const { data: feed, refresh: refreshFeed } = useApi(() => api.feed(), [], 3000);
  const convs = useMemo(() => buildConversations(cases ?? [], feed ?? []), [cases, feed]);
  return {
    convs,
    loaded: !!cases && !!feed,
    refresh: () => {
      void refreshCases();
      void refreshFeed();
    },
  };
}

/** Conversations grouped by what they wait on, filtered by ``q``; ``short`` caps the long groups (the dropdown). */
export function ConversationList({ items, selected, onSelect, q = "", short = false }: {
  items: Conversation[];
  selected: string | null;
  onSelect: (key: string) => void;
  q?: string;
  short?: boolean;
}) {
  const { user } = useAuth();
  // a class representative sees changes to their section's timetable, not colleagues' requests
  const label = (g: Group) => (g === "updates" && user?.role === "student" ? "Changes to your timetable" : GROUP_LABEL[g]);
  // the requests a message was split into are listed under it, not on their own
  const byKey = useMemo(() => new Map(items.map((c) => [c.key, c])), [items]);
  const partsOf = (c: Conversation) => (c.parts ?? []).map((p) => byKey.get(p)).filter((p): p is Conversation => !!p);
  const groups = useMemo(() => {
    const ql = q.trim().toLowerCase();
    const hit = (c: Conversation) => !ql || `${c.case ?? ""} ${c.title} ${c.latest}`.toLowerCase().includes(ql);
    const out: Record<Group, Conversation[]> = { needs: [], progress: [], updates: [], done: [] };
    for (const c of items) {
      if (c.partOf && byKey.has(c.partOf)) continue;
      if (hit(c) || partsOf(c).some(hit)) out[groupOf(c)].push(c);
    }
    return out;
  }, [items, q, byKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const row = (c: Conversation, nested = false) => {
    const s = c.status ? { ...STATUS[c.status], tone: PLAIN_STATUS[c.status].tone } : null;
    const Icon = s ? s.icon : c.feed.some((f) => f.kind === "negotiation") ? MessagesSquare : CalendarClock;
    return (
      <button
        key={c.key}
        onClick={() => onSelect(c.key)}
        className={clsx(
          "flex w-full items-start gap-2.5 pr-3 text-left hover:bg-panel-2",
          nested ? "py-1.5 pl-8" : "py-2 pl-3",
          c.key === selected && "bg-panel-2",
        )}
      >
        <Icon size={nested ? 12 : 14} className={clsx("mt-0.5 shrink-0", s ? TONE_CLASS[s.tone].text : c.needs ? "text-warn" : "text-info")} />
        <span className="min-w-0 flex-1">
          <span className={clsx("block truncate", nested ? "text-[12.5px] text-ink-2" : "text-[13px]", c.unread && "font-semibold")}>{c.title}</span>
          <span className="flex items-center gap-1.5 truncate text-[11.5px] text-ink-3">
            {c.own && c.status && <span className={clsx("size-1.5 shrink-0 rounded-full", OUTCOME_DOT[c.outcome ?? "progress"])} />}
            <span className={clsx("truncate", c.own && c.status && !c.running && OUTCOME_TEXT[c.outcome ?? "progress"])}>{c.latest}</span>
            {!nested && <span className="shrink-0">· {ago(c.at)}</span>}
          </span>
        </span>
        {c.unread && <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-brand" />}
      </button>
    );
  };
  return (
    <>
      {(["needs", "progress", "updates", "done"] as Group[]).map(
        (g) =>
          groups[g].length > 0 && (
            <div key={g}>
              <p className="px-3 pb-1 pt-2 text-[11px] font-medium uppercase tracking-wide text-ink-3">{label(g)}</p>
              {(short ? groups[g].slice(0, g === "done" || g === "updates" ? 8 : 20) : groups[g]).map((c) => {
                const parts = partsOf(c);
                if (!parts.length) return row(c);
                return (
                  <div key={c.key}>
                    {row(c)}
                    <div className="relative before:absolute before:bottom-2 before:left-[18px] before:top-0 before:w-px before:bg-line">
                      {parts.map((p) => row(p, true))}
                    </div>
                  </div>
                );
              })}
            </div>
          ),
      )}
      {!items.length && <p className="px-3 py-6 text-center text-[13px] text-ink-3">Nothing yet. Write a request below.</p>}
      {items.length > 0 && !Object.values(groups).some((g) => g.length) && <p className="px-3 py-6 text-center text-[13px] text-ink-3">Nothing matches “{q}”.</p>}
    </>
  );
}

/** A colleague's request as it reached this person: what was asked, the options sent to them, and how
 * it changed their timetable. */
export function ColleagueThread({ c, onChanged }: { c: Conversation; onChanged: () => void }) {
  const items = [...c.feed].reverse(); // oldest first, like a conversation
  return (
    <div className="space-y-3">
      {c.feed[0]?.request && !c.changeOnly && (
        <ReplyBubble label="A colleague's request">“{c.feed[0].request}”</ReplyBubble>
      )}
      {items.map((f) =>
        f.kind === "negotiation" && f.message ? (
          <Options key={f.id} item={f.message} onDone={onChanged} />
        ) : f.change ? (
          <ChangeCard key={f.id} item={f} change={f.change} />
        ) : (
          <ReplyBubble
            key={f.id}
            tone="info"
            label={f.title}
            footer={`${f.from_name ? `${f.from_name} · ` : ""}${ago(f.at)}${f.version != null ? ` · timetable version ${f.version}` : ""}`}
          >
            {f.text}
          </ReplyBubble>
        ),
      )}
    </div>
  );
}
