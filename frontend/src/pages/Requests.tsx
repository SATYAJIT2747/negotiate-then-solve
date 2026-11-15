import clsx from "clsx";
import { ChevronRight, Gavel, ListChecks, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Avatar, Badge, Card, EmptyState, PageHeader, Skeleton, StatusBadge, Tabs } from "../components/ui";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useApi } from "../lib/hooks";
import { ago, REQUESTS_AS_CHATS } from "../lib/meta";
import type { Role } from "../lib/types";
import { effectiveStatus, kindLabel, SENDER_GROUP, SENDER_LABEL, stageOf, waitsOn, type SenderGroup, type Stage } from "../lib/requests";

type Filter = Stage | "all";

const SELECT = "h-8 rounded-md border border-line bg-panel px-2 text-[13px] outline-none hover:bg-panel-2 focus:border-brand/50";

/** The department's requests (the office, the HoD and the Dean; anyone else, their own). What waits on the
 * person's decision comes first and is marked; the rest by where it stands: still waiting (and on whom), done,
 * or ended without a change. Filters by what was asked and by who asked; statuses in plain words for everyone
 * but the timetable office, who reads the pipeline's steps. */
export default function Requests() {
  const { user, can } = useAuth();
  const office = can("approve");
  // the HoD and the Dean read their own requests as conversations; here (the department's list) each request
  // opens on its detailed page, where an escalation is decided
  const asChats = REQUESTS_AS_CHATS.includes(user?.role as Role);
  const [scope, setScope] = useState<"mine" | "all">(can("see_all") ? "all" : "mine");
  const [filter, setFilter] = useState<Filter>("all");
  const [kind, setKind] = useState("");
  const [sender, setSender] = useState<SenderGroup | "">("");
  const [q, setQ] = useState("");
  const { data, loading } = useApi(() => api.cases(scope), [scope], 3000);

  const byId = useMemo(() => new Map((data ?? []).map((c) => [c.id, c] as const)), [data]);
  const all = useMemo(() => (data ?? []).map((c) => ({ c, stage: stageOf(c, user, byId) })), [data, user, byId]);
  // what waits on the person's decision opens first, once, when there is any
  const [opened, setOpened] = useState(false);
  useEffect(() => {
    if (opened || !data) return;
    setOpened(true);
    if (all.some((r) => r.stage === "needs")) setFilter("needs");
  }, [data]); // eslint-disable-line react-hooks/exhaustive-deps

  const kinds = useMemo(() => [...new Set(all.map((r) => r.c.type).filter((t): t is string => !!t))].sort(), [all]);
  const senders = useMemo(() => [...new Set(all.map((r) => SENDER_GROUP[r.c.role]).filter(Boolean))], [all]);
  // the kind and sender filters narrow every tab, so the tabs' counts follow them
  const narrowed = useMemo(() => {
    const ql = q.toLowerCase();
    return all.filter(
      ({ c }) =>
        (!kind || c.type === kind) &&
        (!sender || SENDER_GROUP[c.role] === sender) &&
        (!ql || `${c.id} ${c.text} ${c.sender_name}`.toLowerCase().includes(ql)),
    );
  }, [all, kind, sender, q]);
  const rows = narrowed
    .filter((r) => filter === "all" || r.stage === filter)
    .sort((a, b) => Number(b.stage === "needs") - Number(a.stage === "needs")); // what needs the person on top
  const count = (f: Filter) => narrowed.filter((r) => f === "all" || r.stage === f).length;
  const needs = count("needs");

  return (
    <>
      <PageHeader
        title={scope === "all" ? "All requests" : "Your requests"}
        subtitle={
          scope === "all"
            ? "The department's requests: what waits on your decision first, then where each one stands. Open one to see how it was handled."
            : "Open a request to see how it was handled, step by step."
        }
        actions={
          can("see_all") && (
            <Tabs
              tabs={[
                { id: "all", label: "Everyone" },
                { id: "mine", label: "Mine" },
              ]}
              value={scope}
              onChange={setScope}
            />
          )
        }
      />
      <Card>
        <div className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-3">
          <Tabs
            tabs={[
              ...(needs > 0 || filter === "needs" ? [{ id: "needs" as const, label: "Needs your decision", icon: Gavel, count: needs }] : []),
              { id: "waiting", label: "Waiting", count: count("waiting") },
              { id: "done", label: "Done", count: count("done") },
              { id: "ended", label: "Not done", count: count("ended") },
              { id: "all", label: "All", count: count("all") },
            ]}
            value={filter}
            onChange={setFilter}
          />
          <div className="ml-auto flex flex-wrap items-center gap-2">
            {kinds.length > 1 && (
              <select value={kind} onChange={(e) => setKind(e.target.value)} aria-label="What was asked" className={SELECT}>
                <option value="">Every kind</option>
                {kinds.map((k) => (
                  <option key={k} value={k}>
                    {kindLabel(k)}
                  </option>
                ))}
              </select>
            )}
            {scope === "all" && senders.length > 1 && (
              <select value={sender} onChange={(e) => setSender(e.target.value as SenderGroup | "")} aria-label="Who asked" className={SELECT}>
                <option value="">Everyone</option>
                {senders.map((g) => (
                  <option key={g} value={g}>
                    {SENDER_LABEL[g]}
                  </option>
                ))}
              </select>
            )}
            <div className="relative w-48">
              <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-3" />
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Find a request"
                className="h-8 w-full rounded-md border border-line bg-panel pl-8 pr-3 text-[13px] outline-none focus:border-brand/50"
              />
            </div>
          </div>
        </div>
        {loading && !data ? (
          <div className="space-y-2 p-4">
            {Array.from({ length: 6 }, (_, i) => (
              <Skeleton key={i} className="h-12" />
            ))}
          </div>
        ) : rows.length ? (
          <ul className="divide-y divide-line">
            {rows.map(({ c, stage }) => {
              const on = waitsOn(c);
              const mine = stage === "needs";
              return (
                <li key={c.id}>
                  <Link
                    to={`/requests/${c.id}${asChats ? "?details=1" : ""}`}
                    className={clsx("group relative flex items-center gap-3 px-4 py-2.5 hover:bg-panel-2/60", mine && "bg-warn/5")}
                  >
                    {mine && <span className="absolute inset-y-0 left-0 w-0.5 bg-warn" />}
                    <Avatar name={c.sender_name} id={c.sender} size={28} />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-[13.5px]" title={c.text}>
                        {c.text}
                      </p>
                      <p className="mt-0.5 flex flex-wrap items-center gap-x-1.5 text-[12px] text-ink-3">
                        <span className="font-mono text-[11.5px]">{c.id}</span>
                        <span>·</span>
                        <span>{c.sender_name}</span>
                        <span>·</span>
                        <span>{ago(c.received_at)}</span>
                        {c.type && (
                          <>
                            <span>·</span>
                            <span>{kindLabel(c.type)}</span>
                          </>
                        )}
                        {on && !mine && (
                          <>
                            <span>·</span>
                            <span className="text-ink-2">waiting for {on}</span>
                          </>
                        )}
                      </p>
                    </div>
                    {c.rounds > 0 && (
                      <span className="hidden sm:inline">
                        <Badge tone="muted">
                          {c.rounds} round{c.rounds > 1 ? "s" : ""}
                        </Badge>
                      </span>
                    )}
                    {mine ? (
                      <span className="rounded-md bg-warn/15 px-2 py-1 text-[12px] font-medium text-warn">Decide</span>
                    ) : (
                      <StatusBadge status={effectiveStatus(c, byId)} live={c.running} plain={!office} />
                    )}
                    <ChevronRight size={16} className="shrink-0 text-ink-3 transition-transform group-hover:translate-x-0.5" />
                  </Link>
                </li>
              );
            })}
          </ul>
        ) : (
          <EmptyState
            icon={ListChecks}
            title={filter === "needs" ? "Nothing waits on your decision" : "No requests here"}
            text={q || kind || sender ? "Nothing matches the filters." : "Requests appear here with their full audit trail."}
          />
        )}
      </Card>
    </>
  );
}
