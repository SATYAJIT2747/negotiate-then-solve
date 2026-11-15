import { ArrowRight, ExternalLink, Gavel, Plus, Search, Trash2, Undo2 } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { RequestBuilder } from "../components/RequestBuilder";
import { Button, Skeleton, StatusBadge, Tabs } from "../components/ui";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useApi } from "../lib/hooks";
import { ago } from "../lib/meta";
import { decidesOn, effectiveStatus, partsRunning } from "../lib/requests";
import type { CaseSummary, Status } from "../lib/types";

/** Requests that have run their course: they can be taken off the list (as web.app.FINISHED). */
const FINISHED = new Set<Status>(["published", "refused", "denied", "answered", "forwarded", "withdrawn"]);

type Show = "all" | "waiting" | "progress" | "done";

/** Where a request stands for its sender: waiting on their answer, still under way, or run its course. */
const stage = (r: CaseSummary, byId?: Map<string, CaseSummary>): Exclude<Show, "all"> => {
  const status = byId ? effectiveStatus(r, byId) : r.status; // a split message stands where its parts do
  return r.waits_on_you || status === "clarification_requested" ? "waiting" : FINISHED.has(status) ? "done" : "progress";
};

/** The Requests tab of faculty and class representatives (CHAT-PLAN step 6): their requests as a table, newest
 * first: the number, what was asked, the status, and quick actions (Open, Withdraw where allowed, Delete from
 * list once finished). New requests are made in the chat; /requests/R-id opens one as a conversation. */
export default function RequestsTable() {
  const navigate = useNavigate();
  const { can } = useAuth(); // the office sees the pipeline's step names, everyone else plain statuses
  const { data, loading, refresh } = useApi(() => api.cases("mine"), [], 5000);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [building, setBuilding] = useState(false); // the New request builder is open
  const [show, setShow] = useState<Show>("all");
  const found = (data ?? []).filter((r) => !q || `${r.id} ${r.text}`.toLowerCase().includes(q.toLowerCase()));
  const byId = new Map((data ?? []).map((r) => [r.id, r] as const));
  const rows = found.filter((r) => show === "all" || stage(r, byId) === show);
  const count = (x: Show) => found.filter((r) => stage(r, byId) === x).length;

  const act = async (r: CaseSummary, what: "withdraw" | "hide") => {
    if (what === "withdraw" && !window.confirm(`Withdraw ${r.id}? Everything it set in motion is undone.`)) return;
    setBusy(`${r.id}:${what}`);
    setError(null);
    try {
      await (what === "withdraw" ? api.withdraw(r.id) : api.hide(r.id));
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="mx-auto w-full space-y-4 lg:w-[88%]">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-[17px] font-semibold">Requests</h1>
        <div className="flex items-center gap-2">
          {can("see_all") && (
            <Link to="/requests?details=1" className="mr-1 text-[12.5px] text-ink-3 hover:text-brand hover:underline" title="Every request in the department, where it stands">
              All department requests
            </Link>
          )}
          <div className="flex h-8 items-center gap-2 rounded-md border border-line bg-panel px-2.5">
            <Search size={13} className="text-ink-3" />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Find a request" className="w-40 bg-transparent text-[13px] outline-none placeholder:text-ink-3" />
          </div>
          <Button size="sm" variant="primary" icon={Plus} onClick={() => setBuilding(true)}>
            New Request
          </Button>
        </div>
      </div>
      <Tabs
        tabs={[
          { id: "all", label: "All" },
          { id: "waiting", label: "Waiting on you", count: count("waiting") },
          { id: "progress", label: "In progress", count: count("progress") },
          { id: "done", label: "Finished", count: count("done") },
        ]}
        value={show}
        onChange={setShow}
      />
      {can("see_all") && <NeedsDecision />}
      {error && <p className="text-[12.5px] text-bad">{error}</p>}

      {loading && !data ? (
        <Skeleton className="h-48" />
      ) : !rows.length ? (
        <p className="py-16 text-center text-[13.5px] text-ink-3">
          {q || show !== "all" ? "No request matches." : "No requests yet. Make one in the chat."}
        </p>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-line">
          <table className="w-full text-left text-[13px]">
            <thead className="border-b border-line bg-panel text-[11.5px] uppercase tracking-wide text-ink-3">
              <tr>
                <th className="px-3 py-2 font-medium">Request</th>
                <th className="px-3 py-2 font-medium">Description</th>
                <th className="px-3 py-2 font-medium">Status</th>
                <th className="px-3 py-2 text-right font-medium">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              {rows.map((r) => {
                const finished = FINISHED.has(r.status);
                return (
                  <tr key={r.id} onClick={() => navigate(`/requests/${r.id}`)} className="cursor-pointer hover:bg-panel-2/50">
                    <td className="whitespace-nowrap px-3 py-2.5 font-mono text-[12px]">
                      <Link to={`/requests/${r.id}`} onClick={(e) => e.stopPropagation()} className="text-brand hover:underline">
                        {r.id}
                      </Link>
                    </td>
                    <td className="max-w-0 px-3 py-2.5">
                      <p className="truncate" title={r.text}>{r.text}</p>
                      {r.waits_on_you && <p className="text-[11.5px] font-medium text-warn">Waiting on your answer</p>}
                      <p className="text-[11.5px] text-ink-3">{new Date(r.received_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}</p>
                    </td>
                    <td className="whitespace-nowrap px-3 py-2.5">
                      <StatusBadge status={effectiveStatus(r, byId)} live={r.running || partsRunning(r, byId)} plain={!can("approve")} />
                      {!!r.parts && r.parts.length > 1 && <p className="mt-0.5 text-[11.5px] text-ink-3">{r.parts.length} requests</p>}
                    </td>
                    <td className="whitespace-nowrap px-3 py-2.5">
                      <div className="flex justify-end gap-1" onClick={(e) => e.stopPropagation()}>
                        <IconAction title="Open" icon={ExternalLink} onClick={() => navigate(`/requests/${r.id}`)} />
                        {r.can_withdraw && (
                          <IconAction title="Withdraw" icon={Undo2} busy={busy === `${r.id}:withdraw`} onClick={() => void act(r, "withdraw")} />
                        )}
                        {finished && (
                          <IconAction title="Delete from list" icon={Trash2} busy={busy === `${r.id}:hide`} onClick={() => void act(r, "hide")} />
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {building && (
        <RequestBuilder
          onClose={() => setBuilding(false)}
          onSent={() => {
            setBuilding(false);
            void refresh();
          }}
        />
      )}
    </div>
  );
}

/** The head of department and the Dean: their Requests page is their own requests (the HoD teaches); on top of it,
 * only the requests that wait on their decision (escalations sent to them), each opening on its decision page. */
function NeedsDecision() {
  const { user } = useAuth();
  const { data } = useApi(() => api.cases("all"), [], 5000);
  const waiting = (data ?? []).filter((c) => decidesOn(c, user));
  if (!data) return null;
  if (!waiting.length)
    return (
      <p className="flex items-center gap-2 rounded-lg border border-line px-3 py-2 text-[12.5px] text-ink-3">
        <Gavel size={14} /> Nothing waits on your decision.
      </p>
    );
  return (
    <div className="overflow-hidden rounded-lg border border-warn/40">
      <p className="flex items-center gap-2 border-b border-warn/30 bg-warn/5 px-3 py-2 text-[12.5px] font-medium text-warn">
        <Gavel size={14} /> Needs your decision · {waiting.length}
        <span className="font-normal text-ink-3">Colleagues' requests sent to you</span>
      </p>
      <ul className="divide-y divide-line">
        {waiting.map((c) => (
          <li key={c.id}>
            <Link to={`/requests/${c.id}?details=1`} className="group flex items-center gap-3 px-3 py-2.5 hover:bg-panel-2/50">
              <span className="font-mono text-[12px] text-brand">{c.id}</span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[13px]" title={c.text}>
                  {c.text}
                </span>
                <span className="block text-[11.5px] text-ink-3">
                  {c.sender_name} · {ago(c.received_at)}
                </span>
              </span>
              <span className="flex items-center gap-1 rounded-md bg-warn/15 px-2 py-1 text-[12px] font-medium text-warn">
                Decide <ArrowRight size={12} className="transition-transform group-hover:translate-x-0.5" />
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}

function IconAction({ title, icon: Icon, onClick, busy }: { title: string; icon: typeof Trash2; onClick: () => void; busy?: boolean }) {
  return (
    <button
      onClick={onClick}
      disabled={busy}
      title={title}
      aria-label={title}
      className="grid size-7 place-items-center rounded-md text-ink-3 hover:bg-panel-2 hover:text-ink disabled:opacity-40"
    >
      <Icon size={14} />
    </button>
  );
}
