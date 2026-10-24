import { ExternalLink, Plus, Search, Trash2, Undo2 } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { RequestBuilder } from "../components/RequestBuilder";
import { Button, Skeleton, StatusBadge } from "../components/ui";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import type { CaseSummary, Status } from "../lib/types";

/** Requests that have run their course: they can be taken off the list (as web.app.FINISHED). */
const FINISHED = new Set<Status>(["published", "refused", "denied", "answered", "forwarded", "withdrawn"]);

/** The Requests tab of faculty and class representatives (CHAT-PLAN step 6): their requests as a table, newest
 * first: the number, what was asked, the status, and quick actions (Open, Withdraw where allowed, Delete from
 * list once finished). New requests are made in the chat; /requests/R-id opens one as a conversation. */
export default function RequestsTable() {
  const navigate = useNavigate();
  const { data, loading, refresh } = useApi(() => api.cases("mine"), [], 5000);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [building, setBuilding] = useState(false); // the New request builder is open
  const rows = (data ?? []).filter((r) => !q || `${r.id} ${r.text}`.toLowerCase().includes(q.toLowerCase()));

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
          <div className="flex h-8 items-center gap-2 rounded-md border border-line bg-panel px-2.5">
            <Search size={13} className="text-ink-3" />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Find a request" className="w-40 bg-transparent text-[13px] outline-none placeholder:text-ink-3" />
          </div>
          <Button size="sm" variant="primary" icon={Plus} onClick={() => setBuilding(true)}>
            New Request
          </Button>
        </div>
      </div>
      {error && <p className="text-[12.5px] text-bad">{error}</p>}

      {loading && !data ? (
        <Skeleton className="h-48" />
      ) : !rows.length ? (
        <p className="py-16 text-center text-[13.5px] text-ink-3">
          {q ? "No request matches." : "No requests yet. Make one in the chat."}
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
                      <StatusBadge status={r.status} live={r.running} />
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
