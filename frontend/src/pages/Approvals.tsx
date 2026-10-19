import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, BadgeCheck, Bell, ExternalLink, Gavel, MessagesSquare, Scale, TriangleAlert, Users, XCircle } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { ForwardedList } from "../components/Forwarded";
import { DiffTable } from "../components/Negotiation";
import { Avatar, Button, Card, CardHeader, EmptyState, PageHeader, Skeleton, Toast } from "../components/ui";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { ago, num } from "../lib/meta";
import type { CaseDetail, CaseSummary } from "../lib/types";

export default function Approvals() {
  const { data, refresh, loading } = useApi(() => api.approvals(), [], 3000);
  const [toast, setToast] = useState<string | null>(null);
  const [gone, setGone] = useState<Set<string>>(new Set());

  // escalations sent to the office: not yet a change to publish, but a decision only it can make
  const { data: all } = useApi(() => api.cases("all"), [], 3000);
  const decisions = (all ?? []).filter((c) => c.status === "escalated" && c.escalated_to === "coordinator");
  // messages the system could not act on, passed to the office for a reply
  const forwarded = (all ?? []).filter((c) => c.forwarded_to === "coordinator" && !c.handled);

  const list = (data ?? []).filter((c) => !gone.has(c.id));
  return (
    <>
      <PageHeader
        title="Approvals"
        subtitle="Nothing is published until you approve it."
      />
      {decisions.length > 0 && <Decisions cases={decisions} />}
      {forwarded.length > 0 && <ForwardedList cases={forwarded} />}
      {decisions.length + forwarded.length > 0 && <h2 className="mb-3 text-[13px] font-medium text-ink-3">Changes to publish</h2>}
      {loading && !data ? (
        <Skeleton className="h-80" />
      ) : list.length ? (
        <div className="space-y-6">
          <AnimatePresence>
            {list.map((c) => (
              <motion.div key={c.id} exit={{ opacity: 0, transition: { duration: 0.2 } }}>
                <ApprovalCard
                  c={c}
                  onDone={async (msg) => {
                    setGone((g) => new Set(g).add(c.id));
                    setToast(msg);
                    await refresh();
                  }}
                />
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      ) : (
        <Card>
          <EmptyState
            icon={BadgeCheck}
            title={decisions.length ? "No changes to publish" : "All caught up"}
            text={decisions.length ? "A granted escalation comes back here once it has been fitted in." : "Changes that need your approval will appear here."}
          />
        </Card>
      )}
      <Toast message={toast} onDone={() => setToast(null)} />
    </>
  );
}

function Decisions({ cases }: { cases: CaseSummary[] }) {
  return (
    <Card className="mb-8 overflow-hidden border-warn/30">
      <CardHeader
        icon={Gavel}
        title={`Decisions waiting for you (${cases.length})`}
        subtitle="Requests nobody could make room for without breaking something fixed. Grant or decline each one."
      />
      <ul className="divide-y divide-line">
        {cases.map((c) => (
          <li key={c.id}>
            <Link to={`/requests/${c.id}`} className="group flex items-center gap-4 px-5 py-3.5 hover:bg-panel-2/60">
              <span className="grid size-8 shrink-0 place-items-center rounded-full bg-warn/12 text-warn">
                <TriangleAlert size={15} />
              </span>
              <div className="min-w-0 flex-1">
                <p className="truncate text-[14px]">“{c.text}”</p>
                <p className="mt-0.5 text-[12.5px] text-ink-3">
                  {c.sender_name} · {ago(c.received_at)} · <span className="font-mono">{c.id}</span>
                </p>
              </div>
              <span className="flex shrink-0 items-center gap-1 text-[13px] font-medium text-brand">
                Decide <ArrowRight size={14} className="transition-transform group-hover:translate-x-0.5" />
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </Card>
  );
}

/** ``compact``: one column (the inbox's pane beside its list), the change first and the rest under it. */
export function ApprovalCard({ c, onDone, compact = false }: { c: CaseDetail; onDone: (msg: string) => Promise<void>; compact?: boolean }) {
  const [busy, setBusy] = useState<"approve" | "reject" | null>(null);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState("");
  const diff = c.proposal?.diff ?? [];
  const teachers = [...new Set(diff.map((d) => d.faculty_name ?? "").filter(Boolean))];
  const sections = [...new Set(diff.flatMap((d) => d.groups.map((g) => g.replace("G-", "").replace(/^0/, ""))))].sort((a, b) => Number(a) - Number(b));
  const o = c.outcome;

  const approve = async () => {
    setBusy("approve");
    try {
      const r = await api.approve(c.id);
      await onDone(`Published version ${r.version}. Notified ${Object.keys(r.notified).length} affected people.`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <Card>
      <div className="flex flex-wrap items-start gap-4 border-b border-line p-5">
        <Avatar name={c.sender_name} id={c.sender} size={40} />
        <div className="min-w-0 flex-1">
          <p className="text-[15px] font-medium leading-snug">“{c.text}”</p>
          <p className="mt-1 text-[12.5px] text-ink-3">
            <Link to={`/requests/${c.id}`} className="font-mono text-brand hover:underline">
              {c.id}
            </Link>{" "}
            · {c.sender_name} · {ago(c.received_at)} · proposes version {c.proposal?.version}
            {c.proposal?.week ? ` for week ${c.proposal.week}` : " (semester timetable)"}
          </p>
          {c.proposal?.builds_on && (
            <p className="mt-1 text-[12.5px] text-ink-2">
              Builds on{" "}
              <Link to={`/requests/${c.proposal.builds_on}`} className="font-mono text-brand hover:underline">
                {c.proposal.builds_on}
              </Link>
              , which is still waiting: approve or reject that first.
            </p>
          )}
        </div>
        <div className="flex gap-2">
          <Button icon={XCircle} onClick={() => setRejecting(!rejecting)} disabled={!!busy}>
            Reject
          </Button>
          <Button variant="primary" icon={BadgeCheck} loading={busy === "approve"} onClick={approve}
                  disabled={!!c.proposal?.builds_on}>
            Approve and publish
          </Button>
        </div>
      </div>
      <AnimatePresence>
        {rejecting && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden border-b border-line bg-panel-2">
            <div className="flex flex-wrap items-center gap-3 p-4">
              <input
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                placeholder="Why? The case goes back to negotiation with your note."
                className="h-10 flex-1 rounded-md border border-line bg-panel px-3 text-sm outline-none focus:border-bad/50"
              />
              <Button
                variant="danger"
                loading={busy === "reject"}
                onClick={async () => {
                  setBusy("reject");
                  try {
                    await api.reject(c.id, reason);
                    await onDone("Sent back to negotiation.");
                  } finally {
                    setBusy(null);
                  }
                }}
              >
                Confirm rejection
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <div className={compact ? "" : "grid gap-0 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]"}>
        <div className={compact ? "border-b border-line p-5" : "border-b border-line p-5 lg:border-b-0 lg:border-r"}>
          <p className="mb-3 text-xs font-medium text-ink-3">Changes ({diff.length})</p>
          <div className={compact && diff.length > 8 ? "max-h-[420px] overflow-y-auto" : undefined}>
            <DiffTable rows={diff} />
          </div>
        </div>
        <div className={compact ? "grid gap-5 p-5 sm:grid-cols-2" : "space-y-5 p-5"}>
          <div className={compact ? "sm:col-span-2" : undefined}>
            <p className="mb-2 flex items-center gap-1.5 text-xs font-medium text-ink-3">
              <Users size={13} /> Affected, will be notified
            </p>
            {teachers.length || sections.length ? (
              <div className="space-y-1 text-[13px] text-ink-2">
                {teachers.length > 0 && (
                  <p>
                    <Bell size={11} className="mr-1 inline text-info" />
                    <span className="text-ink-3">{teachers.length === 1 ? "Teacher" : `${teachers.length} teachers`}:</span> {teachers.join(", ")}
                  </p>
                )}
                {sections.length > 0 && (
                  <p>
                    <Bell size={11} className="mr-1 inline text-info" />
                    <span className="text-ink-3">{sections.length === 1 ? "Section" : `${sections.length} sections`}:</span> {sections.join(", ")}
                  </p>
                )}
              </div>
            ) : (
              <span className="text-sm text-ink-3">Nobody else.</span>
            )}
          </div>
          <div>
            <p className="mb-2 flex items-center gap-1.5 text-xs font-medium text-ink-3">
              <MessagesSquare size={13} /> How it was resolved
            </p>
            <p className="text-[13.5px] text-ink-2">
              {o?.status === "agreed"
                ? `Negotiated in ${o.rounds} round${o.rounds > 1 ? "s" : ""}: ${o.concessions.map((x) => x.name).join(", ") || "—"} conceded.`
                : o?.step === 2
                  ? "Solved directly; some preferences could not be kept and their owners were told."
                  : "Solved directly; no one had to concede."}
            </p>
            {c.policy && (
              <p className="mt-1.5 text-[13px] text-ink-3">
                Policy: {c.policy.verdict}
                {c.policy.obligations.length ? ` · obligation ${c.policy.obligations.join(", ")}` : ""}
              </p>
            )}
          </div>
          <div>
            <p className="mb-2 flex items-center gap-1.5 text-xs font-medium text-ink-3">
              <Scale size={13} /> Fairness
            </p>
            <p className="text-[13.5px]">
              Gini {num(c.fairness.gini_before ?? 0, 3)} → <span className="font-semibold">{num(c.fairness.gini_after ?? 0, 3)}</span>
            </p>
          </div>
          <Link to={`/requests/${c.id}`} className="inline-flex items-center gap-1.5 text-[13px] text-brand hover:underline">
            Full audit trail <ExternalLink size={13} />
          </Link>
        </div>
      </div>
    </Card>
  );
}
