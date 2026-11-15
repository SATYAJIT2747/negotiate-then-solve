import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { changedWeeks, groupWeeks, versionsLabel, weeksLabel } from "../lib/weeks";
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

const FULL_DAY: Record<string, string> = { Mon: "Monday", Tue: "Tuesday", Wed: "Wednesday", Thu: "Thursday", Fri: "Friday", Sat: "Saturday" };
/** "a", "a and b", "a, b and c" */
const listed = (xs: string[]) => (xs.length < 2 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);

/** Approve or reject a change waiting for the office: shared by the approval card and the request page. */
export function useApprovalActions(c: CaseDetail, onDone: (msg: string) => Promise<void> | void) {
  const [busy, setBusy] = useState<"approve" | "reject" | null>(null);
  const confirmOnly = !c.proposal; // already fits: approving keeps the requirement in force, nothing moves
  const approve = async () => {
    setBusy("approve");
    try {
      const r = await api.approve(c.id);
      await onDone(r.version === null
        ? "Confirmed. The requirement is in force; nothing moved."
        : `Published version ${r.version}. Notified ${Object.keys(r.notified).length} affected people.`);
    } finally {
      setBusy(null);
    }
  };
  const reject = async (reason: string) => {
    setBusy("reject");
    try {
      await api.reject(c.id, reason);
      await onDone(confirmOnly ? "Declined; the requirement is not in force." : "Sent back to negotiation.");
    } finally {
      setBusy(null);
    }
  };
  return { busy, approve, reject, confirmOnly };
}

/** ``compact``: one column (the inbox's pane beside its list), the change first and the rest under it. */
export function ApprovalCard({ c, onDone, compact = false }: { c: CaseDetail; onDone: (msg: string) => Promise<void>; compact?: boolean }) {
  const { busy, approve, reject, confirmOnly } = useApprovalActions(c, onDone);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState("");
  const diff = c.proposal?.diff ?? [];
  // every week the approval publishes (a request over weeks 11-12 has a version for each), the same changes together
  const weeks = changedWeeks(c.proposal);
  const groups = groupWeeks(weeks);
  const span = weeks.some((w) => w.week != null)
    ? `${versionsLabel(weeks.map((w) => w.version))} for ${weeksLabel(weeks.map((w) => w.week))}`
    : `${versionsLabel(weeks.map((w) => w.version))} (semester timetable)`;
  const teachers = [...new Set(diff.map((d) => d.faculty_name ?? "").filter(Boolean))];
  // classes the request did not ask to move, moved to free a time one of its classes needs
  const displaced = diff.filter((d) => d.made_room_for);
  const sections = [...new Set(diff.flatMap((d) => d.groups.map((g) => g.replace("G-", "").replace(/^0/, ""))))].sort((a, b) => Number(a) - Number(b));
  // what the change does, in one line: how many classes are cancelled or moved, whose, on which days, in which weeks
  const cancelled = diff.filter((d) => d.before && !d.after);
  const moved = diff.filter((d) => d.before && d.after);
  const added = diff.filter((d) => !d.before && d.after);
  const days = [...new Set(cancelled.map((d) => FULL_DAY[d.before!.day.slice(0, 3)] ?? d.before!.day))];
  const inWeeks = weeks.some((w) => w.week != null) ? ` in ${weeksLabel(weeks.map((w) => w.week))}` : " every week";
  const whose = teachers.length === 1 ? `${teachers[0]}'s ` : "";
  const awayOnly = !!cancelled.length && !moved.length && !added.length && c.makeup === false;
  const summary = !diff.length
    ? null
    : awayOnly
      ? `Cancels ${cancelled.length} of ${whose || "the "}classes${days.length ? ` (${listed(days)})` : ""}${inWeeks} while ${teachers.length === 1 ? "they are" : "their teachers are"} away. No make-up classes, as asked.`
      : [
          cancelled.length && `cancels ${cancelled.length} ${cancelled.length === 1 ? "class" : "classes"}`,
          moved.length && `moves ${moved.length}`,
          added.length && `adds ${added.length}`,
        ].filter(Boolean).join(", ").replace(/^./, (x) => x.toUpperCase()) + `${inWeeks}.`;
  const makeupWaived = c.makeup === false && !!c.policy?.obligations.includes("P-MAKEUP");
  const fairnessSame = (c.fairness.gini_before ?? 0).toFixed(3) === (c.fairness.gini_after ?? 0).toFixed(3);
  const o = c.outcome;

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
            · {c.sender_name} · {ago(c.received_at)} ·{" "}
            {confirmOnly
              ? "already fits the timetable; confirm the requirement it adds"
              : <>proposes {span}</>}
          </p>
          {!!c.proposal?.more?.length && (
            <p className="mt-1 text-[12.5px] text-ink-2">Approving publishes all {weeks.length} weeks at once.</p>
          )}
          {c.proposal?.builds_on && (
            <p className="mt-1 text-[12.5px] text-ink-2">
              Builds on{" "}
              <Link to={`/requests/${c.proposal.builds_on}`} className="font-mono text-brand hover:underline">
                {c.proposal.builds_on}
              </Link>
              , which is still waiting: approve or reject that first.
            </p>
          )}
          {c.stand_in && (
            <p className="mt-1 text-[12.5px] text-warn">
              Read by the rule-based stand-in, not the model: check what it understood before approving.
            </p>
          )}
          {c.injection && (
            <p className="mt-1 text-[12.5px] text-warn">
              The message contained an instruction to the system (such as “ignore your rules”); it was left out
              when the request was read.
            </p>
          )}
          {!!c.understood?.length && (
            <div className="mt-2 text-[12.5px] text-ink-2">
              <p className="font-medium">Read as:</p>
              <ul className="ml-4 list-disc">
                {c.understood.map((u) => (
                  <li key={u}>{u}</li>
                ))}
              </ul>
            </div>
          )}
          {!!c.unused?.length && (
            <div className="mt-1 text-[12.5px] text-warn">
              <p className="font-medium">Not used from the message (check before approving):</p>
              <ul className="ml-4 list-disc">
                {c.unused.map((u) => (
                  <li key={u}>{u}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
        <div className="flex gap-2">
          <Button icon={XCircle} onClick={() => setRejecting(!rejecting)} disabled={!!busy}>
            Reject
          </Button>
          <Button variant="primary" icon={BadgeCheck} loading={busy === "approve"} onClick={approve}
                  disabled={!!c.proposal?.builds_on}>
            {confirmOnly ? "Confirm" : "Approve and publish"}
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
                placeholder={confirmOnly ? "Why? The sender sees your note; the requirement is dropped." : "Why? The case goes back to negotiation with your note."}
                className="h-10 flex-1 rounded-md border border-line bg-panel px-3 text-sm outline-none focus:border-bad/50"
              />
              <Button
                variant="danger"
                loading={busy === "reject"}
                onClick={() => reject(reason)}
              >
                Confirm rejection
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
      <div className={compact ? "" : "grid gap-0 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]"}>
        <div className={compact ? "border-b border-line p-5" : "border-b border-line p-5 lg:border-b-0 lg:border-r"}>
          {summary && <p className="mb-3 text-[13.5px] leading-relaxed text-ink">{summary}</p>}
          <p className="mb-2 text-xs font-medium text-ink-3">
            Changes · {diff.length} {diff.length === 1 ? "class" : "classes"}
            {weeks.length > 1 ? `, each of ${weeks.length} weeks` : ""}
          </p>
          <div className={clsx("space-y-3", compact && diff.length > 8 && "max-h-[420px] overflow-y-auto")}>
            {groups.map((g) => (
              <div key={g.versions.join()}>
                {groups.length > 1 && <p className="mb-1 text-[11.5px] font-medium uppercase tracking-wide text-ink-3">{weeksLabel(g.weeks)}</p>}
                <DiffTable rows={g.diff} weeks={g.weeks.some((w) => w != null) ? weeksLabel(g.weeks) : undefined} />
              </div>
            ))}
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
                    {listed(teachers)}
                  </p>
                )}
                {sections.length > 0 && (
                  <p>
                    <Bell size={11} className="mr-1 inline text-info" />
                    {sections.length === 1 ? `Section ${sections[0]}` : `Sections ${listed(sections)}`}
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
                ? `Negotiated in ${o.rounds} round${o.rounds > 1 ? "s" : ""}: ${listed(o.concessions.map((x) => x.name)) || "—"} gave way.`
                : o?.step === 2
                  ? "Fitted directly; some preferences could not be kept, and their owners are told."
                  : awayOnly
                    ? "The classes are simply not held; nothing else in the timetable moves."
                    : displaced.length
                      ? "Fitted directly."
                      : "Fitted directly; nobody else had to give anything up."}
            </p>
            {displaced.length > 0 && (
              <p className="mt-1.5 text-[13px] text-ink-2">
                To make room, {displaced.length === 1 ? "one other class" : `${displaced.length} other classes`} also
                moved ({listed([...new Set(displaced.map((d) => d.faculty_name ?? "").filter(Boolean))])}'s); the
                reason is under each in the changes.
              </p>
            )}
            {c.policy && (
              <p className="mt-1.5 text-[13px] text-ink-3">
                {c.policy.verdict === "allowed" ? "Within the rules." : `Rules: ${c.policy.verdict.replace("_", " ")}.`}
                {makeupWaived
                  ? " The rules ask for a make-up class (P-MAKEUP); the teacher asked for none."
                  : c.policy.obligations.length
                    ? ` Applies: ${c.policy.obligations.join(", ")}.`
                    : ""}
              </p>
            )}
          </div>
          <div>
            <p className="mb-2 flex items-center gap-1.5 text-xs font-medium text-ink-3">
              <Scale size={13} /> Fairness
            </p>
            {fairnessSame ? (
              <p className="text-[13px] text-ink-3">Unchanged (Gini {num(c.fairness.gini_after ?? 0, 3)}): nobody gives way.</p>
            ) : (
              <p className="text-[13.5px]">
                Gini {num(c.fairness.gini_before ?? 0, 3)} → <span className="font-semibold">{num(c.fairness.gini_after ?? 0, 3)}</span>
              </p>
            )}
          </div>
          <Link to={`/requests/${c.id}`} className="inline-flex items-center gap-1.5 text-[13px] text-brand hover:underline">
            Full audit trail <ExternalLink size={13} />
          </Link>
        </div>
      </div>
    </Card>
  );
}
