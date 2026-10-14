import clsx from "clsx";
import { ArrowUpRight, HelpCircle, Inbox, Undo2 } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { ago, EVENT_LABEL } from "../lib/meta";
import type { CaseDetail, EventRow } from "../lib/types";
import { Button, LiveDot } from "./ui";

/** Steps worth a line in the conversation; the rest (parsing, compiling, solving) stay on the case page. */
const SHOWN = new Set([
  "policy_checked",
  "choice_offered",
  "choice_made",
  "negotiating",
  "negotiation_message",
  "negotiation_reply",
  "awaiting_approval",
  "escalated",
  "forwarded",
  "handled",
  "published",
  "denied",
  "refused",
  "answered",
  "withdrawn",
  "approval_refused",
]);

const VERDICT: Record<string, string> = { allowed: "allowed", needs_approval: "needs approval", forbidden: "not allowed" };

function stepText(e: EventRow, c: CaseDetail): string {
  if (e.kind === "policy_checked") {
    const v = (e.verdict as string | undefined) ?? c.policy?.verdict;
    return v && VERDICT[v] ? `Rules checked: ${VERDICT[v]}` : EVENT_LABEL[e.kind];
  }
  return EVENT_LABEL[e.kind] ?? e.kind;
}

/** One request as a conversation: what was sent, the steps that matter to the sender, and the reply,
 * with the rules it rests on. Answering a question back is done in the composer under the thread. */
export function RequestThread({ c, onChanged }: { c: CaseDetail; onChanged: () => void }) {
  const { user } = useAuth();
  const steps = (c.events ?? []).filter((e) => SHOWN.has(e.kind)).filter((e, i, all) => i === 0 || all[i - 1].kind !== e.kind);
  const asking = c.status === "clarification_requested" && c.sender === user?.id;
  const toChoose = (c.inbox ?? []).some((i) => i.to === user?.id && !i.reply);
  const rules = [...(c.policy?.cited ?? []).map((r) => r.cite), ...(c.policy?.obligations ?? []).map((id) => `${id} (applies)`)];

  return (
    <div className="space-y-3">
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-lg rounded-br-sm bg-brand/12 px-3 py-2">
          <p className="whitespace-pre-wrap text-[13.5px] leading-relaxed">{c.text}</p>
          <p className="mt-1 text-right text-[11px] text-ink-3">{ago(c.received_at)}</p>
        </div>
      </div>

      {steps.length > 0 && (
        <ul className="space-y-1 pl-1">
          {steps.map((e) => (
            <li key={e.n} className="flex items-center gap-2 text-[12px] text-ink-3">
              <span className="size-1.5 shrink-0 rounded-full bg-ink-3/50" />
              {stepText(e, c)}
            </li>
          ))}
        </ul>
      )}

      {c.running ? (
        <p className="flex items-center gap-2 pl-1 text-[12.5px] text-ink-2">
          <LiveDot tone="warn" pulse /> {c.status === "negotiating" ? "Waiting for a colleague's reply…" : "Working on it…"}
        </p>
      ) : (
        c.reply && (
          <div className={clsx("max-w-[92%] rounded-lg rounded-bl-sm border px-3 py-2", asking ? "border-warn/40 bg-warn/5" : "border-line bg-panel-2/60")}>
            {asking && (
              <p className="mb-1 flex items-center gap-1.5 text-[11.5px] font-medium text-warn">
                <HelpCircle size={13} /> A question for you. Answer it below.
              </p>
            )}
            <p className="whitespace-pre-wrap text-[13.5px] leading-relaxed">{c.reply}</p>
            {c.answer && <p className="mt-1 text-[11.5px] text-ink-3">From {c.answer.by_name}</p>}
            {rules.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1">
                {rules.map((r) => (
                  <span key={r} className="rounded border border-line bg-panel px-1.5 py-0.5 text-[11px] text-ink-2" title="The rule this rests on">
                    {r}
                  </span>
                ))}
              </div>
            )}
          </div>
        )
      )}

      {toChoose && (
        <Link to="/inbox" className="flex items-center gap-2 rounded-lg border border-brand/40 bg-brand/5 px-3 py-2 text-[13px] text-brand hover:bg-brand/10">
          <Inbox size={14} /> There is something for you to choose in your inbox
          <ArrowUpRight size={13} className="ml-auto" />
        </Link>
      )}

      <div className="flex items-center gap-2 pt-1">
        {c.can_withdraw && <Withdraw id={c.id} onDone={onChanged} />}
        <Link to={`/requests/${c.id}`} className="ml-auto flex items-center gap-1 text-[12px] text-ink-3 hover:text-brand">
          Details <ArrowUpRight size={12} />
        </Link>
      </div>
    </div>
  );
}

function Withdraw({ id, onDone }: { id: string; onDone: () => void }) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!confirming)
    return (
      <Button size="sm" variant="ghost" icon={Undo2} onClick={() => setConfirming(true)}>
        Withdraw
      </Button>
    );
  return (
    <span className="flex items-center gap-2 text-[12.5px]">
      Withdraw it?
      <Button
        size="sm"
        variant="danger"
        loading={busy}
        onClick={async () => {
          setBusy(true);
          setError(null);
          try {
            await api.withdraw(id);
            setConfirming(false);
            onDone();
          } catch (e) {
            setError((e as Error).message);
          } finally {
            setBusy(false);
          }
        }}
      >
        Yes
      </Button>
      <Button size="sm" variant="ghost" onClick={() => setConfirming(false)}>
        No
      </Button>
      {error && <span className="text-bad">{error}</span>}
    </span>
  );
}
