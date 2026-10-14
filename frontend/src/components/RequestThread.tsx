import clsx from "clsx";
import { ArrowUpRight, Check, Clock3, HelpCircle, Undo2, X } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { ago, EVENT_LABEL } from "../lib/meta";
import type { CaseDetail, EventRow, InboxItem } from "../lib/types";
import { OfferCard } from "./Negotiation";
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
  const forMe = (c.inbox ?? []).filter((i) => i.to === user?.id);
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

      {forMe.map((i) => (
        <Options key={i.id} item={i} onDone={onChanged} />
      ))}

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

/** Options the negotiator sent this person (a time to choose for their own request, or a way to resolve a
 * clash): chosen here, through the same reply the Inbox sends. A counter-offer or a reply in their own
 * words stays on the Inbox page. */
export function Options({ item, onDone }: { item: InboxItem; onDone: () => void }) {
  const [choice, setChoice] = useState<string | null>(null);
  const [busy, setBusy] = useState<"accept" | "reject" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const m = item.message;
  const choosing = m.explanation_mode === "choice"; // their own request fits in more than one way
  const act = async (kind: "accept" | "reject") => {
    setBusy(kind);
    setError(null);
    try {
      await api.reply(
        item.id,
        kind === "accept" ? { decision: "accept", choice: choice!, text: `Option ${choice} works for me.` } : { decision: "reject", text: "None of these work for me." },
      );
      onDone();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  if (item.reply)
    return (
      <p className="flex items-center gap-2 pl-1 text-[12.5px] text-ink-2">
        <Check size={14} className="text-ok" />
        {item.reply.decision === "accept" && item.reply.choice
          ? `You chose option ${item.reply.choice}`
          : item.reply.decision === "reject"
            ? "You declined these options"
            : `You answered: ${item.reply.decision}`}
        {item.answered_by === "deadline" ? " (deadline passed)" : item.answered_by === "simulator" ? " (simulated)" : ""}
      </p>
    );

  return (
    <div className="space-y-2.5 rounded-lg border border-brand/40 bg-brand/5 p-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-[12.5px] font-medium text-brand">{choosing ? "Choose a time" : "A clash with your timetable"}</p>
        <span className="flex items-center gap-1 text-[11px] text-ink-3">
          <Clock3 size={11} /> reply by {new Date(item.deadline).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </span>
      </div>
      <p className="whitespace-pre-line text-[13px] leading-relaxed">{m.text.split("\n\nOptions:")[0]}</p>
      <div className="space-y-2">
        {m.offers.map((o) => (
          <OfferCard key={o.key} offer={o} selected={choice === o.key} onSelect={() => setChoice(o.key)} />
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" size="sm" icon={Check} disabled={!choice} loading={busy === "accept"} onClick={() => act("accept")}>
          {choice ? `${choosing ? "Choose" : "Accept"} option ${choice}` : "Pick an option"}
        </Button>
        {!choosing && (
          <Button variant="ghost" size="sm" icon={X} loading={busy === "reject"} onClick={() => act("reject")}>
            Decline
          </Button>
        )}
        <Link to="/inbox" className="ml-auto flex items-center gap-1 text-[12px] text-ink-3 hover:text-brand">
          {choosing ? "Reply in your own words" : "Another time or your own words"} <ArrowUpRight size={12} />
        </Link>
      </div>
      {error && <p className="text-[12.5px] text-bad">{error}</p>}
    </div>
  );
}
