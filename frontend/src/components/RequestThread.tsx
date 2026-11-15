import clsx from "clsx";
import { ArrowRight, Check, CheckCircle2, Clock3, HelpCircle, Undo2, X, XCircle, type LucideIcon } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { ago, EVENT_LABEL, OUTCOME_DOT, OUTCOME_TEXT, outcomeOf, PLAIN_STATUS, stepOutcome } from "../lib/meta";
import type { CaseDetail, DiffRow, EventRow, InboxItem } from "../lib/types";
import { changedWeeks, groupWeeks, weeksLabel } from "../lib/weeks";
import { OfferCard, SwapDetails } from "./Negotiation";
import { ReplyBubble, StepList, Tag, YouBubble } from "./thread/Bubble";
import { ReplyText } from "./thread/ReplyText";
import { Button, LiveDot } from "./ui";

/** Steps worth a line in the conversation; the rest (parsing, compiling, solving) stay on the case page. */
const SHOWN = new Set([
  "held",
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

/** The reply's colour and label, from where the request stands now. */
function verdict(c: CaseDetail): { tone: "ok" | "bad" | "warn" | "plain"; icon?: LucideIcon; label?: string } {
  const o = outcomeOf(c.status);
  if (o === "ok")
    return { tone: "ok", icon: CheckCircle2, label: c.status === "answered" ? "Answered" : c.proposal ? "Approved · in the timetable" : "Done" };
  if (o === "bad") return { tone: "bad", icon: XCircle, label: c.status === "refused" ? "Couldn't be done" : "Not approved" };
  if (o === "pending") return { tone: "warn", icon: Clock3, label: PLAIN_STATUS[c.status].label };
  return { tone: "plain" };
}

/** A reply written while the change waited for approval still says so once it is published: say what happened. */
function current(reply: string, c: CaseDetail): string {
  if (c.status !== "published") return reply;
  return reply
    .replace(/The change is waiting for the coordinator's approval\./, "The coordinator approved it, and it is now in the timetable.")
    .replace(/It is waiting for the coordinator's approval\./, "The coordinator approved it, and it is now in the timetable.");
}

function stepText(e: EventRow, c: CaseDetail): string {
  if (e.kind === "published" && e.unmet) return "Nothing changed";
  if (e.kind === "answered" && c.part_details?.length) return `Split into ${c.part_details.length} requests`;
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
  // the reply card is labelled with how it ended ("Answered"): the same word as the last step only repeats it
  if (!c.running && c.reply && !c.part_details?.length && steps.length
      && stepText(steps[steps.length - 1], c) === verdict(c).label) steps.pop();
  const asking = c.status === "clarification_requested" && c.sender === user?.id;
  const forMe = (c.inbox ?? []).filter((i) => i.to === user?.id);
  const waitingOnMe = forMe.some((i) => !i.reply);
  const rules = [...(c.policy?.cited ?? []).map((r) => r.cite), ...(c.policy?.obligations ?? []).map((id) => `${id} (applies)`)];

  return (
    <div className="space-y-3">
      <YouBubble text={c.text} note={ago(c.received_at)} />

      <StepList
        steps={steps.map((e, i) => ({
          key: e.n,
          text: stepText(e, c),
          // the step the request stands at is coloured; earlier ones (a wait that is over) stay grey
          outcome: c.part_details?.length && e.kind === "answered" ? null
            : i === steps.length - 1 || e.kind === "denied" || e.kind === "refused" ? stepOutcome(e.kind) : null,
        }))}
      />

      {c.running ? (
        <p className="flex items-center gap-2 pl-1 text-[12.5px] text-ink-2">
          {waitingOnMe ? (
            <>
              <LiveDot tone="brand" pulse={false} /> Waiting for your answer
            </>
          ) : (
            <>
              <LiveDot tone="warn" pulse /> {c.held
                ? "Waiting for the model that reads requests; nothing is done with it until then…"
                : c.status === "negotiating" ? "Waiting for a colleague's reply…" : "Working on it…"}
            </>
          )}
        </p>
      ) : c.part_details?.length ? (
        <SplitParts parts={c.part_details} />
      ) : (
        c.reply && (
          <ReplyBubble
            tone={asking ? "warn" : verdict(c).tone}
            icon={asking ? HelpCircle : verdict(c).icon}
            label={asking ? "A question for you. Answer it below." : verdict(c).label}
            footer={
              (c.answer || rules.length > 0) && (
                <>
                  {c.answer && <p>From {c.answer.by_name}</p>}
                  {rules.length > 0 && (
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      {rules.map((r) => (
                        <Tag key={r} title="The rule this rests on">
                          {r}
                        </Tag>
                      ))}
                    </div>
                  )}
                </>
              )
            }
          >
            <ReplyText text={current(c.reply, c)} />
          </ReplyBubble>
        )
      )}

      {forMe.map((i) => (
        <Options key={i.id} item={i} onDone={onChanged} />
      ))}

      <Change c={c} me={user?.id} />

      <div className="flex items-center gap-2 pt-1">
        {c.can_withdraw && <Withdraw id={c.id} onDone={onChanged} />}
      </div>
    </div>
  );
}

/** A message that asked for several things: one row per request it was split into, with where that request
 * stands and, while it waits on the sender, its question and a way to answer it. */
function SplitParts({ parts }: { parts: NonNullable<CaseDetail["part_details"]> }) {
  const navigate = useNavigate();
  return (
    <div className="max-w-[92%] rounded-lg border border-line">
      <div className="border-b border-line px-3 py-2">
        <p className="text-[12.5px] font-medium">Your message asks for {parts.length} things</p>
        <p className="text-[11.5px] text-ink-3">Each is handled as a request of its own.</p>
      </div>
      <ol className="divide-y divide-line/60">
        {parts.map((p, i) => {
          const o = outcomeOf(p.status);
          const asking = p.status === "clarification_requested";
          return (
            <li key={p.id} className="flex gap-3 px-3 py-2.5">
              <span className="mt-0.5 grid size-5 shrink-0 place-items-center rounded-full border border-line text-[11px] text-ink-2">{i + 1}</span>
              <div className="min-w-0 flex-1 space-y-1.5">
                <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                  <p className="text-[13px] font-medium">{p.text.replace(/^./, (x) => x.toUpperCase())}</p>
                  <span className={clsx("flex shrink-0 items-center gap-1.5 text-[11.5px] font-medium", OUTCOME_TEXT[o])}>
                    <span className={clsx("size-1.5 rounded-full", OUTCOME_DOT[o])} />
                    {PLAIN_STATUS[p.status].label}
                  </span>
                </div>
                {p.reply && (
                  <p className={clsx("rounded-md px-2.5 py-1.5 text-[12.5px] leading-relaxed", asking ? "bg-warn/8 text-ink" : "bg-panel-2/70 text-ink-2")}>
                    {asking && <HelpCircle size={12} className="mr-1 inline -translate-y-px text-warn" />}
                    {p.reply}
                  </p>
                )}
                <Button size="sm" variant={asking ? "primary" : "ghost"} icon={asking ? undefined : ArrowRight} onClick={() => navigate(`/requests/${p.id}`)}>
                  {asking ? "Answer" : "Open"}
                </Button>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

/** The change a request makes to the timetable, as times: each class before and after, the weeks it is
 * for, and how far it has got. Shown in the conversation, since a change for one week (or several) does
 * not fit on a one-week timetable. */
function Change({ c, me }: { c: CaseDetail; me?: string }) {
  // every week of the request (an absence in weeks 10-12 changes each), weeks that change the same way together
  const groups = groupWeeks(changedWeeks(c.proposal)).filter((g) => g.diff.length);
  if (!groups.length || ["denied", "refused", "withdrawn"].includes(c.status)) return null;
  const all = groups.flatMap((g) => g.weeks);
  const state =
    c.status === "published"
      ? `In the timetable · version ${c.proposal!.version}`
      : c.status === "awaiting_approval"
        ? "Waiting for the coordinator's approval"
        : c.status === "escalated"
          ? "Waiting for a decision"
          : "Proposed";
  const where = (p: NonNullable<DiffRow["before"]>) => `${p.day} ${p.time} · ${p.room}`;
  return (
    <div className="max-w-[92%] rounded-lg border border-line">
      <div className="flex items-baseline justify-between gap-3 border-b border-line px-3 py-2">
        <p className="text-[12.5px] font-medium">Requested change</p>
        <p className="text-[11.5px] text-ink-3">
          {all.some((w) => w != null) ? `${weeksLabel(all).replace(/^w/, "W")} only` : "Every week"}
        </p>
      </div>
      {groups.map((g) => (
        <div key={g.versions.join()}>
          {groups.length > 1 && (
            <p className="border-b border-line/60 px-3 pt-2 pb-1 text-[11px] font-medium uppercase tracking-wide text-ink-3">{weeksLabel(g.weeks)}</p>
          )}
          <ul className="divide-y divide-line/60">
            {g.diff.map((r) => (
              <li key={r.session} className="px-3 py-2">
                <p className="truncate text-[13px] font-medium">
                  {r.session_name.replace(/^(the|my) /, "").replace(/\s*\([A-Z]-[\w-]+\)$/, "")}
                  {r.faculty && r.faculty !== me && <span className="font-normal text-ink-3"> · {r.faculty_name}</span>}
                </p>
                <p className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[12.5px]">
                  <span className={clsx("text-ink-3", r.before && "line-through decoration-bad/50")}>{r.before ? where(r.before) : "new"}</span>
                  <ArrowRight size={12} className="shrink-0 text-ink-3" />
                  <span className={r.after ? "font-medium text-ok" : "font-medium text-bad"}>
                    {r.after ? where(r.after) : g.weeks.some((w) => w != null) ? `not held in ${weeksLabel(g.weeks)}` : "not held (make-up owed)"}
                  </span>
                </p>
              </li>
            ))}
          </ul>
        </div>
      ))}
      <p className={clsx("flex items-center gap-1.5 border-t border-line px-3 py-1.5 text-[11.5px]", c.status === "published" ? "text-ok" : "text-warn")}>
        {c.status === "published" ? <CheckCircle2 size={12} /> : <Clock3 size={12} />} {state}
      </p>
    </div>
  );
}

/** Take a request back (with a confirm): offered to its sender wherever the request is shown. */
export function Withdraw({ id, onDone }: { id: string; onDone: () => void }) {
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
 * clash): chosen here, through the same reply the Inbox sends. A reply in their own words (another time,
 * a question) goes through the composer under the thread. */
export function Options({ item, onDone }: { item: InboxItem; onDone: () => void }) {
  const { user } = useAuth();
  const [choice, setChoice] = useState<string | null>(null);
  const [busy, setBusy] = useState<"accept" | "reject" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const m = item.message;
  const choosing = m.explanation_mode === "choice"; // their own request fits in more than one way
  const swap = m.explanation_mode === "swap" && m.offers.length === 1; // a colleague asks to swap: yes or no
  const act = async (kind: "accept" | "reject") => {
    setBusy(kind);
    setError(null);
    try {
      await api.reply(
        item.id,
        swap
          ? kind === "accept"
            ? { decision: "accept", choice: "A", text: "I agree to the swap." }
            : { decision: "reject", text: "I don't agree to the swap." }
          : kind === "accept"
            ? { decision: "accept", choice: choice!, text: `Option ${choice} works for me.` }
            : { decision: "reject", text: "None of these work for me." },
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
        {item.answered_by === "deadline" || item.reply.decision === "no_reply"
          ? swap
            ? "No answer before the deadline, so the swap was not made"
            : "No answer before the deadline, so the first option was used"
          : swap
            ? item.reply.decision === "accept"
              ? "You agreed to the swap"
              : "You declined the swap"
          : item.reply.decision === "accept" && item.reply.choice
            ? `You chose option ${item.reply.choice}`
            : item.reply.decision === "reject"
              ? "You declined these options"
              : `You answered: ${item.reply.decision}`}
        {item.answered_by === "simulator" ? " (simulated)" : ""}
      </p>
    );

  const deadline = (
    <span className="flex items-center gap-1 text-[11px] text-ink-3">
      <Clock3 size={11} /> reply by {new Date(item.deadline).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
    </span>
  );

  if (swap) {
    const from = m.offers[0].placements.find((p) => p.faculty !== user?.id)?.faculty_name;
    return (
      <div className="space-y-2.5 rounded-lg border border-brand/40 bg-brand/5 p-3">
        <div className="flex items-center justify-between gap-2">
          <p className="text-[12.5px] font-medium text-brand">{from ? `${from} asks to swap classes` : "A request to swap classes"}</p>
          {deadline}
        </div>
        <p className="text-[13.5px] leading-relaxed">{m.text.split("\n\n")[0]}</p>
        <SwapDetails offer={m.offers[0]} me={user?.id} />
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="primary" size="sm" icon={Check} loading={busy === "accept"} onClick={() => act("accept")}>
            Agree to swap
          </Button>
          <Button variant="ghost" size="sm" icon={X} loading={busy === "reject"} onClick={() => act("reject")}>
            Decline
          </Button>
          <span className="ml-auto text-[11.5px] text-ink-3">or reply in your own words below</span>
        </div>
        {error && <p className="text-[12.5px] text-bad">{error}</p>}
      </div>
    );
  }

  return (
    <div className="space-y-2.5 rounded-lg border border-brand/40 bg-brand/5 p-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-[12.5px] font-medium text-brand">{choosing ? "Choose a time" : "A clash with your timetable"}</p>
        {deadline}
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
        <span className="ml-auto text-[11.5px] text-ink-3">or reply in your own words below</span>
      </div>
      {error && <p className="text-[12.5px] text-bad">{error}</p>}
    </div>
  );
}
