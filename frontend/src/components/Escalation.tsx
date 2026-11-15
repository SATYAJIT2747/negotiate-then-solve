import clsx from "clsx";
import { BookOpen, Bot, Check, CircleSlash, Gavel, History, ListX, MessageSquareQuote, MessagesSquare, TriangleAlert, Undo2, X } from "lucide-react";
import { useState } from "react";
import { api } from "../lib/api";
import type { CaseDetail } from "../lib/types";
import { ConstraintCard } from "./Negotiation";
import { Badge, Button, Card, Collapse, Label } from "./ui";

/** Who an escalation names, in words: "coordinator", "dean", "hod", "exam_cell" or the heads' names. */
export function decider(to: string, toName: string): string {
  return { coordinator: "the timetable office", dean: "the dean", hod: "the head of department", exam_cell: "the examination cell" }[to] ?? toName;
}

const sentence = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1).replace(/\.?$/, ".") : "");

const REPLY: Record<string, { label: string; tone: "ok" | "bad" | "info" | "muted" }> = {
  accept: { label: "agreed", tone: "ok" },
  reject: { label: "declined", tone: "bad" },
  counter: { label: "proposed something else", tone: "info" },
  no_reply: { label: "did not reply", tone: "muted" },
};

/** An escalated request: why it could not be fitted, what was tried, and (for the person it was sent
 * to) the decision itself: grant, setting the conflicting commitments aside, or decline with a note. */
export function EscalationPanel({ c, onDecided }: { c: CaseDetail; onDecided?: (msg: string) => void }) {
  const o = c.outcome!;
  const e = o.escalation!;
  const conflict = o.mus_log[o.mus_log.length - 1] ?? [];
  const who = decider(e.to, e.to_name);
  const d = c.decision;
  const open = c.status === "escalated";

  return (
    <Card className="overflow-hidden">
      <div className={clsx("flex items-start gap-4 border-b border-line px-5 py-4", open ? "bg-warn/[0.06]" : "bg-panel-2/50")}>
        <span className={clsx("grid size-10 shrink-0 place-items-center rounded-full", open ? "bg-warn/15 text-warn" : "bg-panel-2 text-ink-3")}>
          {open ? <TriangleAlert size={19} /> : <Gavel size={18} />}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-[15px] font-semibold">
              {open ? (c.can_decide ? "Waiting for your decision" : `Waiting for a decision from ${who}`) : `Escalated to ${who}`}
            </h3>
            {d && <Badge tone={d.granted ? "ok" : "bad"}>{d.granted ? "granted" : "declined"} by {d.by_name}</Badge>}
          </div>
          <p className="mt-1 max-w-3xl text-[13.5px] leading-relaxed text-ink-2">
            {e.rule ? `${c.sender_name} asks for a class ${e.reason}${e.asked ? ` (${e.asked})` : ""}, which only ${who} can allow.` : sentence(e.reason)}
          </p>
          {d?.note && <p className="mt-2 text-[13px] text-ink-2">“{d.note}”</p>}
        </div>
      </div>

      {e.rule ? (
        <RuleException c={c} rule={e.rule} brief={e.text} />
      ) : (
      <div className="grid gap-0 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
        <div className="space-y-3 border-b border-line p-5 lg:border-b-0 lg:border-r">
          <Label className="flex items-center gap-1.5">
            <ListX size={13} /> What cannot all hold
          </Label>
          {conflict.length ? (
            <div className="space-y-2">
              {conflict.map((k, i) => (
                <ConstraintCard key={k.id} c={k} highlight index={i} />
              ))}
            </div>
          ) : (
            <p className="text-[13.5px] text-ink-3">The policy check sent it up before it reached the solver.</p>
          )}
        </div>
        <div className="space-y-3 p-5">
          <Label className="flex items-center gap-1.5">
            <History size={13} /> What was tried
          </Label>
          {o.messages.length ? (
            <ol className="space-y-2">
              {o.messages.map((m, i) => {
                const r = REPLY[o.replies[i]?.decision ?? "no_reply"] ?? REPLY.no_reply;
                return (
                  <li key={i} className="flex items-center gap-2.5 text-[13.5px]">
                    <span className="grid size-6 shrink-0 place-items-center rounded-full bg-panel-2 text-[11.5px] font-medium text-ink-3">{i + 1}</span>
                    <MessagesSquare size={14} className="shrink-0 text-ink-3" />
                    <span className="min-w-0 flex-1 truncate">Asked {m.to_name}</span>
                    <Badge tone={r.tone}>{r.label}</Badge>
                  </li>
                );
              })}
            </ol>
          ) : (
            <p className="text-[13.5px] leading-relaxed text-ink-2">
              Nobody could be asked to move: every way to fit it would break something only a person with the authority can change.
            </p>
          )}
          <Collapse title="Brief as written">
            <pre className="whitespace-pre-wrap font-sans text-[12.5px] leading-relaxed text-ink-3">{e.text}</pre>
          </Collapse>
        </div>
      </div>
      )}

      {c.can_simulate && <SimulateReply c={c} who={who} onDecided={onDecided} />}
      {c.can_decide && <Decide c={c} onDecided={onDecided} />}
    </Card>
  );
}

/** An exception to a rule (a class outside teaching hours): what was asked, in the sender's words and as
 * read, beside the rule that needs the decider's approval. Nothing was solved, so there is no conflict to show. */
function RuleException({ c, rule, brief }: { c: CaseDetail; rule: { id: string; number: string; title: string; text: string }; brief: string }) {
  return (
    <div className="grid gap-0 lg:grid-cols-2">
      <div className="space-y-3 border-b border-line p-5 lg:border-b-0 lg:border-r">
        <Label className="flex items-center gap-1.5">
          <MessageSquareQuote size={13} /> The request
        </Label>
        <blockquote className="border-l-2 border-line pl-3 text-[13.5px] leading-relaxed text-ink">“{c.text.trim()}”</blockquote>
        <p className="text-[12.5px] text-ink-3">{c.sender_name}</p>
        {!!c.understood?.length && (
          <div>
            <p className="text-[12px] font-medium text-ink-3">Read as</p>
            <ul className="mt-1 list-disc space-y-0.5 pl-5 text-[13px] text-ink-2">
              {c.understood.map((u) => (
                <li key={u}>{u}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
      <div className="space-y-3 p-5">
        <Label className="flex items-center gap-1.5">
          <BookOpen size={13} /> The rule
        </Label>
        <p className="text-[13.5px] font-medium">
          §{rule.number} {rule.title} <span className="font-mono text-[11.5px] font-normal text-ink-3">{rule.id}</span>
        </p>
        <p className="text-[13px] leading-relaxed text-ink-2">{rule.text}</p>
        <Collapse title="Brief as written">
          <pre className="whitespace-pre-wrap font-sans text-[12.5px] leading-relaxed text-ink-3">{brief}</pre>
        </Collapse>
      </div>
    </div>
  );
}

/** The demo's stand-in for the person deciding (the examination cell), as the inbox simulates a teacher's
 * reply: it answers as the cell would, and the request goes on like any granted one. */
function SimulateReply({ c, who, onDecided }: { c: CaseDetail; who: string; onDecided?: (msg: string) => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const go = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.simulateDecision(c.id);
      onDecided?.(`${who.charAt(0).toUpperCase() + who.slice(1)} answered (simulated): granted, re-solving it now, then it comes to Approvals.`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="flex flex-wrap items-center gap-3 border-t border-line px-5 py-3">
      <p className="text-[13px] text-ink-3">Demo: have {who} answer, as a simulated stakeholder.</p>
      <Button variant="ghost" icon={Bot} loading={busy} className="ml-auto" onClick={go}>
        Simulate reply
      </Button>
      {error && <p className="w-full text-[13px] text-bad">{error}</p>}
    </div>
  );
}

function Decide({ c, onDecided }: { c: CaseDetail; onDecided?: (msg: string) => void }) {
  const e = c.outcome!.escalation!;
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState<"grant" | "decline" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const go = async (grant: boolean) => {
    setBusy(grant ? "grant" : "decline");
    setError(null);
    try {
      await api.decide(c.id, grant, note);
      onDecided?.(
        grant
          ? e.rule?.id === "P-HOURS" ? `Granted; ${c.sender_name} has been told and the timetable office books the room.` : "Granted: re-solving it now, then it comes to Approvals."
          : `Declined; ${c.sender_name} has been told.`,
      );
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="border-t border-line bg-panel-2/40 p-5">
      <div className="grid gap-4 md:grid-cols-2">
        <div className={clsx("rounded-md border p-4", e.cannot_grant ? "border-line opacity-70" : "border-ok/30 bg-ok/[0.04]")}>
          <p className="flex items-center gap-2 text-[13.5px] font-semibold">
            <Check size={15} className="text-ok" /> Grant
          </p>
          {e.rule ? (
            <p className="mt-1.5 text-[13px] leading-relaxed text-ink-2">{e.grant}</p>
          ) : e.cannot_grant ? (
            <p className="mt-1.5 text-[13px] text-ink-3">Not possible: {e.cannot_grant}.</p>
          ) : (
            <>
              <p className="mt-1.5 text-[13px] leading-relaxed text-ink-2">
                Sets aside {e.set_aside.length === 1 ? "this commitment" : `these ${e.set_aside.length} commitments`} and fits the request in. Nothing is
                published yet: the change still comes to Approvals.
              </p>
              <ul className="mt-2.5 space-y-1">
                {e.set_aside.map((k) => (
                  <li key={k.id} className="flex items-start gap-2 text-[12.5px] text-ink-2">
                    <Undo2 size={13} className="mt-0.5 shrink-0 text-ink-3" />
                    <span>
                      {k.text}
                      {k.owner_name && <span className="text-ink-3"> · {k.owner_name}</span>}
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
        <div className="rounded-md border border-bad/25 bg-bad/[0.03] p-4">
          <p className="flex items-center gap-2 text-[13.5px] font-semibold">
            <X size={15} className="text-bad" /> Decline
          </p>
          <p className="mt-1.5 text-[13px] leading-relaxed text-ink-2">
            {e.rule
              ? `The class stays where it is in the timetable and ${c.sender_name} is told, with your note below.`
              : `The timetable stays as it is and ${c.sender_name} is told, with your note below.`}
          </p>
        </div>
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <input
          value={note}
          onChange={(ev) => setNote(ev.target.value)}
          placeholder="A note for the sender (optional)"
          className="h-9 min-w-[220px] flex-1 rounded-md border border-line bg-panel px-3 text-sm outline-none focus:border-brand/50"
        />
        <Button variant="danger" icon={CircleSlash} loading={busy === "decline"} disabled={!!busy} onClick={() => go(false)}>
          Decline
        </Button>
        <Button variant="primary" icon={Gavel} loading={busy === "grant"} disabled={!!busy || !!e.cannot_grant} onClick={() => go(true)}>
          Grant
        </Button>
      </div>
      {error && <p className="mt-2 text-[13px] text-bad">{error}</p>}
    </div>
  );
}
