import clsx from "clsx";
import { AlertTriangle, ArrowLeft, Cpu, Undo2 } from "lucide-react";
import { useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { EventTimeline, Lifecycle } from "../components/Lifecycle";
import { ConstraintCard, DiffTable, GroundedExplanation, MessageBubble } from "../components/Negotiation";
import { ClarifyForm } from "../components/Clarify";
import { decider, EscalationPanel } from "../components/Escalation";
import { ForwardedPanel } from "../components/Forwarded";
import { Avatar, Badge, Button, Card, CardHeader, Collapse, EmptyState, Json, Label, Meter, Skeleton, StatusBadge, Tabs, Toast } from "../components/ui";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { ago, num, pct, ROLE_LABEL, secs } from "../lib/meta";
import { study } from "../lib/study";
import type { CaseDetail as Detail } from "../lib/types";
import { useStudySession } from "../components/StudyBanner";

type Tab = "summary" | "negotiation" | "details";
const OLD_TABS: Record<string, Tab> = { overview: "summary", change: "summary", parse: "details", policy: "details", trace: "details" };

/** A request in full: lifecycle, reply, negotiation, parse, policy and solver. ``embedded``: shown in a
 * panel beside its conversation (the requests chat), so without the back link and the title block, and
 * with the tab kept locally instead of in the address. */
export default function CaseDetail({ id: given, embedded = false }: { id?: string; embedded?: boolean } = {}) {
  const { id: routed = "" } = useParams();
  const id = given ?? routed;
  const [params, setParams] = useSearchParams();
  const [localTab, setLocalTab] = useState<Tab>("summary");
  const raw = embedded ? localTab : (params.get("tab") ?? "summary");
  const tab: Tab = OLD_TABS[raw] ?? (raw as Tab);
  const setTab = (t: Tab) => (embedded ? setLocalTab(t) : setParams(t === "summary" ? {} : { tab: t }, { replace: true }));
  const { data: c, error, refresh } = useApi(() => api.case(id), [id], 1500);
  const [toast, setToast] = useState<string | null>(null);

  if (error) return <EmptyState icon={AlertTriangle} title="Cannot open this request" text={error.message} />;
  if (!c)
    return (
      <div className="space-y-4">
        <Skeleton className="h-20" />
        <Skeleton className="h-64" />
      </div>
    );

  const msgs = c.outcome?.messages ?? [];
  return (
    <>
      {!embedded && (
      <>
      <Link to="/requests" className="mb-4 inline-flex items-center gap-1.5 text-[13px] text-ink-3 hover:text-ink">
        <ArrowLeft size={14} /> Requests
      </Link>

      <div className="mb-5 flex flex-wrap items-start justify-between gap-4 border-b border-line pb-5">
        <div className="min-w-0 max-w-3xl">
          <h1 className="font-serif text-[22px] leading-snug font-semibold">“{c.text}”</h1>
          <p className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[13px] text-ink-3">
            <Avatar name={c.sender_name} id={c.sender} size={20} />
            <span className="text-ink-2">{c.sender_name}</span>
            <span>· {ROLE_LABEL[c.role] ?? c.role}</span>
            <span>· via {c.channel}</span>
            <span>· {ago(c.received_at)}</span>
            <span className="font-mono text-[12px]">· {c.id}</span>
          </p>
        </div>
        <div className="flex items-center gap-2">
          {c.can_withdraw && (
            <Withdraw
              c={c}
              onDone={(msg) => {
                setToast(msg);
                void refresh();
              }}
            />
          )}
          <StatusBadge status={c.status} live={c.running} />
        </div>
      </div>
      </>
      )}

      <Card className="mb-6 px-5 py-4">
        <Lifecycle events={c.events} status={c.status} live={c.running} />
      </Card>

      {c.outcome?.escalation && (
        <div className="mb-6">
          <EscalationPanel
            c={c}
            onDecided={(msg) => {
              setToast(msg);
              void refresh();
            }}
          />
        </div>
      )}

      {c.forwarded_to === "coordinator" && (
        <div className="mb-6">
          <ForwardedPanel
            c={c}
            onHandled={(msg) => {
              setToast(msg);
              void refresh();
            }}
          />
        </div>
      )}

      <div className="mb-5">
        <Tabs
          tabs={[
            { id: "summary", label: "Summary" },
            { id: "negotiation", label: "Negotiation", count: msgs.length },
            { id: "details", label: "Details" },
          ]}
          value={tab}
          onChange={setTab}
        />
      </div>

      {tab === "summary" && <Summary c={c} />}
      {tab === "negotiation" && <Negotiation c={c} />}
      {tab === "details" && <Details c={c} />}
      <Toast message={toast} onDone={() => setToast(null)} />
    </>
  );
}

/** The sender takes the request back while nothing it asked for is in the published timetable. */
function Withdraw({ c, onDone }: { c: Detail; onDone: (msg: string) => void }) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const go = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.withdraw(c.id);
      setConfirming(false);
      onDone("Withdrawn. Anything it set in motion has been undone.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  if (!confirming)
    return (
      <Button size="sm" icon={Undo2} onClick={() => setConfirming(true)}>
        Withdraw
      </Button>
    );
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-[13px] text-ink-2">{error ?? "Withdraw this request?"}</span>
      <Button size="sm" onClick={() => setConfirming(false)} disabled={busy}>
        Keep it
      </Button>
      <Button size="sm" variant="danger" icon={Undo2} loading={busy} onClick={go}>
        Withdraw
      </Button>
    </div>
  );
}

function outcomeText(c: Detail): string | null {
  if (c.extra?.placement)
    return `An extra ${c.extra.label} class fits on ${c.extra.placement.text}, week ${c.extra.week}. It was placed in a free hour, so no other class moves.`;
  const o = c.outcome;
  if (!o) return null;
  if (o.status === "agreed") {
    const who = o.concessions.map((x) => x.name).join(", ");
    return `Resolved by negotiation in ${o.rounds} round${o.rounds === 1 ? "" : "s"}${who ? `; ${who} agreed to move` : ""}.`;
  }
  if (o.status === "escalated") return `Escalated to ${o.escalation ? decider(o.escalation.to, o.escalation.to_name) : "someone with authority"}: ${o.escalation?.reason ?? "no agreement"}.`;
  if (o.step === 2) return "Fitted into the timetable directly; some preferences could not be kept and their owners were told.";
  return "Fitted into the timetable directly; nobody had to give anything up.";
}

function Summary({ c }: { c: Detail }) {
  const f = c.fairness;
  const outcome = outcomeText(c);
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <UnderstoodCheck c={c} />
      <Card>
        <CardHeader title={`Reply to ${c.sender_name}`} />
        <div className="p-5">
          {c.status === "clarification_requested" ? (
            <ClarifyForm caseId={c.id} sender={c.sender} question={c.reply || "Could you give more detail?"} />
          ) : null}
          {c.status === "clarification_requested" && c.reply ? null : c.reply ? (
            <p className="text-[14.5px] leading-relaxed">{c.reply}</p>
          ) : (
            <p className="text-[13.5px] text-ink-3">Still working on it…</p>
          )}
          {Object.keys(c.notices).length > 0 && (
            <div className="mt-5">
              <Label>Also told (only people affected)</Label>
              <ul className="space-y-1.5">
                {Object.entries(c.notices).map(([who, text]) => (
                  <li key={who} className="text-[13px] text-ink-2">
                    <span className="font-mono text-[12px] text-ink-3">{who}</span> {text}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </Card>

      <Card>
        <CardHeader title="What happened" />
        <div className="space-y-3 p-5 text-[14px] leading-relaxed">
          {outcome ? <p>{outcome}</p> : <p className="text-ink-3">The request did not reach the solver.</p>}
          {c.policy && (
            <p className="text-ink-2">
              Policy check: <span className="font-medium text-ink">{c.policy.verdict.replace("_", " ")}</span>
              {c.policy.cited.length ? ` under ${c.policy.cited.map((x) => x.cite).join(", ")}` : ""}.
              {c.policy.obligations.length ? " A make-up class is required." : ""}
            </p>
          )}
          {f.gini_after !== undefined && (
            <p className="text-ink-2">
              Fairness: concession Gini {num(f.gini_before ?? 0, 2)} → <span className="font-medium text-ink">{num(f.gini_after ?? 0, 2)}</span>
              {f.preferences_lost?.length ? `; preferences not met: ${f.preferences_lost.join(", ")}` : ""}.
            </p>
          )}
        </div>
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader
          title={c.proposal ? `Change to the timetable (version ${c.proposal.version}${c.proposal.week ? `, week ${c.proposal.week}` : ""})` : "Change to the timetable"}
          subtitle={c.proposal ? "Classes that move, compared with the version it builds on" : undefined}
        />
        <div className="px-5 pb-4 pt-2">{c.proposal ? <DiffTable rows={c.proposal.diff} /> : <p className="py-2 text-[13.5px] text-ink-3">No change to the timetable.</p>}</div>
      </Card>
    </div>
  );
}

function Negotiation({ c }: { c: Detail }) {
  const [msgIdx, setMsgIdx] = useState(0);
  const o = c.outcome;
  if (!o) return <EmptyState icon={Cpu} title="Not solved yet" />;
  return (
    <div className="space-y-6">
      <Card className="grid grid-cols-2 gap-4 p-5 sm:grid-cols-4">
        {[
          ["Outcome", o.status.charAt(0).toUpperCase() + o.status.slice(1)],
          ["Step on the ladder", `${o.step} of 4`],
          ["Rounds", String(o.rounds)],
          ["Solver", o.solver ? `${o.solver.status} · ${secs(o.solver.wall_time)}` : "–"],
        ].map(([k, v]) => (
          <div key={k}>
            <p className="text-[12.5px] text-ink-3">{k}</p>
            <p className="mt-0.5 text-[15px] font-medium">{v}</p>
          </div>
        ))}
      </Card>

      {!o.mus_log.length && !o.escalation && (
        <p className="text-[14px] text-ink-2">
          There was no conflict: every hard constraint fitted, so the solver changed the timetable directly{o.step === 2 ? " and told the owners of preferences it could not keep" : ""}.
        </p>
      )}

      {o.mus_log.length > 0 && (
        <Card>
          <CardHeader title="The conflict" subtitle="The smallest set of constraints that cannot all hold (from CP-SAT), round by round" />
          <div className="space-y-5 p-5">
            {o.mus_log.map((mus, r) => (
              <div key={r}>
                {o.mus_log.length > 1 && <Label>Round {r + 1}</Label>}
                <div className="grid gap-2 md:grid-cols-2">
                  {mus.map((k, i) => (
                    <ConstraintCard key={k.id} c={k} highlight index={i} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {o.messages.length > 0 && (
        <Card>
          <CardHeader title="Messages" subtitle="Each offers alternatives the solver has already checked" />
          <div className="space-y-6 p-5">
            {o.messages.map((m, i) => (
              <MessageBubble key={i} m={m} reply={o.replies[i] ? { ...o.replies[i], decision: o.replies[i].decision ?? null } : undefined} />
            ))}
          </div>
        </Card>
      )}

      {o.messages.length > 0 && (
        <Card>
          <CardHeader
            title="Why each sentence is true"
            subtitle="Each sentence of the message and the solver fact it rests on. A sentence whose days, numbers or names are not in its facts is struck out."
            action={
              o.messages.length > 1 && (
                <Tabs tabs={o.messages.map((_, i) => ({ id: String(i), label: `Round ${i + 1}` }))} value={String(msgIdx)} onChange={(v) => setMsgIdx(Number(v))} />
              )
            }
          />
          <div className="p-5">
            <GroundedExplanation m={o.messages[msgIdx]} />
          </div>
        </Card>
      )}

    </div>
  );
}

function Details({ c }: { c: Detail }) {
  const r = c.routing && "tau" in c.routing ? c.routing : null;
  const p = c.parse;
  const pol = c.policy;
  const stages = Object.entries(c.timings);
  const maxT = Math.max(0.001, ...stages.map(([, v]) => v));
  return (
    <div className="space-y-3">
      <Collapse title={`Routing: ${c.route === "fast" ? "fast path" : c.route ? "System Two" : "not routed"}`} defaultOpen>
        {r ? (
          <div className="space-y-4">
            <p className="text-[13.5px] text-ink-2">
              System One read it as <span className="font-medium text-ink">{r.request_type.replace("_", " ")}</span> →{" "}
              <span className="font-medium text-ink">{r.action}</span> in {num(r.latency_ms, 1)} ms, and{" "}
              {r.fast_path ? "was confident enough to use the fast path." : "sent it to the slower, more capable parser."}
            </p>
            {(
              [
                ["Type confidence", r.type_p],
                ["Action confidence", r.action_p],
              ] as const
            ).map(([label, v]) => (
              <div key={label} className="max-w-md">
                <div className="mb-1 flex justify-between text-[12.5px]">
                  <span className="text-ink-2">{label}</span>
                  <span className="font-mono">{pct(v, 1)}</span>
                </div>
                <div className="relative">
                  <Meter value={v} tone={v >= r.tau ? "ok" : "warn"} />
                  <div className="absolute -top-1 h-3.5 w-px bg-ink" style={{ left: `${r.tau * 100}%` }} title={`τ = ${r.tau}`} />
                </div>
              </div>
            ))}
            <p className="text-[12px] text-ink-3">The mark is the threshold τ = {r.tau}.</p>
          </div>
        ) : (
          <p className="text-[13.5px] text-ink-3">No routing decision recorded.</p>
        )}
      </Collapse>

      <Collapse title={p ? `Parser: ${p.action}, ${p.constraints.length} constraint${p.constraints.length === 1 ? "" : "s"}` : "Parser"}>
        {p ? (
          <div className="grid gap-5 lg:grid-cols-2">
            <div>
              <Label>What the request asks for</Label>
              <div className="space-y-2">
                {p.constraints.map((k, i) => (
                  <ConstraintCard key={k.id} c={k} index={i} />
                ))}
                {!p.constraints.length && <p className="text-[13.5px] text-ink-3">None: the request was {p.action === "answer" ? "a question" : p.action}.</p>}
              </div>
              {(p.refusal || p.errors.length > 0) && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {p.refusal && <Badge tone="bad">refused: {p.refusal}</Badge>}
                  {p.errors.map((e) => (
                    <Badge key={e} tone="warn">
                      {e}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
            <div>
              <Label>Raw model output</Label>
              <Json value={p.output} className="max-h-[420px]" />
            </div>
          </div>
        ) : (
          <p className="text-[13.5px] text-ink-3">Not parsed.</p>
        )}
      </Collapse>

      <Collapse title={pol ? `Policy check: ${pol.verdict.replace("_", " ")}` : "Policy check"}>
        {pol ? (
          <div className="grid gap-5 lg:grid-cols-2">
            <div className="space-y-3 text-[13.5px] leading-relaxed">
              <p>{pol.answer || pol.explanation}</p>
              {pol.alternative && (
                <p className="text-ink-2">
                  <span className="font-medium text-ok">Allowed alternative: </span>
                  {pol.alternative}
                </p>
              )}
              {pol.obligations.length > 0 && <p className="text-ink-2">Obligation: {pol.obligations.join(", ")} (a make-up class within two weeks).</p>}
              {pol.uncited_escalation && <p className="text-warn">The model denied without a valid citation, so a person decides.</p>}
              <p className="text-[12px] text-ink-3">Policy agent: {pol.mode}</p>
            </div>
            <div>
              <Label>Rules retrieved (citations must come from these)</Label>
              <ol className="space-y-1">
                {pol.retrieved.map((rid, i) => {
                  const cited = pol.cited.find((x) => x.id === rid);
                  return (
                    <li key={rid} className={clsx("flex items-center gap-2.5 rounded-md border px-3 py-2 text-[13px]", cited ? "border-brand/40" : "border-line")}>
                      <span className="w-4 font-mono text-[11px] text-ink-3">{i + 1}</span>
                      <span className="font-mono text-[12px]">{rid}</span>
                      <span className="flex-1 truncate text-ink-2">{cited?.cite ?? ""}</span>
                      {cited && <Badge tone="brand">cited</Badge>}
                      {pol.obligations.includes(rid) && <Badge tone="info">obligation</Badge>}
                    </li>
                  );
                })}
              </ol>
              {pol.hallucinated.length > 0 && <p className="mt-2 text-[12.5px] text-bad">Dropped invented citations: {pol.hallucinated.join(", ")}</p>}
            </div>
          </div>
        ) : (
          <p className="text-[13.5px] text-ink-3">Refused, forwarded and clarification requests stop before the policy check.</p>
        )}
      </Collapse>

      <Collapse title="Time per stage">
        <div className="max-w-2xl space-y-2">
          {stages.map(([k, v]) => (
            <div key={k} className="grid grid-cols-[150px_minmax(0,1fr)_70px] items-center gap-3 text-[13px]">
              <span className="text-ink-2">{k.replace(/_/g, " ")}</span>
              <Meter value={v} max={maxT} />
              <span className="text-right font-mono text-[12px] text-ink-2">{secs(v)}</span>
            </div>
          ))}
          {!stages.length && <p className="text-[13.5px] text-ink-3">No timings yet.</p>}
        </div>
      </Collapse>

      <Collapse title={`Event trace (${c.events.length})`}>
        <EventTimeline events={c.events} />
      </Collapse>

      {c.outcome?.concessions.length ? (
        <Collapse title="Ledger entries recorded">
          <Json value={c.outcome.concessions} />
        </Collapse>
      ) : null}
    </div>
  );
}

/** Study: a pilot participant says whether the system understood their own request. */
function UnderstoodCheck({ c }: { c: Detail }) {
  const session = useStudySession();
  const [answer, setAnswer] = useState<string | null>(null);
  const [comment, setComment] = useState("");
  if (!session || c.sender !== session.persona || c.running || !c.parse) return null;
  const understood = c.parse.constraints.map((k) => k.text);
  const save = async (value: string) => {
    setAnswer(value);
    await study.live({ kind: "understood", case: c.id, text: c.text, reading: understood.join(" ") || c.parse?.action, value,
                       ...(comment.trim() ? { comment: comment.trim() } : {}) });
  };
  return (
    <Card className="border-brand/30 lg:col-span-2">
      <CardHeader title="Study: did the system understand you?" subtitle="Compare what you wrote with how it was read" />
      <div className="space-y-3 p-5 text-[14px]">
        {understood.length ? (
          <ul className="list-disc space-y-1 pl-5 text-ink-2">
            {understood.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        ) : (
          <p className="text-ink-2">It was read as: {c.parse.action}{c.parse.refusal ? ` (${c.parse.refusal})` : ""}.</p>
        )}
        {answer ? (
          <p className="text-[13.5px] text-ok">Saved: {answer}. Thank you.</p>
        ) : (
          <>
            <input value={comment} onChange={(e) => setComment(e.target.value)} placeholder="What was missed or wrong? (optional)"
                   className="w-full rounded-md border border-line bg-panel px-3 py-2 text-[13.5px] outline-none focus:border-brand/60" />
            <div className="flex flex-wrap gap-2">
              {(["yes", "partly", "no"] as const).map((v) => (
                <button key={v} onClick={() => void save(v)}
                        className="rounded-md border border-line px-3.5 py-1.5 text-[13.5px] capitalize hover:border-brand/60 hover:bg-panel-2">
                  {v === "yes" ? "Yes, exactly" : v === "partly" ? "Partly" : "No"}
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </Card>
  );
}
