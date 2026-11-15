import clsx from "clsx";
import {
  Activity,
  ArrowRight,
  BookOpenText,
  Brain,
  ChevronDown,
  CircleAlert,
  CircleCheck,
  CircleX,
  Cpu,
  Database,
  Handshake,
  Info,
  Radio,
  Route,
  ShieldCheck,
  Timer,
  TriangleAlert,
  Wifi,
  WifiOff,
  type LucideIcon,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { ActivityFeed } from "../components/Lifecycle";
import { Badge, Card, CardHeader, LiveDot, Meter, PageHeader, Skeleton } from "../components/ui";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { EVENT_LABEL, pct, secs } from "../lib/meta";
import type { HealthCheck, Observability as Data } from "../lib/types";
import { useLiveEvents } from "./Dashboard";

/* System health: is it working, and what should we improve next? The checks come from the server
 * (web.app._health), worst first, each with what it means and a next step; the rest of the page is the
 * evidence behind them, every number with a label saying what it counts. */

const LEVEL: Record<HealthCheck["level"], { icon: LucideIcon; text: string; bg: string; border: string; label: string }> = {
  bad: { icon: CircleX, text: "text-bad", bg: "bg-bad/10", border: "border-bad/40", label: "Fix now" },
  warn: { icon: TriangleAlert, text: "text-warn", bg: "bg-warn/10", border: "border-warn/40", label: "Look into" },
  info: { icon: Info, text: "text-info", bg: "bg-info/10", border: "border-line", label: "Good to know" },
  ok: { icon: CircleCheck, text: "text-ok", bg: "bg-ok/10", border: "border-line", label: "Working" },
};

/** Where a request ended up: label, what it means, colour. */
const OUTCOME: Record<string, [string, string, string]> = {
  published: ["Published", "approved and in the timetable", "bg-ok"],
  awaiting_approval: ["Waiting for approval", "the office has not decided yet", "bg-brand"],
  negotiating: ["Negotiating", "waiting for a colleague's reply", "bg-violet"],
  escalated: ["Escalated", "a person with authority decides", "bg-warn"],
  clarification_requested: ["Asked back", "waiting for the sender to clarify", "bg-info"],
  denied: ["Denied by policy", "breaks a rule; the reply cites it", "bg-bad"],
  refused: ["Refused", "no authority, or only instructions", "bg-bad/60"],
  forwarded: ["Passed to the office", "a report or a message for a person", "bg-teal"],
  answered: ["Answered", "a question, or met without changes", "bg-ok/50"],
  withdrawn: ["Withdrawn", "taken back by the sender", "bg-ink-3/50"],
  in_progress: ["Being handled", "reading or solving right now", "bg-ink-3/30"],
};

const STAGE: Record<string, [string, string]> = {
  parse: ["Reading the message", "the model (or rules) turning the message into intents"],
  policy: ["Checking policy", "the handbook rules and the sender's authority"],
  solve_and_negotiate: ["Solving and negotiating", "CP-SAT and the ladder; includes waiting for colleagues' replies"],
};

const READER: Record<string, [string, string]> = {
  model: ["The model", "the fine-tuned intents reader"],
  rules: ["Rule stand-ins", "while the model was away, or System One's fast path"],
  form: ["A form", "made with clicks: nothing to read"],
  none: ["Not read", "only instructions to the system"],
  unknown: ["Not recorded", "older requests"],
};

/** "system_two+intents+swap" -> "Read by the model, then the swap flow". */
function routeLabel(route: string): string {
  if (!route) return "Not routed (older requests)";
  const [base, ...rest] = route.split("+");
  const head = { system_two: "Read by the model", fast: "Fast path (rules)", form: "Form", refused: "Refused before reading" }[base] ?? base;
  const flows = rest.filter((r) => r !== "intents");
  return flows.length ? `${head}, then the ${flows.join(", ")} flow` : head;
}

export default function Observability() {
  const { data: o } = useApi(() => api.observability(), [], 3000);
  const events = useLiveEvents(1500);
  if (!o)
    return (
      <>
        <PageHeader title="System health" subtitle="Is it working, and what should we improve next?" />
        <Skeleton className="h-40" />
        <Skeleton className="mt-6 h-80" />
      </>
    );
  const h = o.health;
  const attention = h.checks.filter((c) => c.level === "bad" || c.level === "warn");
  const up = o.uptime_s;

  return (
    <>
      <PageHeader
        title="System health"
        subtitle={`Is it working, and what should we improve next? Across the ${h.total} requests in this department so far; the server has been up ${
          up > 3600 ? `${(up / 3600).toFixed(1)} h` : `${Math.max(1, Math.round(up / 60))} min`
        }.`}
        actions={
          attention.length ? (
            <Badge tone={attention.some((c) => c.level === "bad") ? "bad" : "warn"}>
              <LiveDot tone={attention.some((c) => c.level === "bad") ? "bad" : "warn"} />
              {attention.length} thing{attention.length === 1 ? "" : "s"} to look at
            </Badge>
          ) : (
            <Badge tone="ok">
              <LiveDot /> all checks pass
            </Badge>
          )
        }
      />

      <NextSteps checks={h.checks} />

      <Card className="mt-6">
        <CardHeader icon={Route} title="Where requests ended up" subtitle={`All ${h.total} requests, by where each one is now`} />
        <Outcomes outcomes={h.outcomes} total={h.total} />
      </Card>

      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader icon={BookOpenText} title="How messages were read" subtitle="Who read each request, and how often the reading needed help" />
          <div className="space-y-5 p-5">
            <Shares
              parts={Object.entries(h.reading.readers).map(([k, v]) => ({ key: k, label: READER[k]?.[0] ?? k, hint: READER[k]?.[1], value: v }))}
              colour={(k) => (k === "model" ? "bg-ok" : k === "rules" ? "bg-warn" : k === "form" ? "bg-brand" : "bg-ink-3/40")}
            />
            <div className="grid grid-cols-2 gap-3">
              <Figure value={`${h.reading.asked}`} of={h.reading.received} label="needed a question back" hint="which week, which class: asked rather than guessed" tone={h.reading.asked / Math.max(1, h.reading.received) > 0.15 ? "warn" : undefined} />
              <Figure value={`${h.reading.unused}`} label="left part of the message unused" hint="shown to the sender and the approver" tone={h.reading.unused ? "warn" : undefined} />
              <Figure value={`${h.reading.split}`} label="split into separate requests" hint="one message asking for several things" />
              <Figure value={`${h.reading.injected}`} label="carried instructions to the system" hint="left out before reading; never obeyed" />
            </div>
          </div>
        </Card>

        <Card>
          <CardHeader icon={Handshake} title="Negotiations" subtitle="Clashes the solver could not fix alone, settled by asking colleagues" />
          <div className="grid grid-cols-2 gap-3 p-5 sm:grid-cols-3">
            <Figure value={`${h.negotiation.negotiated}`} label="negotiated" hint={`${h.negotiation.messages} messages sent`} />
            <Figure value={`${h.negotiation.agreed}`} of={h.negotiation.negotiated} label="agreed" hint={`${h.negotiation.first_round} on the first options`} tone="ok" />
            <Figure value={`${h.negotiation.escalated}`} label="escalated" hint="no agreement: a person with authority decides" tone={h.negotiation.escalated ? "warn" : undefined} />
            <Figure value={h.negotiation.mean_rounds == null ? "–" : h.negotiation.mean_rounds.toFixed(1)} label="rounds on average" hint="one round = one set of options" />
            <Figure value={`${h.negotiation.declined}`} label="options declined" hint="the next options look elsewhere" />
            <Figure value={`${h.negotiation.no_reply}`} label="asks with no reply" hint="one reminder, then escalated" tone={h.negotiation.no_reply ? "warn" : undefined} />
          </div>
        </Card>

        <Card>
          <CardHeader icon={Timer} title="Speed" subtitle="How long each stage takes: the typical request and the slowest 5%" />
          <Speed o={o} />
        </Card>

        <Card>
          <CardHeader icon={Brain} title="Models" subtitle="What reads messages now, and the answers cached so far" />
          <Models o={o} />
        </Card>

        <Card>
          <CardHeader icon={ShieldCheck} title="Safety guarantees" subtitle="Each one is enforced by code; these count whether it ever slipped" />
          <div className="grid grid-cols-2 gap-3 p-5 sm:grid-cols-3">
            <Figure value={`${o.safety.unapproved_publishes}`} label="published without approval" hint="must stay 0" tone={o.safety.unapproved_publishes ? "bad" : "ok"} />
            <Figure value={`${o.safety.leaks}`} label="private reasons leaked" hint="checked on every message; must stay 0" tone={o.safety.leaks ? "bad" : "ok"} />
            <Figure
              value={o.explanations.messages ? pct(o.explanations.faithfulness ?? 1) : "–"}
              label="explanations faithful"
              hint={`claims checked against the constraints, over ${o.explanations.messages} messages; ${o.explanations.rejected_claims} rejected`}
              tone={(o.explanations.faithfulness ?? 1) < 0.95 ? "warn" : "ok"}
            />
            <Figure value={`${o.safety.denied}`} label="denied by policy" hint="each reply cites the rule" />
            <Figure value={`${o.safety.refused}`} label="refused" hint="no authority over that class, or nothing but instructions" />
            <Figure value={`${h.reading.injected}`} label="instructions ignored" hint={'"ignore your rules" and the like'} />
          </div>
        </Card>

        <Card>
          <CardHeader icon={Cpu} title="Routing" subtitle="Which path each request took through the pipeline" />
          <Routing o={o} />
        </Card>
      </div>

      <Card className="mt-6">
        <CardHeader icon={Activity} title="Audit log" subtitle="Every step of every request is logged; counts by kind" />
        <EventCounts kinds={o.events_by_kind} />
      </Card>

      <Card className="mt-6">
        <CardHeader
          icon={Radio}
          title="Live event stream"
          subtitle="The newest steps as they happen"
          action={
            <Badge tone="ok">
              <LiveDot /> live
            </Badge>
          }
        />
        <div className="max-h-96 overflow-y-auto p-3">
          <ActivityFeed events={events.slice(-60)} />
        </div>
      </Card>
      {o.errors.length > 0 && (
        <Card className="mt-6 border-bad/40">
          <CardHeader icon={TriangleAlert} title="Recent errors" subtitle="The last ten, newest last" />
          <div className="space-y-2 p-5">
            {o.errors.map((e) => (
              <p key={e.n} className="font-mono text-[12px] text-bad">
                {e.case ? <Link to={`/requests/${e.case}`} className="underline">{e.case}</Link> : "system"} · {String(e.error)}
              </p>
            ))}
          </div>
        </Card>
      )}
    </>
  );
}

/* --------------------------------------------------------------------------------------------------------- */

function NextSteps({ checks }: { checks: HealthCheck[] }) {
  const [showOk, setShowOk] = useState(false);
  const attention = checks.filter((c) => c.level === "bad" || c.level === "warn");
  const info = checks.filter((c) => c.level === "info");
  const ok = checks.filter((c) => c.level === "ok");
  return (
    <Card>
      <CardHeader
        icon={CircleAlert}
        title="What to look at next"
        subtitle="Worst first. Each check says what it means and what to do about it"
      />
      <div className="space-y-3 p-5">
        {!attention.length && (
          <p className="flex items-center gap-2 rounded-md border border-ok/40 bg-ok/8 p-3 text-[13px] text-ok">
            <CircleCheck size={15} /> Nothing needs attention right now.
          </p>
        )}
        {[...attention, ...info].map((c, i) => (
          <CheckRow key={i} c={c} />
        ))}
        {ok.length > 0 && (
          <div>
            <button onClick={() => setShowOk((v) => !v)} className="flex items-center gap-1.5 text-[12.5px] text-ink-3 hover:text-ink">
              <ChevronDown size={14} className={clsx("transition-transform", showOk && "rotate-180")} />
              {ok.length} check{ok.length === 1 ? "" : "s"} passing
            </button>
            {showOk && (
              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                {ok.map((c, i) => (
                  <p key={i} className="flex items-start gap-2 rounded-md border border-line p-2.5 text-[12.5px]">
                    <CircleCheck size={14} className="mt-0.5 shrink-0 text-ok" />
                    <span>
                      <span className="font-medium">{c.title}.</span> <span className="text-ink-3">{c.detail}</span>
                    </span>
                  </p>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </Card>
  );
}

function CheckRow({ c }: { c: HealthCheck }) {
  const l = LEVEL[c.level];
  return (
    <div className={clsx("rounded-lg border p-4", l.border)}>
      <div className="flex items-start gap-3">
        <span className={clsx("grid size-8 shrink-0 place-items-center rounded-md", l.bg, l.text)}>
          <l.icon size={16} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="flex flex-wrap items-center gap-2 text-[11.5px]">
            <span className={clsx("font-medium", l.text)}>{l.label}</span>
            <span className="text-ink-3">· {c.area}</span>
          </p>
          <p className="mt-0.5 text-[14px] font-semibold">{c.title}</p>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{c.detail}</p>
          {c.next && (
            <p className="mt-2 flex items-start gap-1.5 text-[13px]">
              <ArrowRight size={14} className="mt-0.5 shrink-0 text-brand" />
              <span>
                <span className="font-medium text-brand">Next step: </span>
                {c.next}
              </span>
            </p>
          )}
          {c.cases.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {c.cases.map((id) => (
                <Link key={id} to={`/requests/${id}`} className="rounded bg-panel-2 px-1.5 py-0.5 font-mono text-[11.5px] text-brand hover:underline">
                  {id}
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Outcomes({ outcomes, total }: { outcomes: Record<string, number>; total: number }) {
  const rows = Object.entries(outcomes).filter(([, v]) => v > 0);
  if (!total) return <p className="p-5 text-[13px] text-ink-3">No requests yet.</p>;
  return (
    <div className="p-5">
      <div className="flex h-4 overflow-hidden rounded-full bg-panel-2">
        {rows.map(([k, v]) => (
          <div key={k} className={clsx("h-full", OUTCOME[k]?.[2] ?? "bg-ink-3")} style={{ width: `${(v / total) * 100}%` }} title={`${OUTCOME[k]?.[0] ?? k}: ${v}`} />
        ))}
      </div>
      <div className="mt-4 grid gap-x-6 gap-y-2.5 sm:grid-cols-2 lg:grid-cols-3">
        {rows.map(([k, v]) => (
          <div key={k} className="flex items-start gap-2.5 text-[13px]">
            <span className={clsx("mt-1 size-2.5 shrink-0 rounded-sm", OUTCOME[k]?.[2] ?? "bg-ink-3")} />
            <div className="min-w-0">
              <p>
                <span className="font-semibold tabular-nums">{v}</span> <span className="font-medium">{OUTCOME[k]?.[0] ?? k}</span>{" "}
                <span className="text-ink-3">({pct(v / total)})</span>
              </p>
              <p className="text-[11.5px] text-ink-3">{OUTCOME[k]?.[1]}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/** A labelled share bar with a legend. */
function Shares({ parts, colour }: { parts: { key: string; label: string; hint?: string; value: number }[]; colour: (k: string) => string }) {
  const total = parts.reduce((a, p) => a + p.value, 0);
  if (!total) return <p className="text-[13px] text-ink-3">No requests yet.</p>;
  return (
    <div>
      <p className="mb-2 text-[12px] font-medium text-ink-3">Read by</p>
      <div className="flex h-3 overflow-hidden rounded-full bg-panel-2">
        {parts.map((p) => (
          <div key={p.key} className={clsx("h-full", colour(p.key))} style={{ width: `${(p.value / total) * 100}%` }} />
        ))}
      </div>
      <div className="mt-2.5 flex flex-wrap gap-x-5 gap-y-1.5 text-[12.5px]">
        {parts.map((p) => (
          <span key={p.key} className="flex items-center gap-1.5" title={p.hint}>
            <span className={clsx("size-2.5 rounded-sm", colour(p.key))} />
            <span className="font-medium">{p.label}</span>
            <span className="text-ink-3">
              {p.value} ({pct(p.value / total)})
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}

function Figure({ value, of, label, hint, tone }: { value: ReactNode; of?: number; label: string; hint?: string; tone?: "ok" | "warn" | "bad" }) {
  return (
    <div className="rounded-md border border-line p-3">
      <p className={clsx("font-serif text-[22px] leading-none font-semibold tabular-nums", tone === "bad" ? "text-bad" : tone === "warn" ? "text-warn" : tone === "ok" ? "text-ok" : "")}>
        {value}
        {of != null && <span className="ml-1 font-sans text-[12px] font-normal text-ink-3">of {of}</span>}
      </p>
      <p className="mt-1.5 text-[12.5px] font-medium leading-tight">{label}</p>
      {hint && <p className="mt-0.5 text-[11.5px] leading-snug text-ink-3">{hint}</p>}
    </div>
  );
}

function Speed({ o }: { o: Data }) {
  const stages = Object.entries(o.stages);
  const max = Math.max(0.001, ...stages.map(([, s]) => s.p95 ?? 0));
  return (
    <div className="space-y-4 p-5">
      <div className="flex gap-4 text-[11.5px] text-ink-3">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded-sm bg-brand" /> typical (median)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-4 rounded-sm bg-brand/25" /> slowest 5% (95th percentile)
        </span>
      </div>
      {stages.map(([k, s]) => (
        <div key={k} className="space-y-1">
          <div className="flex items-baseline justify-between gap-3 text-[13px]">
            <span>
              <span className="font-medium">{STAGE[k]?.[0] ?? k}</span> <span className="text-[11.5px] text-ink-3">{STAGE[k]?.[1]}</span>
            </span>
            <span className="shrink-0 font-mono text-[12px] text-ink-2">
              {secs(s.p50)} · {secs(s.p95)}
            </span>
          </div>
          <div className="relative h-3 rounded bg-panel-2">
            <div className="absolute inset-y-0 left-0 rounded bg-brand/25" style={{ width: `${((s.p95 ?? 0) / max) * 100}%` }} />
            <div className="absolute inset-y-0 left-0 rounded bg-brand" style={{ width: `${((s.p50 ?? 0) / max) * 100}%` }} />
          </div>
          <p className="text-[11px] text-ink-3">
            {s.n} request{s.n === 1 ? "" : "s"} · slowest {secs(s.max)}
          </p>
        </div>
      ))}
      <div className="grid grid-cols-2 gap-3 border-t border-line pt-4">
        <Figure value={secs(o.solver.p50)} label="one CP-SAT solve, typical" hint={`slowest 5%: ${secs(o.solver.p95)} · ${o.solver.n} solves`} />
        <Figure
          value={o.routing.system_one_latency_ms_p50 == null ? "–" : `${o.routing.system_one_latency_ms_p50.toFixed(1)} ms`}
          label="System One routing, typical"
          hint="the classifier that picks the path"
        />
      </div>
    </div>
  );
}

function Models({ o }: { o: Data }) {
  const m = o.health.model;
  const inUse = o.models.parser ?? "";
  const local = o.llm.filter((x) => x.model.startsWith("local:"));
  const cloud = o.llm.filter((x) => !x.model.startsWith("local:"));
  const name = (x: string) => x.replace(/^local:/, "").replace(/^Qwen3\.5-4B-Base-Q8_0\+?/, "").replace(/\.gguf$/, "") || "base model";
  const current = (x: string) => inUse.includes(name(x));
  return (
    <div className="space-y-4 p-5 text-[13px]">
      {m && (
        <div className={clsx("flex items-start gap-3 rounded-md border p-3", m.up ? "border-ok/40 bg-ok/6" : "border-bad/40 bg-bad/6")}>
          <span className={clsx("grid size-8 shrink-0 place-items-center rounded-full", m.up ? "bg-ok/15 text-ok" : "bg-bad/15 text-bad")}>
            {m.up ? <Wifi size={16} /> : <WifiOff size={16} />}
          </span>
          <div className="min-w-0">
            <p className="font-medium">{m.up ? "The model is reachable" : "The model is unreachable"}</p>
            <p className="text-[12px] text-ink-3">
              {m.up ? "Requests are read by it." : `${m.error ?? ""} · ${m.held} request${m.held === 1 ? "" : "s"} waiting for it.`}{" "}
              {m.outages ? `Dropped out ${m.outages} time${m.outages === 1 ? "" : "s"} since start.` : "No outage since start."}
            </p>
          </div>
        </div>
      )}
      <p className="text-[12px] leading-relaxed text-ink-3">
        <span className="font-medium text-ink-2">In use: </span>
        {inUse || "offline rules"}
      </p>
      {cloud.length > 0 && (
        <div className="space-y-3">
          <p className="text-[12px] font-medium text-ink-3">Cloud models: calls today against the free daily limit</p>
          {cloud.map((x) => (
            <div key={x.model}>
              <div className="mb-1 flex justify-between gap-3">
                <span className="font-mono text-[12px]">{x.model}</span>
                <span className="text-ink-3">
                  <span className="font-medium text-ink">{x.used_today}</span> / {x.rpd ?? "?"} today · {x.cached_responses} cached
                </span>
              </div>
              <Meter value={x.used_today} max={x.rpd ?? 1} tone={x.rpd && x.used_today / x.rpd > 0.85 ? "bad" : x.rpd && x.used_today / x.rpd > 0.6 ? "warn" : "brand"} />
            </div>
          ))}
        </div>
      )}
      {local.length > 0 && (
        <div>
          <p className="mb-2 text-[12px] font-medium text-ink-3">
            Local adapters: answers cached{local.some((x) => x.counting) ? " (still counting)" : ""}, from use and evaluation runs; no quota
          </p>
          <div className="grid gap-x-4 gap-y-1 sm:grid-cols-2">
            {[...local]
              .sort((a, b) => Number(current(b.model)) - Number(current(a.model)) || b.cached_responses - a.cached_responses)
              .map((x) => (
                <div key={x.model} className="flex items-center justify-between gap-2 border-b border-line/60 py-1">
                  <span className={clsx("truncate font-mono text-[11.5px]", current(x.model) ? "text-ink" : "text-ink-3")} title={x.model}>
                    {name(x.model)}
                    {current(x.model) && <span className="ml-1.5 rounded bg-ok/12 px-1 font-sans text-[10px] text-ok">in use</span>}
                  </span>
                  <span className="shrink-0 font-mono text-[11.5px] text-ink-3">{x.cached_responses}</span>
                </div>
              ))}
          </div>
        </div>
      )}
      {!o.llm.length && <p className="text-ink-3">No model calls: running on the offline parser and policy rules.</p>}
      <p className="text-[11.5px] text-ink-3">Answers are cached by prompt, so asking the same thing again costs nothing.</p>
    </div>
  );
}

function Routing({ o }: { o: Data }) {
  const counts = Object.entries(o.routing.counts).sort((a, b) => b[1] - a[1]);
  const total = counts.reduce((a, [, v]) => a + v, 0);
  return (
    <div className="space-y-4 p-5 text-[13px]">
      <div className="space-y-1.5">
        {counts.map(([k, v]) => (
          <div key={k} className="grid grid-cols-[minmax(0,1fr)_90px_32px] items-center gap-3">
            <span className="truncate text-ink-2" title={k}>
              {routeLabel(k)}
            </span>
            <Meter value={v} max={total || 1} />
            <span className="text-right font-mono text-[12px]">{v}</span>
          </div>
        ))}
      </div>
      <div className="flex items-start gap-2 rounded-md bg-panel-2/60 p-3 text-[12px] leading-relaxed text-ink-2">
        <Database size={14} className="mt-0.5 shrink-0 text-ink-3" />
        <span>
          <span className="font-medium">Fast path: {pct(o.routing.fast_path_share)}.</span> System One (a small classifier, threshold τ = {o.routing.tau}) can
          send requests it is sure about to the rule reader. While the model is up it reads everything instead, because it is far more accurate, so 0% is
          expected; the fast path only takes over while the model is away.
        </span>
      </div>
    </div>
  );
}

function EventCounts({ kinds }: { kinds: Record<string, number> }) {
  const rows = Object.entries(kinds).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...rows.map(([, v]) => v));
  return (
    <div className="grid gap-x-8 gap-y-1.5 p-5 sm:grid-cols-2 lg:grid-cols-3">
      {rows.map(([k, v]) => (
        <div key={k} className="grid grid-cols-[minmax(0,1fr)_60px_32px] items-center gap-2 text-[12.5px]">
          <span className="truncate text-ink-2" title={k}>
            {EVENT_LABEL[k] ?? k.replace(/_/g, " ")}
          </span>
          <Meter value={v} max={max} />
          <span className="text-right font-mono text-[11.5px]">{v}</span>
        </div>
      ))}
    </div>
  );
}
