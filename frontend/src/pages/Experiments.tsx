import clsx from "clsx";
import { useState, type ReactNode } from "react";
import {
  BookOpen, CheckCircle2, ChevronDown, CircleDashed, Cpu, FlaskConical, MessagesSquare, Scale, Search, ShieldCheck,
  Swords, XCircle,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Card, EmptyState, PageHeader, Skeleton } from "../components/ui";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { ago, num, pct } from "../lib/meta";
import type { Experiment, ModelRow, NegotiationSummary, PairedTest, TableCol, TableRow } from "../lib/types";

// -- the page ------------------------------------------------------------------------------------------------------

/** Sections in reading order: the answers first, then the experiment behind each. */
const ORDER = ["findings", "negotiation-main", "head-to-head", "fairness", "ablations", "bias", "rag", "retrieval",
  "safety", "agent-adapters", "models"];

const NAV: Record<string, string> = {
  findings: "Key findings", "negotiation-main": "Main comparison", "head-to-head": "Pipeline vs agent",
  fairness: "Fairness", ablations: "Ablations", bias: "Same model", rag: "RAG", retrieval: "Retrieval",
  safety: "Safety", "agent-adapters": "Agent adapters", models: "Models",
};

export default function Experiments() {
  const { data } = useApi(() => api.experiments(), []);
  const sections = [...(data ?? [])].sort((a, b) => rank(a.id) - rank(b.id));
  return (
    <>
      <PageHeader
        title="Experiments"
        subtitle="What we tested, what we found, and the numbers behind it. Every number is read from the result files in runs/."
      />
      {!data ? (
        <Skeleton className="h-96" />
      ) : !data.length ? (
        <Card>
          <EmptyState icon={FlaskConical} title="No results yet" text="Run uv run python -m evaluation.negotiation --offline to produce the first report." />
        </Card>
      ) : (
        <>
          <nav className="sticky top-0 z-10 -mx-1 mb-5 flex flex-wrap gap-1.5 bg-bg/90 px-1 py-2 backdrop-blur">
            {sections.map((e) => (
              <a key={e.id} href={`#${e.id}`} className="rounded-full border border-line bg-panel px-3 py-1 text-[12.5px] text-ink-2 hover:border-brand hover:text-ink">
                {NAV[e.id] ?? e.title}
              </a>
            ))}
          </nav>
          <div className="space-y-8">
            {sections.map((e) => (
              <section key={e.id} id={e.id} className="scroll-mt-16">
                <Section e={e} />
              </section>
            ))}
          </div>
        </>
      )}
    </>
  );
}

const rank = (id: string) => (ORDER.indexOf(id) === -1 ? ORDER.length : ORDER.indexOf(id));

function Section({ e }: { e: Experiment }) {
  if (e.kind === "findings") return <Findings e={e} />;
  if (e.kind === "negotiation") return <Main e={e} />;
  if (e.kind === "models") return <Models e={e} />;
  if (e.kind === "table") return <TableSection e={e} />;
  return null;
}

// -- shared pieces -------------------------------------------------------------------------------------------------

const ICONS: Record<string, LucideIcon> = {
  scale: Scale, flask: FlaskConical, book: BookOpen, search: Search, shield: ShieldCheck, swords: Swords, cpu: Cpu,
};

/** Title, the question the experiment asks, and its answer in one sentence: what a reader needs before any table. */
function Lead({ icon: Icon, title, question, takeaway, setup, at }: { icon: LucideIcon; title: string; question?: string; takeaway?: string; setup?: string; at?: string }) {
  return (
    <div className="px-6 pb-4 pt-5">
      <div className="flex items-center gap-2 text-[12px] font-semibold uppercase tracking-wide text-ink-3">
        <Icon size={14} /> {title}
      </div>
      {question && <p className="mt-2 text-[13.5px] text-ink-2">{question}</p>}
      {takeaway && <p className="mt-2 border-l-[3px] border-brand pl-3 font-serif text-[18px] leading-snug text-ink">{takeaway}</p>}
      {(setup || at) && (
        <p className="mt-2 text-[11.5px] text-ink-3">
          {setup}
          {setup && at ? " · " : ""}
          {at ? `updated ${ago(at)}` : ""}
        </p>
      )}
    </div>
  );
}

/** One horizontal bar per row for the section's main measure; our system in the brand colour. */
function Bars({ rows, value, fmt, max }: { rows: { label: ReactNode; v: number | null; ours?: boolean; muted?: boolean }[]; value: string; fmt: (v: number) => string; max?: number }) {
  const top = max ?? Math.max(1e-9, ...rows.map((r) => r.v ?? 0));
  return (
    <div className="px-6 pb-5">
      <p className="mb-2 text-[11.5px] font-medium text-ink-3">{value}</p>
      <div className="space-y-1.5">
        {rows.map((r, i) => (
          <div key={i} className="grid grid-cols-[minmax(150px,240px)_1fr_64px] items-center gap-3">
            <div className={clsx("truncate text-[12.5px]", r.ours ? "font-semibold text-ink" : r.muted ? "text-ink-3" : "text-ink-2")}>{r.label}</div>
            <div className="h-3 overflow-hidden rounded-sm bg-panel-2">
              <div
                className={clsx("h-full rounded-sm", r.ours ? "bg-brand" : r.muted ? "bg-line" : "bg-ink-3/60")}
                style={{ width: `${Math.max(0, Math.min(1, (r.v ?? 0) / top)) * 100}%` }}
              />
            </div>
            <div className={clsx("text-right font-mono text-[12.5px] tabular-nums", r.ours ? "font-semibold text-ink" : "text-ink-2")}>
              {r.v === null ? "–" : fmt(r.v)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Toggle({ open, onClick, children }: { open: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button onClick={onClick} className="flex items-center gap-1.5 text-[12.5px] font-medium text-brand hover:underline">
      <ChevronDown size={14} className={clsx("transition-transform", open && "rotate-180")} />
      {children}
    </button>
  );
}

function Tag({ children }: { children: ReactNode }) {
  return <span className="ml-1.5 rounded bg-panel-2 px-1 py-px font-mono text-[10.5px] text-ink-3">{children}</span>;
}

function Notes({ notes }: { notes: string[] }) {
  if (!notes.length) return null;
  return (
    <ul className="space-y-1.5 border-t border-line px-6 py-4 text-[12.5px] leading-relaxed text-ink-2">
      {notes.map((n) => (
        <li key={n} className="flex gap-2">
          <span className="mt-[7px] size-1 shrink-0 rounded-full bg-ink-3" />
          <span>{n}</span>
        </li>
      ))}
    </ul>
  );
}

const fmtP = (p: number) => (p < 0.001 ? "< 0.001" : p.toFixed(3));

function fmtCell(c: TableCol, v: number | string | null | undefined): string {
  if (v === null || v === undefined) return "–";
  if (typeof v === "string") return v;
  switch (c.fmt) {
    case "pct":
      return pct(v, c.digits ?? 0);
    case "int":
      return String(Math.round(v));
    case "p":
      return fmtP(v);
    case "text":
      return String(v);
    default:
      return num(v, c.digits ?? (Math.abs(v) >= 10 ? 1 : 2));
  }
}

// -- key findings --------------------------------------------------------------------------------------------------

const VERDICT: Record<string, { label: string; icon: LucideIcon; cls: string }> = {
  supported: { label: "Supported", icon: CheckCircle2, cls: "bg-ok/10 text-ok" },
  partly: { label: "Partly", icon: CircleDashed, cls: "bg-warn/10 text-warn" },
  not: { label: "Not supported", icon: XCircle, cls: "bg-bad/10 text-bad" },
};

function Findings({ e }: { e: Extract<Experiment, { kind: "findings" }> }) {
  return (
    <Card>
      <Lead icon={CheckCircle2} title="Key findings" question="Each claim the project makes, whether the experiments support it, and the deciding numbers. Click a claim for the experiment." />
      <div className="divide-y divide-line border-t border-line">
        {e.items.map((f) => {
          const v = VERDICT[f.verdict] ?? VERDICT.partly;
          return (
            <a key={f.claim} href={`#${f.section}`} className="grid gap-x-4 gap-y-1 px-6 py-3.5 hover:bg-panel-2 sm:grid-cols-[120px_1fr]">
              <span className={clsx("inline-flex h-fit w-fit items-center gap-1.5 rounded-full px-2 py-0.5 text-[11.5px] font-semibold", v.cls)}>
                <v.icon size={13} /> {v.label}
              </span>
              <span>
                <span className="block text-[14px] font-medium text-ink">{f.claim}</span>
                <span className="mt-0.5 block text-[12.5px] leading-relaxed text-ink-2">{f.evidence}</span>
              </span>
            </a>
          );
        })}
      </div>
      {e.pending.length > 0 && (
        <p className="border-t border-line px-6 py-3 text-[12px] text-ink-3">
          <span className="font-medium text-ink-2">Still open: </span>
          {e.pending.join(" ")}
        </p>
      )}
    </Card>
  );
}

// -- a results table: the takeaway, a chart of the main measure, the key columns, the rest on demand --------------

function TableSection({ e }: { e: Extract<Experiment, { kind: "table" }> }) {
  const [all, setAll] = useState(false);
  const chart = e.columns.find((c) => c.key === e.chart);
  const cols = all ? e.columns : e.columns.filter((c) => c.key_col !== false);
  const hidden = e.columns.length - e.columns.filter((c) => c.key_col !== false).length;
  return (
    <Card>
      <Lead icon={ICONS[e.icon ?? ""] ?? FlaskConical} title={e.title} question={e.question} takeaway={e.takeaway} setup={e.subtitle} at={e.at} />
      {chart && (
        <Bars
          value={`${chart.label}${chart.better ? ` (${chart.better === "high" ? "higher" : "lower"} is better)` : ""}`}
          fmt={(v) => fmtCell(chart, v)}
          max={chart.fmt === "pct" ? 1 : undefined}
          rows={e.rows.map((r) => ({ label: <RowName r={r} />, v: typeof r.cells[chart.key] === "number" ? (r.cells[chart.key] as number) : null, ours: r.highlight }))}
        />
      )}
      <ResultTable cols={cols} rows={e.rows} />
      <div className="flex items-center gap-4 px-6 py-3">
        {hidden > 0 && (
          <Toggle open={all} onClick={() => setAll(!all)}>
            {all ? "Fewer measures" : `All measures (${hidden} more)`}
          </Toggle>
        )}
        <span className="text-[11.5px] text-ink-3">Hover a column for its definition; green is the best value.</span>
      </div>
      <Notes notes={e.notes} />
    </Card>
  );
}

function RowName({ r }: { r: TableRow }) {
  return (
    <>
      {r.label}
      {r.tag && <Tag>{r.tag}</Tag>}
    </>
  );
}

function ResultTable({ cols, rows }: { cols: TableCol[]; rows: TableRow[] }) {
  const best = (c: TableCol, v: unknown) => {
    if (!c.better || typeof v !== "number") return false;
    const vals = rows.map((r) => r.cells[c.key]).filter((x): x is number => typeof x === "number");
    return vals.length > 1 && new Set(vals).size > 1 && v === (c.better === "high" ? Math.max(...vals) : Math.min(...vals));
  };
  return (
    <div className="overflow-x-auto border-t border-line">
      <table className="w-full text-[13px]">
        <thead>
          <tr className="border-b border-line text-left text-[11.5px] text-ink-3">
            <th className="px-6 py-2.5 font-medium" />
            {cols.map((c) => (
              <th key={c.key} title={c.hint || undefined} className={clsx("max-w-[150px] px-3 py-2.5 text-right align-bottom font-medium leading-tight", c.hint && "cursor-help underline decoration-dotted underline-offset-2")}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.label}-${i}`} className={clsx("border-b border-line/60 last:border-0", r.highlight && "bg-brand/5")}>
              <td className="px-6 py-2.5">
                <span className={clsx(r.highlight && "font-semibold")}>
                  <RowName r={r} />
                </span>
                {r.sub && <span className="block text-[11.5px] text-ink-3">{r.sub}</span>}
              </td>
              {cols.map((c) => {
                const v = r.cells[c.key];
                return (
                  <td key={c.key} className={clsx("px-3 py-2.5 text-right font-mono tabular-nums", best(c, v) ? "font-semibold text-ok" : v === null ? "text-ink-3" : "text-ink-2")}>
                    {fmtCell(c, v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// -- the main comparison ---------------------------------------------------------------------------------------------

type ConfigInfo = { name: string; code: string; desc: string; main?: boolean };

// The configurations in plain words; the brief's codes (B1-B4, A1-A3) only as tags.
const CONFIGS: Record<string, ConfigInfo> = {
  "ours-llm": { name: "Our system", code: "", desc: "options only from the solver; LLM explains and reads replies", main: true },
  B4: { name: "LLM proposes, solver filters", code: "B4", desc: "the main baseline", main: true },
  B3: { name: "LLM proposes, unchecked", code: "B3", desc: "invented options offered as they are", main: true },
  B2: { name: "Solver decides, nobody asked", code: "B2", desc: "the cheapest fix is imposed", main: true },
  B1: { name: "LLM writes the timetable", code: "B1", desc: "no solver at all", main: true },
  oracle: { name: "Oracle", code: "", desc: "knows everyone's hidden flexibility: the best possible", main: true },
  ours: { name: "Our system, scripted", code: "", desc: "template messages and scripted replies; no LLM" },
  A1: { name: "Without tiers or ledger", code: "A1", desc: "see Ablations" },
  A2: { name: "Without grounded explanations", code: "A2", desc: "see Ablations" },
  A3: { name: "Without LLM reply reading", code: "A3", desc: "see Ablations" },
  "A1-offline": { name: "Without tiers or ledger, scripted", code: "A1", desc: "see Fairness" },
};

const info = (k: string): ConfigInfo => CONFIGS[k] ?? { name: k, code: "", desc: "" };

const METRICS: { key: keyof NegotiationSummary; label: string; hint: string; fmt: (v: number | null) => string; better: "high" | "low"; key_col?: boolean }[] = [
  { key: "agreement_rate", label: "Conflicts resolved", hint: "Resolvable conflicts that ended in agreement", fmt: (v) => pct(v), better: "high", key_col: true },
  { key: "correct_outcome", label: "Right outcome", hint: "Ended as the oracle did: no change needed, resolved, or escalated", fmt: (v) => pct(v), better: "high", key_col: true },
  { key: "objective_gap_abs_mean", label: "Gap to oracle", hint: "How far the final timetable's cost is from the best possible", fmt: (v) => num(v), better: "low", key_col: true },
  { key: "rounds_mean", label: "Rounds to agree", hint: "Mean negotiation rounds, over resolved conflicts only", fmt: (v) => num(v), better: "low", key_col: true },
  { key: "validity_of_timetables", label: "Valid timetables", hint: "Final timetables that break no hard constraint", fmt: (v) => pct(v), better: "high" },
  { key: "escalation_precision", label: "Escalations justified", hint: "Escalations that the oracle also escalated (precision)", fmt: (v) => num(v), better: "high" },
  { key: "escalation_recall", label: "Escalations caught", hint: "Oracle escalations this configuration also escalated (recall)", fmt: (v) => num(v), better: "high" },
  { key: "first_proposal_acceptance", label: "First offer taken", hint: "Negotiations where the first option offered was accepted", fmt: (v) => pct(v), better: "high" },
  { key: "explanation_faithfulness", label: "Faithful messages", hint: "Claims in messages that match the solver's facts", fmt: (v) => pct(v), better: "high" },
  { key: "private_leaks", label: "Privacy leaks", hint: "Private reasons that reached a colleague", fmt: (v) => String(v ?? 0), better: "low" },
  { key: "concession_gini", label: "Burden unevenness", hint: "Gini of concessions; 0 = shared evenly (see Fairness)", fmt: (v) => num(v), better: "low" },
];

const prettyModel = (m: string) => m.replace(/^local:/, "").replace(/\.gguf$/i, "").replace(/-/g, " ");

function Main({ e }: { e: Extract<Experiment, { kind: "negotiation" }> }) {
  const [allRows, setAllRows] = useState(false);
  const [allCols, setAllCols] = useState(false);
  const [tests, setTests] = useState(false);
  const entries = Object.entries(e.configs);
  const shown = entries.filter(([k]) => allRows || info(k).main);
  const ours = e.configs["ours-llm"] ?? e.configs.ours;
  const b4 = e.configs.B4;
  const takeaway =
    ours && b4
      ? `Restricting the negotiator to the solver's options resolves ${pct(ours.agreement_rate)} of conflicts against ${pct(b4.agreement_rate)} for an LLM whose own ideas the solver filters, and lands far closer to the best possible timetable (gap ${num(ours.objective_gap_abs_mean)} against ${num(b4.objective_gap_abs_mean)}). It does not take fewer rounds.`
      : undefined;
  const setup = [
    "60 scenarios with planted conflicts; our system and the LLM baselines × 3 seeds",
    ...(e.models ?? []).map((m) => `${prettyModel(m.model)}: ${m.configs.map((c) => info(c).name).join(", ")}`),
    ...(e.simulator ? [`simulated people: ${prettyModel(e.simulator)}`] : []),
  ].join(" · ");
  const cols = METRICS.filter((m) => allCols || m.key_col);
  return (
    <Card>
      <Lead
        icon={MessagesSquare}
        title="Main comparison"
        question="When requests conflict, does offering only solver-checked options beat letting an LLM invent them, even when the solver filters what the LLM invents?"
        takeaway={takeaway}
        setup={setup}
        at={e.at}
      />
      <Bars
        value="Conflicts resolved (higher is better)"
        fmt={(v) => pct(v)}
        max={1}
        rows={entries
          .filter(([k]) => info(k).main)
          .map(([k, s]) => ({
            label: (
              <>
                {info(k).name}
                {info(k).code && <Tag>{info(k).code}</Tag>}
              </>
            ),
            v: s.agreement_rate,
            ours: k === "ours-llm",
            muted: k === "oracle",
          }))}
      />
      <div className="overflow-x-auto border-t border-line">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-line text-left text-[11.5px] text-ink-3">
              <th className="px-6 py-2.5 font-medium" />
              <th className="px-3 py-2.5 text-right font-medium">Runs</th>
              {cols.map((m) => (
                <th key={m.key} title={m.hint} className="max-w-[130px] cursor-help px-3 py-2.5 text-right align-bottom font-medium leading-tight underline decoration-dotted underline-offset-2">
                  {m.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map(([k, s]) => {
              const c = info(k);
              return (
                <tr key={k} className={clsx("border-b border-line/60 last:border-0", k === "ours-llm" && "bg-brand/5")}>
                  <td className="px-6 py-2.5">
                    <span className={clsx(k === "ours-llm" && "font-semibold")}>{c.name}</span>
                    {c.code && <Tag>{c.code}</Tag>}
                    {c.desc && <span className="block text-[11.5px] text-ink-3">{c.desc}</span>}
                  </td>
                  <td className="px-3 py-2.5 text-right font-mono text-ink-3">{s.n}</td>
                  {cols.map((m) => {
                    const vals = shown.filter(([kk]) => kk !== "oracle").map(([, x]) => x[m.key] as number | null).filter((x): x is number => typeof x === "number");
                    const v = s[m.key] as number | null;
                    const best = k !== "oracle" && typeof v === "number" && vals.length > 1 && new Set(vals).size > 1 && v === (m.better === "high" ? Math.max(...vals) : Math.min(...vals));
                    return (
                      <td key={m.key} className={clsx("px-3 py-2.5 text-right font-mono tabular-nums", best ? "font-semibold text-ok" : "text-ink-2")}>
                        {m.fmt(v)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-6 py-3">
        <Toggle open={allCols} onClick={() => setAllCols(!allCols)}>
          {allCols ? "Fewer measures" : `All measures (${METRICS.length - METRICS.filter((m) => m.key_col).length} more)`}
        </Toggle>
        <Toggle open={allRows} onClick={() => setAllRows(!allRows)}>
          {allRows ? "Main configurations only" : "Also the ablation runs"}
        </Toggle>
        {e.paired && Object.keys(e.paired).length > 0 && (
          <Toggle open={tests} onClick={() => setTests(!tests)}>
            Is the difference real? (paired tests)
          </Toggle>
        )}
        <span className="text-[11.5px] text-ink-3">Green: best, not counting the oracle.</span>
      </div>
      {tests && e.paired && <Paired paired={e.paired} />}
    </Card>
  );
}

function Paired({ paired }: { paired: Record<string, PairedTest> }) {
  return (
    <div className="overflow-x-auto border-t border-line px-6 py-4">
      <p className="mb-2 text-[12px] text-ink-3">
        Same scenario and seed for both systems. "Only ours right" counts scenarios where our system ended as the oracle did and the other did not. p below
        0.05 (green) means the difference is unlikely to be chance.
      </p>
      <table className="w-full min-w-[560px] text-[13px]">
        <thead>
          <tr className="text-left text-[11.5px] text-ink-3">
            <th className="py-1.5 font-medium">Our system vs</th>
            <th className="py-1.5 text-right font-medium">Pairs</th>
            <th className="py-1.5 text-right font-medium">Only ours right</th>
            <th className="py-1.5 text-right font-medium">Only theirs right</th>
            <th className="py-1.5 text-right font-medium">p (outcome)</th>
            <th className="py-1.5 text-right font-medium">p (timetable cost)</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(paired).map(([k, v]) => {
            const b = k.split(" vs ")[1];
            const p = v.correct_outcome_mcnemar.p;
            return (
              <tr key={k} className="border-t border-line/60">
                <td className="py-1.5">
                  {info(b).name}
                  {info(b).code && <Tag>{info(b).code}</Tag>}
                </td>
                <td className="py-1.5 text-right font-mono">{v.pairs}</td>
                <td className="py-1.5 text-right font-mono">{v.correct_outcome_mcnemar.only_first}</td>
                <td className="py-1.5 text-right font-mono">{v.correct_outcome_mcnemar.only_second}</td>
                <td className={clsx("py-1.5 text-right font-mono", p < 0.05 && "font-semibold text-ok")}>{fmtP(p)}</td>
                <td className={clsx("py-1.5 text-right font-mono", v.objective_wilcoxon.n > 0 && v.objective_wilcoxon.p < 0.05 && "font-semibold text-ok")}>
                  {v.objective_wilcoxon.n ? fmtP(v.objective_wilcoxon.p) : "–"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// -- the models on the component benchmarks --------------------------------------------------------------------------

type Col = { key: keyof ModelRow; label: string; group: string; fmt: (v: number | string) => string; better?: "high" | "low"; key_col?: boolean; hint?: string };

const MODEL_COLS: Col[] = [
  { key: "parse_action", label: "Right action", group: "Reading requests", fmt: (v) => pct(v as number, 1), better: "high" },
  { key: "compile_exact", label: "Read exactly", group: "Reading requests", fmt: (v) => pct(v as number, 1), better: "high", key_col: true, hint: "The constraints compiled from the request match the gold ones exactly (151 requests)" },
  { key: "injections", label: "Injections refused", group: "Reading requests", fmt: (v) => String(v) },
  { key: "latency_p50", label: "Median s", group: "Reading requests", fmt: (v) => num(v as number, 2), better: "low", key_col: true },
  { key: "reply_tool", label: "Right tool call", group: "Reading replies", fmt: (v) => pct(v as number, 0), better: "high", key_col: true, hint: "100 stakeholder replies: accept, counter-offer, vague, refusal, injected" },
  { key: "reply_args", label: "Right details", group: "Reading replies", fmt: (v) => pct(v as number, 1), better: "high" },
  { key: "to_coordinator", label: "Sent to a person", group: "Reading replies", fmt: (v) => String(v), better: "low" },
  { key: "policy_allow_deny", label: "Right verdict", group: "Policy check", fmt: (v) => pct(v as number, 0), better: "high", key_col: true, hint: "109 requests: allowed or forbidden under the rules" },
  { key: "deny_recall", label: "Catches rule-breaking", group: "Policy check", fmt: (v) => pct(v as number, 0), better: "high" },
  { key: "makeup_recall", label: "Make-up duty", group: "Policy check", fmt: (v) => pct(v as number, 0), better: "high" },
  { key: "swap_pairs", label: "Right swap", group: "Swaps", fmt: (v) => pct(v as number, 0), better: "high" },
  { key: "swap_wrong", label: "Wrong swaps", group: "Swaps", fmt: (v) => String(v), better: "low" },
];

function Models({ e }: { e: Extract<Experiment, { kind: "models" }> }) {
  const [all, setAll] = useState(false);
  const cols = MODEL_COLS.filter((c) => all || c.key_col);
  const groups: { group: string; span: number }[] = [];
  for (const c of cols) {
    const last = groups[groups.length - 1];
    if (last && last.group === c.group) last.span++;
    else groups.push({ group: c.group, span: 1 });
  }
  const start = (i: number) => i === 0 || cols[i - 1].group !== cols[i].group;
  const ft = e.rows.find((r) => r.model.includes("4B fine-tuned · Q8"));
  const base = e.rows.find((r) => r.model.includes("4B base"));
  const takeaway =
    ft && base && typeof ft.compile_exact === "number" && typeof base.compile_exact === "number"
      ? `Fine-tuning is what makes the small local model usable: the 4B reads ${pct(ft.compile_exact)} of requests exactly with its LoRA adapter against ${pct(base.compile_exact)} prompted, and gets ${pct(ft.reply_tool as number)} of reply tool calls right against ${pct(base.reply_tool as number)}.`
      : undefined;
  return (
    <Card>
      <Lead icon={Cpu} title="Models" question="How does each model do on the parts the language model is used for: reading requests, reading replies and checking policy?" takeaway={takeaway} setup={e.subtitle} at={e.at} />
      <div className="overflow-x-auto border-t border-line">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="text-[11.5px] text-ink-3">
              <th />
              {groups.map((g) => (
                <th key={g.group} colSpan={g.span} className="border-l border-line px-2 pt-2.5 text-center font-medium">
                  {g.group}
                </th>
              ))}
            </tr>
            <tr className="border-b border-line text-left text-[11.5px] text-ink-3">
              <th className="px-6 py-2 font-medium" />
              {cols.map((c, i) => (
                <th key={c.key} title={c.hint} className={clsx("px-3 py-2 text-right font-medium", start(i) && "border-l border-line", c.hint && "cursor-help underline decoration-dotted underline-offset-2")}>
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {e.rows.map((r) => (
              <tr key={r.model} className={clsx("border-b border-line/60 last:border-0", r.model.includes("4B fine-tuned · Q8") && "bg-brand/5")}>
                <td className={clsx("px-6 py-2.5", r.model.includes("4B fine-tuned · Q8") && "font-semibold")}>{r.model}</td>
                {cols.map((c, i) => {
                  const v = r[c.key];
                  const vals = e.rows.map((x) => x[c.key]).filter((x): x is number => typeof x === "number");
                  const best = typeof v === "number" && !!c.better && vals.length > 1 && v === (c.better === "high" ? Math.max(...vals) : Math.min(...vals));
                  return (
                    <td key={c.key} className={clsx("px-3 py-2.5 text-right font-mono tabular-nums", start(i) && "border-l border-line", best ? "font-semibold text-ok" : v === null ? "text-ink-3" : "text-ink-2")}>
                      {v === null ? "–" : c.fmt(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center gap-4 px-6 py-3">
        <Toggle open={all} onClick={() => setAll(!all)}>
          {all ? "Fewer measures" : `All measures (${MODEL_COLS.length - MODEL_COLS.filter((c) => c.key_col).length} more)`}
        </Toggle>
        <span className="text-[11.5px] text-ink-3">"–": not run on that benchmark.</span>
      </div>
    </Card>
  );
}
