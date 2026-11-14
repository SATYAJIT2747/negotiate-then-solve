import clsx from "clsx";
import { Fragment } from "react";
import { BookOpen, Cpu, FlaskConical, MessagesSquare, Scale, Search, ShieldCheck, Swords } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge, Card, CardHeader, EmptyState, PageHeader, Skeleton } from "../components/ui";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { useTokens } from "../lib/theme";
import { ago, num, pct } from "../lib/meta";
import type { Experiment, ModelRow, NegotiationSummary, PairedTest, TableCol } from "../lib/types";

type ConfigInfo = { name: string; short: string; group: string; desc: string };

// Readable names for the configuration codes in runs/ (src/evaluation/negotiation.py)
const CONFIGS: Record<string, ConfigInfo> = {
  "ours-llm": { name: "Our system", short: "Ours", group: "Our system",
    desc: "Solver-checked options, priority tiers and burden ledger, grounded LLM explanations, LLM reads replies" },
  ours: { name: "Our system, scripted", short: "Ours (scripted)", group: "Our system",
    desc: "Same pipeline with template messages and scripted replies; no LLM calls" },
  B4: { name: "LLM proposes, solver filters", short: "LLM + filter", group: "Baselines",
    desc: "The LLM invents the options; the solver drops infeasible ones and the LLM tries again (main baseline)" },
  B3: { name: "LLM proposes, unchecked", short: "LLM unchecked", group: "Baselines",
    desc: "The LLM invents the options and they are offered without a solver check" },
  B2: { name: "Solver only, nobody asked", short: "Solver only", group: "Baselines",
    desc: "Same solver and priorities, but the cheapest fix is imposed without asking anyone" },
  B1: { name: "LLM only, no solver", short: "LLM only", group: "Baselines",
    desc: "The LLM writes the timetable directly" },
  A1: { name: "Without tiers or ledger", short: "No tiers", group: "Ablations: our system minus one part",
    desc: "Every stakeholder weighted the same, no burden ledger" },
  A2: { name: "Without grounded explanations", short: "Free explanations", group: "Ablations: our system minus one part",
    desc: "The LLM explains options in its own words instead of from solver facts" },
  A3: { name: "Without LLM reply reading", short: "Rule replies", group: "Ablations: our system minus one part",
    desc: "Keyword rules read stakeholder replies instead of the LLM" },
  "A1-offline": { name: "Without tiers or ledger, scripted", short: "No tiers (scripted)", group: "Ablations: our system minus one part",
    desc: "As above, with template messages and scripted replies; no LLM calls" },
  oracle: { name: "Oracle", short: "Oracle", group: "Reference",
    desc: "Solver that knows every stakeholder's hidden flexibility; the best achievable" },
};

const info = (k: string): ConfigInfo => CONFIGS[k] ?? { name: k, short: k, group: "Other", desc: "" };

const prettyModel = (m: string) => m.replace(/^local:/, "").replace(/\.gguf$/i, "").replace(/-/g, " ");

const METRICS: { key: keyof NegotiationSummary; label: string; hint: string; fmt: (v: number | null) => string; better: "high" | "low" }[] = [
  { key: "correct_outcome", label: "Right outcome", hint: "Ended as the oracle did: no change needed, resolved, or escalated", fmt: (v) => pct(v), better: "high" },
  { key: "validity_of_timetables", label: "Valid timetables", hint: "Final timetables that break no hard constraint", fmt: (v) => pct(v), better: "high" },
  { key: "agreement_rate", label: "Conflicts resolved", hint: "Resolvable conflicts that ended in agreement", fmt: (v) => pct(v), better: "high" },
  { key: "rounds_mean", label: "Rounds to agree", hint: "Mean negotiation rounds, over resolved conflicts only", fmt: (v) => num(v), better: "low" },
  { key: "objective_gap_abs_mean", label: "Gap to oracle", hint: "Mean distance of the final timetable's cost from the oracle's", fmt: (v) => num(v), better: "low" },
  { key: "escalation_precision", label: "Escalations justified", hint: "Escalations that the oracle also escalated (precision)", fmt: (v) => num(v), better: "high" },
  { key: "escalation_recall", label: "Escalations caught", hint: "Oracle escalations this configuration also escalated (recall)", fmt: (v) => num(v), better: "high" },
  { key: "concession_gini", label: "Burden unevenness", hint: "Gini of concessions across stakeholders; 0 = shared evenly", fmt: (v) => num(v), better: "low" },
  { key: "burden_max", label: "Most burdened", hint: "Largest total concession borne by one stakeholder", fmt: (v) => num(v), better: "low" },
  { key: "first_proposal_acceptance", label: "First offer taken", hint: "Negotiations where the first option offered was accepted", fmt: (v) => pct(v), better: "high" },
  { key: "explanation_faithfulness", label: "Faithful messages", hint: "Messages whose claims match the solver's facts", fmt: (v) => pct(v), better: "high" },
  { key: "private_leaks", label: "Privacy leaks", hint: "Messages that revealed another stakeholder's private information", fmt: (v) => String(v ?? 0), better: "low" },
];

function WrapTick({ x, y, payload }: { x?: number; y?: number; payload?: { value: string } }) {
  const words = (payload?.value ?? "").split(" ");
  const lines = words.length > 1 ? [words.slice(0, Math.ceil(words.length / 2)).join(" "), words.slice(Math.ceil(words.length / 2)).join(" ")] : words;
  return (
    <text x={x} y={(y ?? 0) + 4} textAnchor="middle" fill="var(--ink-3)" fontSize={11}>
      {lines.map((l, i) => (
        <tspan key={i} x={x} dy={i ? 13 : 9}>
          {l}
        </tspan>
      ))}
    </text>
  );
}

export default function Experiments() {
  const { data } = useApi(() => api.experiments(), []);
  return (
    <>
      <PageHeader
        title="Experiments"
        subtitle="Results from the evaluation scripts in runs/."
      />
      {!data ? (
        <Skeleton className="h-96" />
      ) : !data.length ? (
        <Card>
          <EmptyState icon={FlaskConical} title="No results yet" text="Run uv run python -m evaluation.negotiation --offline to produce the first report." />
        </Card>
      ) : (
        <div className="space-y-6">
          {data.map((e) => (
            <ExperimentCard key={e.id} e={e} />
          ))}
        </div>
      )}
    </>
  );
}

function ExperimentCard({ e }: { e: Experiment }) {
  const t = useTokens();
  if (e.kind === "negotiation") {
    const configs = Object.entries(e.configs);
    const chart = configs.map(([k, s]) => ({
      name: info(k).short,
      full: info(k).name,
      "Right outcome": +(100 * (s.correct_outcome ?? 0)).toFixed(1),
      "Conflicts resolved": +(100 * (s.agreement_rate ?? 0)).toFixed(1),
    }));
    const subtitle = [
      ...(e.models ?? []).map((m) => `${prettyModel(m.model)} runs ${m.configs.map((c) => info(c).short).join(", ")}`),
      ...(e.simulator ? [`simulated stakeholders: ${prettyModel(e.simulator)}`] : []),
      `updated ${ago(e.at)}`,
    ].join(" · ");
    return (
      <Card>
        <CardHeader icon={MessagesSquare} title={e.title} subtitle={subtitle} />
        <div className="h-72 p-5">
          <ResponsiveContainer>
            <BarChart data={chart} margin={{ bottom: 18 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--line)" vertical={false} />
              <XAxis dataKey="name" tick={<WrapTick />} interval={0} axisLine={false} tickLine={false} />
              <YAxis unit="%" tick={{ fill: "var(--ink-3)", fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip
                cursor={{ fill: "var(--panel-2)" }}
                labelFormatter={(_, p) => p?.[0]?.payload?.full ?? ""}
                formatter={(v) => `${v}%`}
                contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 6, fontSize: 12 }}
              />
              <Legend verticalAlign="top" wrapperStyle={{ fontSize: 12, paddingBottom: 8 }} />
              <Bar dataKey="Right outcome" fill={t.brand} radius={[3, 3, 0, 0]} />
              <Bar dataKey="Conflicts resolved" fill={t["ink-3"]} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="overflow-x-auto border-t border-line">
          <table className="w-full min-w-[1180px] text-[12.5px]">
            <thead>
              <tr className="border-b border-line text-left text-[11.5px] text-ink-3">
                <th className="px-5 py-2.5 font-medium">Configuration</th>
                <th className="px-2 py-2.5 text-right font-medium">Runs</th>
                {METRICS.map((m) => (
                  <th key={m.key} title={m.hint} className="cursor-help px-2 py-2.5 text-right font-medium">
                    {m.label}
                    <span className="block text-[10.5px] font-normal text-ink-3/80">{m.better === "high" ? "higher is better" : "lower is better"}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {configs.map(([k, s], i) => {
                const c = info(k);
                const newGroup = i === 0 || info(configs[i - 1][0]).group !== c.group;
                return (
                  <Fragment key={k}>
                    {newGroup && (
                      <tr className="border-b border-line/60">
                        <td colSpan={METRICS.length + 2} className="px-5 pb-1.5 pt-3 text-[11px] font-semibold uppercase tracking-wide text-ink-3">
                          {c.group}
                        </td>
                      </tr>
                    )}
                    <tr className={clsx("border-b border-line/60", c.group === "Our system" && "bg-panel-2")}>
                      <td className="max-w-[260px] px-5 py-2.5">
                        <span className="font-medium">{c.name}</span>
                        <span className="ml-1.5 font-mono text-[10.5px] text-ink-3">{k}</span>
                        {c.desc && <span className="mt-0.5 block text-[11.5px] leading-snug text-ink-3">{c.desc}</span>}
                      </td>
                      <td className="px-2 py-2.5 text-right font-mono text-ink-3">{s.n}</td>
                      {METRICS.map((m) => {
                        const vals = configs.map(([, x]) => x[m.key] as number | null).filter((x): x is number => typeof x === "number");
                        const v = s[m.key] as number | null;
                        const best = typeof v === "number" && vals.length > 1 && v === (m.better === "high" ? Math.max(...vals) : Math.min(...vals));
                        return (
                          <td key={m.key} className={clsx("px-2 py-2.5 text-right font-mono", best ? "font-semibold text-ok" : "text-ink-2")}>
                            {m.fmt(v)}
                          </td>
                        );
                      })}
                    </tr>
                  </Fragment>
                );
              })}
            </tbody>
          </table>
          <p className="px-5 py-3 text-[11.5px] text-ink-3">Hover a column name for its definition. Green marks the best value in each column; "–" means the measure does not apply.</p>
        </div>
        {e.paired && Object.keys(e.paired).length > 0 && <Paired paired={e.paired} />}
      </Card>
    );
  }
  if (e.kind === "models") return <Models e={e} />;
  if (e.kind === "table") return <TableCard e={e} />;
  const s = e.summary as Record<string, number | Record<string, string>>;
  const cats = s.by_category as Record<string, string>;
  return (
    <Card>
      <CardHeader icon={ShieldCheck} title={e.title} subtitle={`${s.n} cases · ${ago(e.at)}`} />
      <div className="grid gap-4 p-5 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="Pass rate" v={pct(s.pass_rate as number, 0)} />
        <Metric label="Unapproved publishes" v={String(s.unapproved_publishes)} />
        <Metric label="Authority violations" v={String(s.authority_violations)} />
        <Metric label="Injection successes" v={String(s.injection_successes)} />
      </div>
      <div className="flex flex-wrap gap-2 border-t border-line p-5">
        {Object.entries(cats ?? {}).map(([k, v]) => (
          <Badge key={k} tone={v.split("/")[0] === v.split("/")[1] ? "ok" : "warn"}>
            {k.replace(/_/g, " ")}: {v}
          </Badge>
        ))}
      </div>
    </Card>
  );
}

function Metric({ label, v }: { label: string; v: string }) {
  return (
    <div>
      <p className="text-[12px] text-ink-3">{label}</p>
      <p className="mt-1 font-serif text-2xl font-semibold tabular-nums">{v}</p>
    </div>
  );
}

function Paired({ paired }: { paired: Record<string, PairedTest> }) {
  return (
    <div className="overflow-x-auto border-t border-line px-5 py-4">
      <p className="text-[12.5px] font-medium text-ink-2">Is the difference real? Paired tests on the same scenario and seed</p>
      <p className="mb-2 text-[11.5px] text-ink-3">p below 0.05 (green) means the difference is unlikely to be chance.</p>
      <table className="w-full min-w-[560px] text-[12.5px]">
        <thead>
          <tr className="text-left text-[11.5px] text-ink-3">
            <th className="py-1.5 font-medium">Comparison</th>
            <th className="py-1.5 text-right font-medium">Pairs</th>
            <th className="py-1.5 text-right font-medium">Only the first right</th>
            <th className="py-1.5 text-right font-medium">Only the second right</th>
            <th title="McNemar test on right outcome" className="cursor-help py-1.5 text-right font-medium">Outcome p</th>
            <th title="Wilcoxon signed-rank test on timetable cost" className="cursor-help py-1.5 text-right font-medium">Timetable cost p</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(paired).map(([k, v]) => {
            const [a, b] = k.split(" vs ");
            const p = v.correct_outcome_mcnemar.p;
            return (
              <tr key={k} className="border-t border-line/60">
                <td className="py-1.5">
                  {info(a).name} <span className="text-ink-3">vs</span> {info(b).name}
                </td>
                <td className="py-1.5 text-right font-mono">{v.pairs}</td>
                <td className="py-1.5 text-right font-mono">{v.correct_outcome_mcnemar.only_first}</td>
                <td className="py-1.5 text-right font-mono">{v.correct_outcome_mcnemar.only_second}</td>
                <td className={clsx("py-1.5 text-right font-mono", p < 0.05 && "font-semibold text-ok")}>{fmtP(p)}</td>
                <td className={clsx("py-1.5 text-right font-mono", v.objective_wilcoxon.n > 0 && v.objective_wilcoxon.p < 0.05 && "font-semibold text-ok")}>
                  {v.objective_wilcoxon.n ? fmtP(v.objective_wilcoxon.p) : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const fmtP = (p: number) => (p < 0.001 ? p.toExponential(1) : p.toFixed(3));

type Col = { key: keyof ModelRow; label: string; group: string; fmt: (v: number | string) => string; better?: "high" | "low" };

const MODEL_COLS: Col[] = [
  { key: "reply_tool", label: "Action", group: "Replies", fmt: (v) => pct(v as number, 0), better: "high" },
  { key: "reply_args", label: "Details", group: "Replies", fmt: (v) => pct(v as number, 1), better: "high" },
  { key: "to_coordinator", label: "To coord.", group: "Replies", fmt: (v) => String(v), better: "low" },
  { key: "parse_action", label: "Action", group: "Parsing", fmt: (v) => pct(v as number, 1), better: "high" },
  { key: "compile_exact", label: "Compile exact", group: "Parsing", fmt: (v) => pct(v as number, 1), better: "high" },
  { key: "injections", label: "Injections", group: "Parsing", fmt: (v) => String(v) },
  { key: "latency_p50", label: "p50 s", group: "Parsing", fmt: (v) => num(v as number, 2), better: "low" },
  { key: "policy_allow_deny", label: "Allow/deny", group: "Policy", fmt: (v) => pct(v as number, 0), better: "high" },
  { key: "deny_recall", label: "Deny recall", group: "Policy", fmt: (v) => pct(v as number, 0), better: "high" },
  { key: "makeup_recall", label: "Make-up", group: "Policy", fmt: (v) => pct(v as number, 0), better: "high" },
  { key: "swap_pairs", label: "Right pair", group: "Swaps", fmt: (v) => pct(v as number, 0), better: "high" },
  { key: "swap_wrong", label: "Wrong commits", group: "Swaps", fmt: (v) => String(v), better: "low" },
];

const groupStart = (i: number) => i === 0 || MODEL_COLS[i - 1].group !== MODEL_COLS[i].group;

function Models({ e }: { e: Extract<Experiment, { kind: "models" }> }) {
  const groups: { group: string; span: number }[] = [];
  for (const c of MODEL_COLS) {
    const last = groups[groups.length - 1];
    if (last && last.group === c.group) last.span++;
    else groups.push({ group: c.group, span: 1 });
  }
  return (
    <Card>
      <CardHeader icon={Cpu} title={e.title} subtitle={`${e.subtitle ?? ""} · updated ${ago(e.at)}`} />
      <div className="overflow-x-auto">
        <table className="w-full min-w-[1100px] text-[12.5px]">
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
              <th className="px-5 py-2 font-medium">Model</th>
              {MODEL_COLS.map((c, i) => (
                <th key={c.key} className={clsx("px-2 py-2 text-right font-medium", groupStart(i) && "border-l border-line")}>
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {e.rows.map((r) => (
              <tr key={r.model} className={clsx("border-b border-line/60", r.model.includes("4B fine-tuned · Q8") && "bg-panel-2")}>
                <td className="px-5 py-2.5 font-medium">{r.model}</td>
                {MODEL_COLS.map((c, i) => {
                  const v = r[c.key];
                  const vals = e.rows.map((x) => x[c.key]).filter((x): x is number => typeof x === "number");
                  const best = typeof v === "number" && !!c.better && vals.length > 1 && v === (c.better === "high" ? Math.max(...vals) : Math.min(...vals));
                  return (
                    <td
                      key={c.key}
                      className={clsx("px-2 py-2.5 text-right font-mono", groupStart(i) && "border-l border-line", best ? "font-semibold text-ok" : v === null ? "text-ink-3" : "text-ink-2")}
                    >
                      {v === null ? "—" : c.fmt(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

const ICONS: Record<string, LucideIcon> = {
  scale: Scale, flask: FlaskConical, book: BookOpen, search: Search, shield: ShieldCheck, swords: Swords, cpu: Cpu,
};

function fmtCell(c: TableCol, v: number | string | null | undefined): string {
  if (v === null || v === undefined) return "–";
  if (typeof v === "string") return v;
  switch (c.fmt) {
    case "pct":
      return pct(v, c.digits ?? (v === 0 || v === 1 ? 0 : 1));
    case "int":
      return String(Math.round(v));
    case "p":
      return fmtP(v);
    case "text":
      return String(v);
    default:
      return num(v, c.digits ?? 2);
  }
}

/** A results table from src/web/experiments.py: one row per configuration, the best value per column in green,
 * an inline bar for the column the section is about, and what the numbers mean underneath. */
function TableCard({ e }: { e: Extract<Experiment, { kind: "table" }> }) {
  const Icon = ICONS[e.icon ?? ""] ?? FlaskConical;
  const best = (c: TableCol, v: unknown) => {
    if (!c.better || typeof v !== "number") return false;
    const vals = e.rows.map((r) => r.cells[c.key]).filter((x): x is number => typeof x === "number");
    return vals.length > 1 && new Set(vals).size > 1 && v === (c.better === "high" ? Math.max(...vals) : Math.min(...vals));
  };
  return (
    <Card>
      <CardHeader icon={Icon} title={e.title} subtitle={`${e.subtitle ?? ""}${e.subtitle ? " · " : ""}updated ${ago(e.at)}`} />
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-[12.5px]">
          <thead>
            <tr className="border-b border-line text-left text-[11.5px] text-ink-3">
              <th className="px-5 py-2.5 font-medium">Configuration</th>
              {e.columns.map((c) => (
                <th key={c.key} title={c.hint || undefined} className={clsx("px-3 py-2.5 text-right font-medium", c.hint && "cursor-help")}>
                  {c.label}
                  {c.better && <span className="block text-[10.5px] font-normal text-ink-3/80">{c.better === "high" ? "higher is better" : "lower is better"}</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {e.rows.map((r, i) => (
              <tr key={`${r.label}-${i}`} className={clsx("border-b border-line/60", r.highlight && "bg-panel-2")}>
                <td className="max-w-[300px] px-5 py-2.5">
                  <span className="font-medium">{r.label}</span>
                  {r.sub && <span className="mt-0.5 block text-[11.5px] leading-snug text-ink-3">{r.sub}</span>}
                </td>
                {e.columns.map((c) => {
                  const v = r.cells[c.key];
                  const good = best(c, v);
                  return (
                    <td key={c.key} className={clsx("px-3 py-2.5 text-right font-mono", good ? "font-semibold text-ok" : v === null ? "text-ink-3" : "text-ink-2")}>
                      {c.bar && typeof v === "number" ? (
                        <span className="inline-flex items-center justify-end gap-2">
                          <span className="hidden h-1.5 w-20 overflow-hidden rounded-full bg-line sm:inline-block">
                            <span className={clsx("block h-full rounded-full", good ? "bg-ok" : "bg-brand")} style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%` }} />
                          </span>
                          {fmtCell(c, v)}
                        </span>
                      ) : (
                        fmtCell(c, v)
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {e.notes.length > 0 && (
        <ul className="list-disc space-y-1 border-t border-line px-5 py-3 pl-9 text-[12px] leading-relaxed text-ink-2">
          {e.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      )}
    </Card>
  );
}
