import clsx from "clsx";
import { useState, type ReactNode } from "react";
import {
  BookOpen, CheckCircle2, ChevronDown, CircleDashed, Cpu, FlaskConical, MessagesSquare, Scale, Search, ShieldCheck,
  Swords, XCircle, Bot, Eye, Lock, Quote, UserRound,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Cell, ErrorBar, LabelList, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip,
  XAxis, YAxis,
} from "recharts";
import { Card, EmptyState, PageHeader, Skeleton } from "../components/ui";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { ago, num, pct } from "../lib/meta";
import { useTokens } from "../lib/theme";
import type { ChapterChart, Experiment, TableCol, TableRow, WorkedExample } from "../lib/types";

type Chapter = Extract<Experiment, { kind: "chapter" }>;
type Intro = Extract<Experiment, { kind: "intro" }>;
type Conclusions = Extract<Experiment, { kind: "conclusions" }>;

const ICONS: Record<string, LucideIcon> = {
  messages: MessagesSquare, scale: Scale, flask: FlaskConical, book: BookOpen, search: Search, shield: ShieldCheck,
  swords: Swords, cpu: Cpu,
};

// -- the page: introduction, chapters, conclusions ---------------------------------------------------------------------

export default function Experiments() {
  const { data } = useApi(() => api.experiments(), []);
  const intro = data?.find((e): e is Intro => e.kind === "intro");
  const chapters = (data ?? []).filter((e): e is Chapter => e.kind === "chapter");
  const end = data?.find((e): e is Conclusions => e.kind === "conclusions");
  return (
    <>
      <PageHeader title="Experiments" subtitle="What we set out to test, how we tested it, and what we found. Every number is read from the result files in runs/." />
      {!data ? (
        <Skeleton className="h-96" />
      ) : !chapters.length ? (
        <Card>
          <EmptyState icon={FlaskConical} title="No results yet" text="Run uv run python -m evaluation.negotiation --offline to produce the first report." />
        </Card>
      ) : (
        <div className="mx-auto max-w-[1080px]">
          {intro && <IntroCard e={intro} chapters={chapters} end={end} />}
          <div className="mt-10 space-y-12">
            {chapters.map((c, i) => (
              <ChapterView key={c.id} c={c} n={i + 1} />
            ))}
          </div>
          {end && <ConclusionsCard e={end} chapters={chapters} />}
        </div>
      )}
    </>
  );
}

function IntroCard({ e, chapters, end }: { e: Intro; chapters: Chapter[]; end?: Conclusions }) {
  const verdict = (id: string) => end?.items.find((f) => f.section === id)?.verdict;
  return (
    <Card>
      <div className="grid gap-8 p-7 lg:grid-cols-[1.15fr_1fr]">
        <div>
          <p className="text-[12px] font-semibold uppercase tracking-wide text-ink-3">The problem</p>
          <p className="mt-2 text-[14.5px] leading-relaxed text-ink-2">{e.problem}</p>
          <p className="mt-5 text-[12px] font-semibold uppercase tracking-wide text-ink-3">The question</p>
          <p className="mt-2 border-l-[3px] border-brand pl-3 font-serif text-[19px] leading-snug text-ink">{e.question}</p>
        </div>
        <div>
          <p className="text-[12px] font-semibold uppercase tracking-wide text-ink-3">How the system works</p>
          <ol className="mt-2 space-y-2">
            {e.approach.map((s, i) => (
              <li key={s} className="flex gap-3 text-[13.5px] leading-snug text-ink-2">
                <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-brand/10 text-[11px] font-semibold text-brand">{i + 1}</span>
                {s}
              </li>
            ))}
          </ol>
        </div>
      </div>
      <div className="border-t border-line px-7 py-5">
        <p className="mb-3 text-[12px] font-semibold uppercase tracking-wide text-ink-3">The experiments</p>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {chapters.map((c, i) => {
            const Icon = ICONS[c.icon] ?? FlaskConical;
            const v = verdict(c.id);
            return (
              <a key={c.id} href={`#${c.id}`} className="group flex items-start gap-3 rounded-lg border border-line px-3 py-2.5 hover:border-brand hover:bg-panel-2">
                <span className="mt-0.5 font-mono text-[11px] text-ink-3">{String(i + 1).padStart(2, "0")}</span>
                <Icon size={15} className="mt-0.5 shrink-0 text-ink-3 group-hover:text-brand" />
                <span className="min-w-0 flex-1">
                  <span className="block text-[13.5px] font-medium text-ink">{c.title}</span>
                </span>
                {v && <VerdictDot v={v} />}
              </a>
            );
          })}
          {end && (
            <a href="#conclusions" className="flex items-center gap-3 rounded-lg border border-dashed border-line px-3 py-2.5 text-[13.5px] font-medium text-ink-2 hover:border-brand">
              <CheckCircle2 size={15} className="text-ink-3" /> Conclusions
            </a>
          )}
        </div>
      </div>
    </Card>
  );
}

// -- one chapter: the question, what we did, what we found, what it means, the details ----------------------------------

function ChapterView({ c, n }: { c: Chapter; n: number }) {
  const [table, setTable] = useState(false);
  const Icon = ICONS[c.icon] ?? FlaskConical;
  return (
    <section id={c.id} className="scroll-mt-6">
      <div className="mb-3 flex items-baseline gap-3">
        <span className="font-mono text-[13px] text-brand">{String(n).padStart(2, "0")}</span>
        <h2 className="flex items-center gap-2 font-serif text-[26px] font-semibold leading-tight text-ink">
          <Icon size={20} className="text-ink-3" /> {c.title}
        </h2>
      </div>
      <p className="mb-4 max-w-[860px] text-[14.5px] leading-relaxed text-ink-2">{c.question}</p>
      <Card>
        <div className="grid gap-6 border-b border-line p-6 lg:grid-cols-[1.2fr_1fr]">
          <div>
            <Kicker>What we did</Kicker>
            <ol className="mt-2 space-y-2">
              {c.method.map((s, i) => (
                <li key={s} className="flex gap-3 text-[13.5px] leading-snug text-ink-2">
                  <span className="mt-px font-mono text-[11.5px] text-ink-3">{i + 1}.</span>
                  {s}
                </li>
              ))}
            </ol>
            {c.setup && <p className="mt-3 text-[11.5px] text-ink-3">{c.setup}</p>}
          </div>
          <div>
            <Kicker>Compared</Kicker>
            <ul className="mt-2 space-y-1.5">
              {c.compared.map((w) => (
                <li key={w.name} className="flex gap-2 text-[13px] leading-snug">
                  <span className={clsx("mt-1.5 size-2 shrink-0 rounded-full", w.ours ? "bg-brand" : "bg-ink-3/50")} />
                  <span>
                    <span className={clsx(w.ours ? "font-semibold text-ink" : "font-medium text-ink-2")}>{w.name}</span>
                    {w.tag && <Tag>{w.tag}</Tag>}
                    {w.desc && <span className="text-ink-3"> · {w.desc}</span>}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="p-6">
          <Kicker>What we found</Kicker>
          <p className="mt-2 max-w-[880px] font-serif text-[19px] leading-snug text-ink">{c.takeaway}</p>
          <div className={clsx("mt-6 grid gap-6", c.charts.length > 1 && "lg:grid-cols-2")}>
            {c.charts.map((ch, i) => (
              <ChartView key={ch.title} ch={ch} wide={c.charts.length % 2 === 1 && i === c.charts.length - 1 && c.charts.length > 1} />
            ))}
          </div>
        </div>
        {(c.examples ?? []).length > 0 && (
          <div className="border-t border-line p-6">
            <Kicker>In practice</Kicker>
            <div className="mt-3 space-y-6">
              {c.examples!.map((x) => (
                <ExampleView key={x.title + x.situation} x={x} />
              ))}
            </div>
          </div>
        )}
        {c.notes.filter(Boolean).length > 0 && (
          <div className="border-t border-line px-6 py-5">
            <Kicker>What it means</Kicker>
            <ul className="mt-2 space-y-2">
              {c.notes.filter(Boolean).map((t) => (
                <li key={t} className="flex gap-2.5 text-[13.5px] leading-relaxed text-ink-2">
                  <span className="mt-[9px] size-1 shrink-0 rounded-full bg-ink-3" />
                  {t}
                </li>
              ))}
            </ul>
          </div>
        )}
        {c.table && (
          <div className="border-t border-line">
            <button onClick={() => setTable(!table)} className="flex w-full items-center gap-1.5 px-6 py-3 text-left text-[12.5px] font-medium text-brand hover:bg-panel-2">
              <ChevronDown size={14} className={clsx("transition-transform", table && "rotate-180")} />
              {table ? "Hide the numbers" : "All the numbers"}
            </button>
            {table && <ResultTable cols={c.table.columns} rows={c.table.rows} />}
          </div>
        )}
        {c.at && <p className="border-t border-line px-6 py-2 text-right text-[11px] text-ink-3">results updated {ago(c.at)}</p>}
      </Card>
    </section>
  );
}

function Kicker({ children }: { children: ReactNode }) {
  return <p className="text-[11.5px] font-semibold uppercase tracking-wide text-ink-3">{children}</p>;
}

function Tag({ children }: { children: ReactNode }) {
  return <span className="ml-1.5 rounded bg-panel-2 px-1 py-px font-mono text-[10.5px] text-ink-3">{children}</span>;
}

// -- charts -----------------------------------------------------------------------------------------------------------

const fmtValue = (fmt: ChapterChart["fmt"], v: number | null | undefined) =>
  v === null || v === undefined ? "–" : fmt === "pct" ? pct(v) : fmt === "int" ? String(Math.round(v)) : num(v, Math.abs(v) >= 10 ? 1 : 2);

function usePalette() {
  const t = useTokens();
  // the brand colour means our system and nothing else; other systems and measures get quieter, distinct colours
  return { ours: t.brand, others: [t["ink-3"], t.info, t.warn, t.bad, t.ok], grid: t.line, ink: t["ink-3"], ink2: t["ink-2"], text: t.ink, muted: t.line };
}

function ChartView({ ch, wide }: { ch: ChapterChart; wide?: boolean }) {
  return (
    <figure className={clsx("rounded-lg border border-line p-4", wide && "lg:col-span-2")}>
      <figcaption className="mb-3">
        <span className="block text-[13.5px] font-semibold text-ink">{ch.title}</span>
        {ch.caption && <span className="mt-0.5 block text-[12px] leading-snug text-ink-3">{ch.caption}</span>}
      </figcaption>
      {ch.type === "line" ? <LineView ch={ch} /> : <BarsView ch={ch} />}
    </figure>
  );
}

function colours(ch: ChapterChart, p: ReturnType<typeof usePalette>) {
  let i = 0;
  return ch.series.map((s) => (s.ours ? p.ours : p.others[i++ % p.others.length]));
}

/** Horizontal bars: one row per category, a bar per series, the value written at every bar's end (also 0, and
 * "n/a" when there is none). With one series our system's bar is the only coloured one; with several, the series
 * are measures in neutral colours (or systems, ours in the brand colour) and our system's row is labelled in bold. */
function BarsView({ ch }: { ch: ChapterChart }) {
  const p = usePalette();
  const single = ch.series.length === 1;
  const seriesOurs = ch.series.some((s) => s.ours);
  const cols = seriesOurs ? colours(ch, p) : ch.series.map((_, j) => (single ? p.others[0] : [p.ink2, p.others[1], p.others[2], p.others[3]][j % 4]));
  const isOurs = (name: string) => ch.ours.includes(name);
  const data = ch.categories.map((c, i) => ({
    name: c,
    ...Object.fromEntries(ch.series.map((s, j) => [`s${j}`, s.data[i] ?? 0])),
    ...Object.fromEntries(ch.series.map((s, j) => [`v${j}`, s.data[i]])),
    ...Object.fromEntries(ch.series.map((s, j) => [`e${j}`, s.error?.[i] && s.data[i] !== null ? [s.data[i]! - s.error[i][0], s.error[i][1] - s.data[i]!] : undefined])),
  }));
  const height = ch.categories.length * (single ? 34 : 18 * ch.series.length + 14) + (single ? 24 : 52) + (ch.reference ? 16 : 0);
  const max = ch.max ?? (ch.fmt === "pct" ? 1 : undefined);
  const Tick = ({ x, y, payload }: { x?: number; y?: number; payload?: { value: string } }) => {
    const v = payload?.value ?? "";
    return (
      <text x={x} y={y} dy={4} textAnchor="end" fontSize={12} fill={isOurs(v) ? p.text : p.ink} fontWeight={isOurs(v) ? 700 : 400}>
        {v.length > 26 ? `${v.slice(0, 25)}…` : v}
      </text>
    );
  };
  return (
    <div style={{ height }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: ch.reference ? 16 : 0, right: 52, bottom: 0, left: 0 }} barCategoryGap={single ? 8 : 10} barGap={2}>
          <CartesianGrid horizontal={false} stroke={p.grid} strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, max ?? "auto"]} tickFormatter={(v) => fmtValue(ch.fmt, v)} tick={{ fill: p.ink, fontSize: 11 }} axisLine={false} tickLine={false} />
          <YAxis type="category" dataKey="name" width={180} tick={<Tick />} axisLine={false} tickLine={false} interval={0} />
          <Tooltip cursor={{ fill: "var(--panel-2)" }} formatter={(_v, name, item) => {
            const j = ch.series.findIndex((s) => s.name === name);
            return [fmtValue(ch.fmt, (item?.payload as Record<string, number | null>)?.[`v${j}`]), name];
          }} contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 6, fontSize: 12 }} />
          {!single && <Legend verticalAlign="top" align="left" iconSize={9} wrapperStyle={{ fontSize: 12, paddingBottom: 6 }} />}
          {ch.reference && (
            <ReferenceLine x={ch.reference.value} stroke={p.ink} strokeDasharray="5 4" ifOverflow="extendDomain"
              label={{ value: `${ch.reference.label}: ${fmtValue(ch.fmt, ch.reference.value)} (best possible)`, position: "top", fill: p.ink, fontSize: 11 }} />
          )}
          {ch.series.map((s, j) => (
            <Bar key={s.name} dataKey={`s${j}`} name={s.name} fill={cols[j]} radius={[0, 3, 3, 0]} isAnimationActive={false} minPointSize={0}>
              {single && data.map((d) => <Cell key={d.name} fill={isOurs(d.name) ? p.ours : p.others[0]} />)}
              <LabelList dataKey={`v${j}`} content={(props) => {
                const { x, y, width, height: h, value } = props as { x: number; y: number; width: number; height: number; value: number | null };
                return (
                  <text x={Number(x) + Number(width) + 5} y={Number(y) + Number(h) / 2} dy={4} fontSize={11} fill={value === null || value === undefined ? p.ink : "var(--ink-2)"}>
                    {value === null || value === undefined ? "n/a" : fmtValue(ch.fmt, value)}
                  </text>
                );
              }} />
              {s.error && <ErrorBar dataKey={`e${j}`} width={4} stroke={p.ink} direction="x" />}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function LineView({ ch }: { ch: ChapterChart }) {
  const p = usePalette();
  const cols = colours(ch, p);
  const data = ch.categories.map((c, i) => ({ name: c, ...Object.fromEntries(ch.series.map((s, j) => [`s${j}`, s.data[i]])) }));
  return (
    <div className="h-72">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 4, right: 16, bottom: 18, left: 4 }}>
          <CartesianGrid stroke={p.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="name" tick={{ fill: p.ink, fontSize: 11 }} axisLine={false} tickLine={false}
            label={ch.x_label ? { value: ch.x_label, position: "insideBottom", offset: -10, fill: p.ink, fontSize: 11 } : undefined} />
          <YAxis domain={[0, ch.max ?? "auto"]} tickFormatter={(v) => fmtValue(ch.fmt, v)} tick={{ fill: p.ink, fontSize: 11 }} axisLine={false} tickLine={false} width={44} />
          <Tooltip formatter={(v) => fmtValue(ch.fmt, Number(v))} labelFormatter={(l) => (ch.x_label ? `${ch.x_label}: ${l}` : String(l))} contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 6, fontSize: 12 }} />
          <Legend verticalAlign="top" align="left" iconSize={9} wrapperStyle={{ fontSize: 12, paddingBottom: 6 }} />
          {ch.series.map((s, j) => (
            <Line key={s.name} dataKey={`s${j}`} name={s.name} stroke={cols[j]} strokeWidth={s.ours ? 3 : 1.75} dot={ch.categories.length <= 6 ? { r: 3 } : false} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// -- the full table ----------------------------------------------------------------------------------------------------

function fmtCell(c: TableCol, v: number | string | null | undefined): string {
  if (v === null || v === undefined) return "–";
  if (typeof v === "string") return v;
  if (c.fmt === "pct") return pct(v, c.digits ?? 0);
  if (c.fmt === "int") return String(Math.round(v));
  if (c.fmt === "p") return v < 0.001 ? "< 0.001" : v.toFixed(3);
  if (c.fmt === "text") return String(v);
  return num(v, c.digits ?? (Math.abs(v) >= 10 ? 1 : 2));
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
                <span className={clsx(r.highlight && "font-semibold")}>{r.label}</span>
                {r.tag && <Tag>{r.tag}</Tag>}
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
      <p className="px-6 py-2 text-[11.5px] text-ink-3">Green: the best value in the column. Hover an underlined column for its definition.</p>
    </div>
  );
}

// -- conclusions -------------------------------------------------------------------------------------------------------

const VERDICT: Record<string, { label: string; icon: LucideIcon; cls: string }> = {
  supported: { label: "Supported", icon: CheckCircle2, cls: "bg-ok/10 text-ok" },
  partly: { label: "Partly", icon: CircleDashed, cls: "bg-warn/10 text-warn" },
  not: { label: "Not supported", icon: XCircle, cls: "bg-bad/10 text-bad" },
};

function VerdictDot({ v }: { v: string }) {
  const x = VERDICT[v] ?? VERDICT.partly;
  return <x.icon size={15} className={clsx("mt-0.5 shrink-0", x.cls.split(" ")[1])} aria-label={x.label} />;
}

function ConclusionsCard({ e, chapters }: { e: Conclusions; chapters: Chapter[] }) {
  const num_ = (id: string) => chapters.findIndex((c) => c.id === id) + 1;
  return (
    <section id="conclusions" className="mt-12 scroll-mt-6">
      <h2 className="mb-3 flex items-center gap-2 font-serif text-[26px] font-semibold text-ink">
        <CheckCircle2 size={20} className="text-ink-3" /> Conclusions
      </h2>
      <Card>
        <div className="divide-y divide-line">
          {e.items.map((f) => {
            const v = VERDICT[f.verdict] ?? VERDICT.partly;
            const n = num_(f.section);
            return (
              <a key={f.claim} href={`#${f.section}`} className="grid gap-x-4 gap-y-1 px-6 py-3.5 hover:bg-panel-2 sm:grid-cols-[120px_1fr_auto]">
                <span className={clsx("inline-flex h-fit w-fit items-center gap-1.5 rounded-full px-2 py-0.5 text-[11.5px] font-semibold", v.cls)}>
                  <v.icon size={13} /> {v.label}
                </span>
                <span>
                  <span className="block text-[14px] font-medium text-ink">{f.claim}</span>
                  <span className="mt-0.5 block text-[12.5px] leading-relaxed text-ink-2">{f.evidence}</span>
                </span>
                {n > 0 && <span className="font-mono text-[11.5px] text-ink-3">ch. {String(n).padStart(2, "0")}</span>}
              </a>
            );
          })}
        </div>
        {e.pending.length > 0 && (
          <p className="border-t border-line px-6 py-3 text-[12.5px] text-ink-3">
            <span className="font-medium text-ink-2">Still open: </span>
            {e.pending.join(" ")}
          </p>
        )}
      </Card>
    </section>
  );
}

// -- worked examples ---------------------------------------------------------------------------------------------------

const TONE: Record<string, string> = {
  ok: "bg-ok/10 text-ok", bad: "bg-bad/10 text-bad", warn: "bg-warn/10 text-warn", muted: "bg-panel-2 text-ink-3",
};

/** One real case: the situation, then each system's conversation side by side as a chat thread, and the lesson. */
function ExampleView({ x }: { x: WorkedExample }) {
  const n = x.cases.length;
  const brief = x.cases.some((c) => c.summary);
  const [full, setFull] = useState(false);
  return (
    <div className="rounded-xl border border-line bg-panel-2/40 p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-[15px] font-semibold text-ink">{x.title}</p>
        {brief && (
          <button onClick={() => setFull((v) => !v)} className="text-[12px] font-medium text-brand hover:underline">
            {full ? "Back to the summary" : "Show the full conversations"}
          </button>
        )}
      </div>
      {x.requests?.length ? (
        <div className="mt-2 rounded-lg bg-panel px-3 py-2.5">
          <p className="text-[13px] text-ink-2">{x.situation}</p>
          <div className="mt-2 space-y-1.5">
            {x.requests.map((r) => (
              <RequestLine key={r.who + r.text} r={r} />
            ))}
          </div>
          {x.key && (
            <p className="mt-2.5 flex items-center gap-1.5 text-[12.5px] font-medium text-ink" title={x.hidden}>
              <Eye className="size-3.5 shrink-0 text-brand" /> {x.key}
            </p>
          )}
        </div>
      ) : (
        <div className="mt-2 flex gap-2.5 rounded-lg border-l-[3px] border-brand/60 bg-panel px-3 py-2.5">
          <Quote className="mt-0.5 size-4 shrink-0 text-brand/70" />
          <p className="text-[13px] leading-relaxed text-ink-2">{x.situation}</p>
        </div>
      )}
      <div className={clsx("mt-3 grid items-start gap-3", n === 2 && "md:grid-cols-2", n === 3 && "lg:grid-cols-3", n >= 4 && "md:grid-cols-2")}>
        {x.cases.map((c) => (c.summary && !full ? <SummaryCard key={c.system} c={c} /> : <CaseCard key={c.system} c={c} />))}
      </div>
      <div className="mt-3 flex gap-2.5 rounded-lg bg-brand/[0.06] px-3 py-2.5">
        <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-brand" />
        <p className="text-[13px] leading-relaxed text-ink-2">
          <span className="font-semibold text-ink">Why it matters: </span>
          {x.lesson}
        </p>
      </div>
    </div>
  );
}

/** One request in a line: who, what they wrote, the private reason marked. */
function RequestLine({ r }: { r: NonNullable<WorkedExample["requests"]>[number] }) {
  const parts = r.private ? r.text.split(new RegExp(`(${esc(r.private)})`, "i")) : [r.text];
  return (
    <div className="flex items-start gap-2 text-[13px] leading-snug">
      <Avatar name={r.who} small />
      <p className="min-w-0 pt-0.5">
        <span className="font-semibold text-ink">{r.who}: </span>
        <span className="text-ink-2">
          “
          {parts.map((p, i) =>
            r.private && p.toLowerCase() === r.private.toLowerCase() ? (
              <span key={i} className="rounded bg-warn/15 px-1 text-warn" title="Said in confidence: no message may repeat it">
                <Lock className="mr-0.5 inline size-3 -translate-y-px" />
                {p}
              </span>
            ) : (
              <span key={i}>{p}</span>
            ),
          )}
          ”
        </span>
      </p>
    </div>
  );
}

const CHIP: Record<string, string> = {
  ok: "border-ok/50 bg-ok/10 text-ok",
  bad: "border-bad/40 bg-bad/[0.08] text-bad line-through decoration-bad/60",
  meh: "border-line bg-panel text-ink-3",
  "": "border-line bg-panel text-ink-2",
};

/** A negotiation at a glance: whom it asked, what it offered (✓ the teacher could take it, ✗ impossible or ruled
 * out, grey allowed but not for them), the reply, and the one sentence it comes down to. */
function SummaryCard({ c }: { c: Case }) {
  const s = c.summary!;
  const [first, ...more] = s.replies;
  return (
    <div className={clsx("overflow-hidden rounded-lg border bg-panel", c.ours ? "border-brand/60 ring-1 ring-brand/20" : "border-line")}>
      <div className={clsx("h-1", STRIPE[c.tone] ?? STRIPE.muted)} />
      <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 pb-1 pt-2.5">
        <span className="text-[13.5px] font-semibold text-ink">
          {c.system}
          {c.tag && <Tag>{c.tag}</Tag>}
          {c.ours && <span className="ml-1.5 rounded bg-brand/15 px-1.5 py-px text-[10.5px] font-semibold text-brand">OURS</span>}
        </span>
        <span className={clsx("rounded-full px-2 py-0.5 text-[11.5px] font-semibold", TONE[c.tone] ?? TONE.muted)}>{c.outcome}</span>
      </div>
      <dl className="space-y-2.5 px-3.5 py-2.5 text-[12.5px]">
        <Row label="Asked">
          <span className="flex items-center gap-1.5">
            <Avatar name={s.asked} small />
            <span className="font-medium text-ink">{s.asked}</span>
            <span className={clsx("rounded px-1.5 py-px text-[11px] font-semibold", s.right ? "bg-ok/10 text-ok" : "bg-bad/10 text-bad")}>
              {s.right ? "can give way" : "can't give way"}
            </span>
          </span>
        </Row>
        <Row label="Offered">
          <span className="flex flex-wrap gap-1">
            {s.offered.map((o, i) => (
              <span key={i} title={o.note} className={clsx("rounded-md border px-1.5 py-0.5 text-[11.5px] font-medium", CHIP[o.verdict] ?? CHIP[""])}>
                {o.verdict === "ok" && "✓ "}
                {o.label}
              </span>
            ))}
          </span>
        </Row>
        {first && (
          <Row label="Reply">
            <span className="line-clamp-2 italic text-ink-2">“{first}”</span>
            {more.length > 0 && <span className="text-[11px] text-ink-3">and {more.length} more like it</span>}
          </Row>
        )}
      </dl>
      <p className={clsx("border-t border-line px-3.5 py-2 text-[12.5px] font-medium leading-snug", c.ours ? "text-ink" : "text-ink-2")}>
        {s.point}
      </p>
    </div>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[58px_1fr] items-start gap-2">
      <dt className="pt-1 text-[10.5px] font-semibold uppercase tracking-wide text-ink-3">{label}</dt>
      <dd className="min-w-0">{children}</dd>
    </div>
  );
}

const initials = (name: string) =>
  name.replace(/^(?:Dr|Prof|Mrs|Mr|Ms)\.?\s+/, "").split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();

function Avatar({ name, small = false }: { name: string; small?: boolean }) {
  const hue = [...name].reduce((h, c) => h + c.charCodeAt(0), 0) % 2 ? "bg-info/15 text-info" : "bg-warn/15 text-warn";
  return (
    <span
      className={clsx("grid shrink-0 place-items-center rounded-full font-semibold", hue, small ? "mt-0.5 size-6 text-[9.5px]" : "size-8 text-[11px]")}
      title={name}
    >
      {name ? initials(name) : <UserRound className="size-3.5" />}
    </span>
  );
}

type Case = WorkedExample["cases"][number];
type Line = Case["lines"][number];

/** Splits a conversation into rounds: each round ends with the teacher's reply. */
function rounds(lines: Line[]): Line[][] {
  const out: Line[][] = [[]];
  for (const l of lines) {
    out[out.length - 1].push(l);
    if (l.who === "Teacher") out.push([]);
  }
  return out.filter((r) => r.length > 0);
}

const STRIPE: Record<string, string> = { ok: "bg-ok", bad: "bg-bad", warn: "bg-warn", muted: "bg-ink-3" };

function CaseCard({ c }: { c: Case }) {
  const flag = [...c.outcome.matchAll(/“([^”]+)”/g)].map((m) => m[1]); // a leaked reason, named in the outcome
  const notes = c.lines.filter((l) => l.who === "Behind the scenes");
  const rs = rounds(c.lines.filter((l) => l.who !== "Behind the scenes"));
  return (
    <div className={clsx("overflow-hidden rounded-lg border bg-panel", c.ours ? "border-brand/60 ring-1 ring-brand/20" : "border-line")}>
      <div className={clsx("h-1", STRIPE[c.tone] ?? STRIPE.muted)} />
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-3.5 py-2.5">
        <span className="text-[13.5px] font-semibold text-ink">
          {c.system}
          {c.tag && <Tag>{c.tag}</Tag>}
          {c.ours && <span className="ml-1.5 rounded bg-brand/15 px-1.5 py-px text-[10.5px] font-semibold text-brand">OURS</span>}
        </span>
        <span className={clsx("rounded-full px-2 py-0.5 text-[11.5px] font-semibold", TONE[c.tone] ?? TONE.muted)}>{c.outcome}</span>
      </div>
      <div className="space-y-3 px-3.5 py-3">
        {notes.map((l, i) => (
          <p key={i} className="flex items-center gap-1.5 rounded-md bg-panel-2 px-2 py-1 text-[11.5px] text-ink-3">
            <Eye className="size-3.5 shrink-0" />
            {l.text}
          </p>
        ))}
        {rs.map((r, ri) => (
          <div key={ri} className="space-y-2">
            {rs.length > 1 && (
              <div className="flex items-center gap-2 text-[10.5px] font-semibold uppercase tracking-wide text-ink-3">
                <span className="h-px flex-1 bg-line" />
                Round {ri + 1}
                <span className="h-px flex-1 bg-line" />
              </div>
            )}
            {bubbles(r).map((b, i) => (
              <Bubble key={i} b={b} flag={flag} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

/** A system message carries the options that follow it; everything else stands alone. */
type B = { who: string; text: string; to?: string; options?: string; verdicts?: Verdict[] };
function bubbles(lines: Line[]): B[] {
  const out: B[] = [];
  for (const l of lines) {
    const last = out[out.length - 1];
    if (l.who === "Options offered" && last?.who === "System" && !last.options) {
      last.options = l.text;
      last.verdicts = l.verdicts;
    }
    else out.push({ ...l });
  }
  return out;
}

function Bubble({ b, flag }: { b: B; flag: string[] }) {
  if (b.who === "Teacher") {
    const silent = b.text.startsWith("(");
    return (
      <div className="flex justify-end gap-2">
        <div className="flex max-w-[88%] flex-col items-end">
          {b.to && <p className="mb-0.5 text-[10.5px] font-semibold text-ink-3">{b.to}</p>}
          <div
            className={clsx(
              "rounded-2xl rounded-tr-sm px-3 py-2 text-[12.5px] leading-relaxed",
              silent ? "border border-dashed border-line italic text-ink-3" : "bg-brand/10 text-ink",
            )}
          >
            {b.text}
          </div>
        </div>
        <Avatar name={b.to ?? ""} small />
      </div>
    );
  }
  const system = b.who === "System" || b.who === "Options offered";
  return (
    <div className="flex gap-2">
      <span className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-full bg-panel-2 text-ink-3" title={b.who}>
        <Bot className="size-3.5" />
      </span>
      <div className="min-w-0 max-w-[92%] rounded-2xl rounded-tl-sm bg-panel-2 px-3 py-2">
        {!system && <p className="mb-0.5 text-[10.5px] font-semibold uppercase tracking-wide text-ink-3">{b.who}</p>}
        {b.to && (
          <p className="mb-1.5 text-[11px] text-ink-3">
            To <span className="font-semibold text-ink-2">{b.to}</span>
          </p>
        )}
        {b.who === "Options offered" ? (
          <OptionList text={b.text} verdicts={b.verdicts} />
        ) : b.who === "System" ? (
          <SystemMessage text={b.text} hasOptions={!!b.options} flag={flag} />
        ) : (
          <p className="text-[12.5px] leading-relaxed text-ink">{rich(b.text, flag)}</p>
        )}
        {b.options && (
          <div className="mt-2 border-t border-line pt-2">
            <OptionList text={b.options} verdicts={b.verdicts} />
          </div>
        )}
      </div>
    </div>
  );
}

const DAY: Record<string, string> = { Monday: "Mon", Tuesday: "Tue", Wednesday: "Wed", Thursday: "Thu", Friday: "Fri", Saturday: "Sat" };

/** "the Computer Networks practical (C3-P) on Monday at 3 pm in Lab 1" -> class "C3-P", slot "Mon 3 pm", room "Lab 1". */
function parseOption(t: string) {
  const m = t.match(/^(?:the )?(.*?)\s*\(([^)]+)\)\s+on\s+(\w+)\s+at\s+(.+?)\s+in\s+(.+)$/);
  if (!m) return null;
  return { name: m[1], code: m[2], slot: `${DAY[m[3]] ?? m[3]} ${m[4]}`, room: m[5] };
}

type Verdict = { verdict: "ok" | "meh" | "bad" | ""; note: string };
const OPTION_LOOK: Record<string, { icon: typeof CheckCircle2; cls: string; row: string }> = {
  ok: { icon: CheckCircle2, cls: "text-ok", row: "border-ok/40 bg-ok/[0.06]" },
  meh: { icon: CircleDashed, cls: "text-ink-3", row: "border-line bg-panel" },
  bad: { icon: XCircle, cls: "text-bad", row: "border-bad/30 bg-bad/[0.05]" },
};

function OptionList({ text, verdicts }: { text: string; verdicts?: Verdict[] }) {
  const items = text.split(" · ");
  const parsed = items.map(parseOption);
  const codes = new Set(parsed.map((o) => o?.code));
  const shared = parsed.every(Boolean) && codes.size === 1 ? parsed[0]! : null;
  return (
    <div>
      <p className="text-[10.5px] font-semibold uppercase tracking-wide text-ink-3">
        Options
        {shared && (
          <span className="font-normal normal-case tracking-normal">
            {" "}· move {shared.name} ({shared.code})
          </span>
        )}
      </p>
      <div className="mt-1.5 grid gap-1">
        {items.map((t, i) => {
          const o = parsed[i];
          const v = verdicts?.[i];
          const look = v?.verdict ? OPTION_LOOK[v.verdict] : null;
          return (
            <div key={i} className={clsx("rounded-md border px-2 py-1 text-[12px]", look?.row ?? "border-line bg-panel")}>
              <div className="flex items-center gap-2">
                <span className="grid size-5 shrink-0 place-items-center rounded bg-panel-2 font-mono text-[11px] font-semibold text-ink-2">
                  {String.fromCharCode(65 + i)}
                </span>
                {o ? (
                  <span className="flex-1 text-ink">
                    {!shared && <span className="mr-1 font-mono text-ink-3">{o.code}</span>}
                    <span className="font-medium">{o.slot}</span>
                    <span className="text-ink-3"> · {o.room}</span>
                  </span>
                ) : (
                  <span className="flex-1 text-ink">{t}</span>
                )}
                {look && <look.icon className={clsx("size-4 shrink-0", look.cls)} />}
              </div>
              {v?.note && <p className={clsx("mt-0.5 pl-7 text-[11px] leading-snug", look?.cls)}>{v.note}</p>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// -- a system message, laid out: what was asked for, why it can't go as asked, what is offered -------------------------

/** Sentences, not split after a title ("Dr. Nair") or inside a time ("9 am."). */
function sentences(text: string): string[] {
  return text
    .split(/(?<=[.?!])\s+(?<!\b(?:Dr|Mr|Mrs|Ms|Prof|St)\.\s+)(?=[A-Z])/)
    .map((s) => s.trim())
    .filter(Boolean);
}

const strip = (s: string) => s.replace(/[.\s]+$/, "");
const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/** Splits the message into its parts. ``hasOptions``: the options are drawn below as chips, so the sentences that
 * spell them out are left out here (they say the same thing). */
function partsOf(text: string, hasOptions: boolean) {
  const asked: string[] = [];
  const why: string[] = [];
  const offers: string[] = [];
  const asks: string[] = [];
  for (const s of sentences(text)) {
    if (/\bOption [A-Z]\b/.test(s)) {
      if (!hasOptions) offers.push(...s.split(/,\s+(?:and\s+)?(?=Option [A-Z])/).map(strip).filter((o) => /^Option [A-Z]\b/.test(o)));
    } else if (/\?$/.test(s)) asks.push(s);
    else if (/^(?:You requested|You asked for)\b/.test(s)) asked.push(cap(strip(s).replace(/^You (?:requested|asked for)\s+(?:the\s+)?/, "")));
    else if (/\bneeds the\b/.test(s)) asked.push(strip(s));
    else why.push(...s.split(/,\s+and\s+(?=[A-Z])/).map(strip));
  }
  return { asked, why, offers, asks };
}

function SystemMessage({ text, hasOptions, flag }: { text: string; hasOptions: boolean; flag: string[] }) {
  const [raw, setRaw] = useState(false);
  const p = partsOf(text, hasOptions);
  const structured = p.asked.length + p.why.length + p.offers.length + p.asks.length > 1;
  if (!structured || raw)
    return (
      <div>
        <p className="text-[12.5px] leading-relaxed text-ink">{rich(text, flag)}</p>
        {structured && <RawToggle raw={raw} onClick={() => setRaw(false)} />}
      </div>
    );
  return (
    <div className="space-y-2">
      {p.asked.map((s, i) => (
        <Part key={`a${i}`} label="Asked for">
          <p className="text-[12.5px] leading-snug text-ink">{rich(s, flag)}</p>
        </Part>
      ))}
      {p.why.length > 0 && (
        <Part label={p.asked.length ? "Why it can't go as asked" : "What it says"}>
          <ul className="space-y-1">
            {p.why.map((s, i) => (
              <li key={i} className="flex gap-2 text-[12.5px] leading-snug text-ink">
                <span className="mt-[7px] size-1.5 shrink-0 rounded-full bg-ink-3/70" />
                <span className="min-w-0">{rich(s, flag)}</span>
              </li>
            ))}
          </ul>
        </Part>
      )}
      {p.offers.length > 0 && (
        <Part label="Offers">
          <ul className="space-y-1">
            {p.offers.map((s, i) => {
              const m = /^Option ([A-Z])\s+(.*)$/.exec(s);
              return (
                <li key={i} className="flex items-start gap-2 text-[12.5px] leading-snug text-ink">
                  <span className="grid size-5 shrink-0 place-items-center rounded bg-panel font-mono text-[11px] font-semibold text-ink-2">
                    {m ? m[1] : "·"}
                  </span>
                  <span className="min-w-0 pt-px">{rich(m ? cap(m[2]) : s, flag)}</span>
                </li>
              );
            })}
          </ul>
        </Part>
      )}
      {p.asks.map((s, i) => (
        <p key={`q${i}`} className="text-[12.5px] italic leading-snug text-ink-2">{rich(s, flag)}</p>
      ))}
      <RawToggle raw={raw} onClick={() => setRaw(true)} />
    </div>
  );
}

function Part({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <p className="mb-0.5 text-[10.5px] font-semibold uppercase tracking-wide text-ink-3">{label}</p>
      {children}
    </div>
  );
}

function RawToggle({ raw, onClick }: { raw: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} className="text-[11px] text-ink-3 underline-offset-2 hover:text-ink hover:underline">
      {raw ? "Show laid out" : "Show the message as sent"}
    </button>
  );
}

const DAY_W = "(?:Mon|Tues|Wednes|Thurs|Fri|Satur)day";
const CLK = "\\d{1,2}(?::\\d{2})?\\s?(?:am|pm)";
const RICH = new RegExp(
  [
    `(\\[[A-Z][A-Z0-9-]+\\])`, // a rule id
    `(\\b(?:Room|Lab|Hall)\\s+\\d+\\b)`, // a room
    `(\\(([A-Z]\\d+-[A-Z])\\))`, // a class code
    `(\\b${DAY_W}(?:,?\\s+(?:from\\s+${CLK}\\s+to\\s+${CLK}|at\\s+${CLK}|in the (?:morning|afternoon)|afternoon|morning))?)`, // a day, with its time
    `(\\b(?:from\\s+)?${CLK}(?:\\s+to\\s+${CLK})?)`, // a time
  ].join("|"),
  "g",
);

/** The message's facts picked out: days and times in medium weight, rooms and classes as tags, the rule a reason
 * comes from as a rule tag; ``flag``: words to mark as a problem (a private reason that leaked). */
function rich(text: string, flag: string[]): ReactNode[] {
  const marked = flag.filter(Boolean);
  const out: ReactNode[] = [];
  let k = 0;
  const pieces = marked.length ? text.split(new RegExp(`(${marked.map(esc).join("|")})`, "gi")) : [text];
  for (const piece of pieces) {
    if (marked.some((f) => f.toLowerCase() === piece.toLowerCase())) {
      out.push(<mark key={k++} className="rounded bg-bad/15 px-0.5 font-medium text-bad">{piece}</mark>);
      continue;
    }
    let at = 0;
    for (const m of piece.matchAll(RICH)) {
      if (m.index! > at) out.push(piece.slice(at, m.index));
      if (m[1]) out.push(<span key={k++} className="rounded bg-info/10 px-1 py-px font-mono text-[10.5px] text-info">{m[1].slice(1, -1)}</span>);
      else if (m[2]) out.push(<span key={k++} className="whitespace-nowrap rounded border border-line bg-panel px-1 py-px text-[11.5px]">{m[2]}</span>);
      else if (m[3]) out.push(<span key={k++} className="ml-0.5 rounded bg-panel px-1 py-px font-mono text-[10.5px] text-ink-3">{m[4]}</span>);
      else out.push(<span key={k++} className="font-semibold">{m[0]}</span>);
      at = m.index! + m[0].length;
    }
    if (at < piece.length) out.push(piece.slice(at));
  }
  return out;
}

const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
