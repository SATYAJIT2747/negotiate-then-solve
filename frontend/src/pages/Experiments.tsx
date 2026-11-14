import clsx from "clsx";
import { useState, type ReactNode } from "react";
import {
  BookOpen, CheckCircle2, ChevronDown, CircleDashed, Cpu, FlaskConical, MessagesSquare, Scale, Search, ShieldCheck,
  Swords, XCircle,
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
import type { ChapterChart, Experiment, TableCol, TableRow } from "../lib/types";

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
