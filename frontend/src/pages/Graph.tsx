import clsx from "clsx";
import { ArrowRight, Info, Network, Search, Swords, X } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { EDGE_STYLE, KIND, KnowledgeGraph, tierColor } from "../components/KnowledgeGraph";
import { Card, CardHeader, PageHeader, Skeleton } from "../components/ui";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { ROLE_LABEL, TIER } from "../lib/meta";
import type { GraphData, GraphNode, Role } from "../lib/types";

/* The knowledge graph as a tool: pick a question (a view), find a person, class, room or request, and click
 * anything to see what it is and what it is tied to, in words. Every link in the details panel moves the
 * focus, so a chain (a request -> its constraint -> the class it touches -> who teaches it -> who they
 * report to) can be followed by clicking. */

type Edge = GraphData["edges"][number];

/** The relations as the page names them: "has_authority_over" becomes teaches / oversees / manages. */
function normalise(data: GraphData): Edge[] {
  const byId = new Map(data.nodes.map((n) => [n.id, n]));
  const teacher = new Map<string, string>();
  for (const n of data.nodes) {
    const t = n.details.find(([k]) => k === "Teacher");
    if (n.kind === "session" && t) teacher.set(n.id, t[1]);
  }
  return data.edges.map((e) => {
    if (e.rel !== "has_authority_over") return e;
    const target = byId.get(e.target);
    const source = byId.get(e.source);
    if (target?.kind === "room") return { ...e, rel: "manages" };
    if (target?.kind === "session") return { ...e, rel: teacher.get(e.target) === source?.label ? "teaches" : "oversees" };
    return { ...e, rel: "oversees" };
  });
}

const VIEWS: { id: string; title: string; question: string; rels: string[]; kinds?: string[] }[] = [
  {
    id: "requests",
    title: "Requests and what they ask",
    question: "Each request becomes constraints; each constraint has an owner and touches classes, rooms or sections.",
    rels: ["derived_from", "owns", "references"],
  },
  {
    id: "conflicts",
    title: "Clashes found in negotiations",
    question: "Constraints that could not all hold together, who owns them, and what they touch. This is what the negotiator resolves.",
    rels: ["conflicts_with", "owns", "references"],
  },
  {
    id: "people",
    title: "Who answers to whom",
    question: "The reporting chain: an escalation goes up it to the person with authority over both sides.",
    rels: ["reports_to", "manages"],
    kinds: ["stakeholder", "room"],
  },
  {
    id: "teaching",
    title: "Who teaches which section",
    question: "Teachers, their classes and the sections attending: who is affected when a class moves.",
    rels: ["teaches", "attended_by", "member_of"],
  },
  { id: "all", title: "Everything", question: "Every relation at once. Dense: search or click to find your way.", rels: Object.keys(EDGE_STYLE) },
];

/** How a relation reads from each end, for the details panel. */
const PHRASE: Record<string, [string, string]> = {
  owns: ["Owns", "Owned by"],
  derived_from: ["Comes from", "Became"],
  references: ["Touches", "Touched by"],
  reports_to: ["Reports to", "Reporting to them"],
  teaches: ["Teaches", "Taught by"],
  oversees: ["Oversees", "Overseen by"],
  manages: ["Manages", "Managed by"],
  attended_by: ["Attended by", "Attends"],
  member_of: ["Represents", "Represented by"],
  conflicts_with: ["Clashes with", "Clashes with"],
  affects: ["Affects", "Affected by"],
};

export default function Graph() {
  const { data } = useApi(() => api.graph(), [], 15000);
  const [viewId, setViewId] = useState("requests");
  // ?node=F-105 opens on that node (links from other pages, and shareable)
  const [selected, setSelected] = useState<string | null>(() => new URLSearchParams(window.location.search).get("node"));
  const [onlyAround, setOnlyAround] = useState(false);
  const [q, setQ] = useState("");

  const edges = useMemo(() => (data ? normalise(data) : []), [data]);
  const byId = useMemo(() => new Map((data?.nodes ?? []).map((n) => [n.id, n])), [data]);
  const view = VIEWS.find((v) => v.id === viewId) ?? VIEWS[0];

  // the nodes and edges of the view (and, when asked, only around the selected node)
  const shown = useMemo(() => {
    if (!data) return { nodes: [] as GraphNode[], edges: [] as Edge[] };
    let es = edges.filter((e) => view.rels.includes(e.rel));
    if (view.id === "conflicts") {
      // only the constraints that clashed, their owners and what they touch
      const clashed = new Set(es.filter((e) => e.rel === "conflicts_with").flatMap((e) => [e.source, e.target]));
      es = es.filter((e) => e.rel === "conflicts_with" || (e.rel === "owns" && clashed.has(e.target)) || (e.rel === "references" && clashed.has(e.source)));
    }
    if (view.kinds) es = es.filter((e) => view.kinds!.includes(byId.get(e.source)?.kind ?? "") && view.kinds!.includes(byId.get(e.target)?.kind ?? ""));
    if (onlyAround && selected) {
      const near = new Set([selected]);
      for (let hop = 0; hop < 2; hop++)
        for (const e of edges) {
          if (near.has(e.source)) near.add(e.target);
          else if (near.has(e.target)) near.add(e.source);
        }
      es = edges.filter((e) => near.has(e.source) && near.has(e.target) && e.rel !== "oversees");
    }
    const used = new Set(es.flatMap((e) => [e.source, e.target]));
    if (selected) used.add(selected);
    return { nodes: data.nodes.filter((n) => used.has(n.id)), edges: es };
  }, [data, edges, view, onlyAround, selected, byId]);

  const matches = useMemo(() => {
    const t = q.trim().toLowerCase();
    if (!t || !data) return [];
    return data.nodes.filter((n) => n.label.toLowerCase().includes(t) || n.title.toLowerCase().includes(t) || n.id.toLowerCase().includes(t)).slice(0, 8);
  }, [q, data]);

  const pick = (id: string) => {
    setSelected(id);
    setQ("");
    // a node outside the view: show its neighbourhood instead, so it can be seen
    if (!shown.nodes.some((n) => n.id === id)) setOnlyAround(true);
  };

  return (
    <>
      <PageHeader
        title="Knowledge graph"
        subtitle="Who owns which constraint, what each request touches, who reports to whom, and what clashed. Click anything to see what it is tied to."
      />
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            onClick={() => {
              setViewId(v.id);
              setOnlyAround(false);
            }}
            className={clsx(
              "flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-[12.5px] transition-colors",
              v.id === viewId ? "border-brand bg-brand/10 text-ink" : "border-line text-ink-2 hover:bg-panel-2",
            )}
          >
            {v.id === "conflicts" && <Swords size={12} className="text-bad" />}
            {v.title}
            {v.id === "conflicts" && data && <span className="text-ink-3">({data.conflicts.length})</span>}
          </button>
        ))}
        <div className="relative ml-auto w-full sm:w-72">
          <Search size={14} className="absolute top-1/2 left-2.5 -translate-y-1/2 text-ink-3" />
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Find a person, class, room, request…"
            className="w-full rounded-md border border-line bg-panel py-1.5 pr-3 pl-8 text-[13px] outline-none focus:border-brand"
          />
          {matches.length > 0 && (
            <div className="absolute z-20 mt-1 w-full overflow-hidden rounded-md border border-line bg-panel shadow-lg">
              {matches.map((n) => (
                <button key={n.id} onClick={() => pick(n.id)} className="flex w-full items-start gap-2 px-3 py-2 text-left hover:bg-panel-2">
                  <KindDot n={n} />
                  <span className="min-w-0">
                    <span className="block truncate text-[13px]">{n.title}</span>
                    <span className="text-[11px] text-ink-3">
                      {KIND[n.kind]?.label ?? n.kind} · {n.id}
                    </span>
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_340px]">
        <Card className="overflow-hidden">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-line px-4 py-2.5 text-[12.5px]">
            <Info size={13} className="shrink-0 text-ink-3" />
            <span className="min-w-0 flex-1 truncate text-ink-2" title={view.question}>{onlyAround && selected ? `Around ${byId.get(selected)?.title ?? selected}: two steps out, every relation.` : view.question}</span>
            {/* always there (greyed out without a selection), so the bar, and the canvas under it, never shift */}
            <label className={clsx("flex shrink-0 items-center gap-1.5", selected ? "text-ink-2" : "text-ink-3/60")} title={selected ? undefined : "Select a node first"}>
              <input
                type="checkbox"
                disabled={!selected}
                checked={onlyAround && !!selected}
                onChange={(e) => setOnlyAround(e.target.checked)}
                className="accent-[var(--brand)]"
              />
              only around the selection
            </label>
          </div>
          <div className="relative h-[620px]">
            {!data ? (
              <Skeleton className="h-full" />
            ) : shown.nodes.length === 0 ? (
              <div className="grid h-full place-items-center p-8 text-center text-[13px] text-ink-3">
                {viewId === "conflicts" ? "No clashes yet: every request so far fitted, or was settled before a negotiation." : "Nothing to show in this view yet."}
              </div>
            ) : (
              <KnowledgeGraph nodes={shown.nodes} edges={shown.edges} selected={selected} onSelect={setSelected} labelAll={onlyAround} />
            )}
          </div>
          <Legend edges={shown.edges} nodes={shown.nodes} />
        </Card>

        <div className="space-y-6">
          {data && selected && byId.get(selected) ? (
            <Details node={byId.get(selected)!} edges={edges} byId={byId} onPick={pick} onClose={() => setSelected(null)} data={data} />
          ) : (
            data && <Overview data={data} byId={byId} onPick={pick} />
          )}
        </div>
      </div>

      {data && data.conflicts.length > 0 && (
        <Card className="mt-6">
          <CardHeader icon={Swords} title="Clashes the negotiator found" subtitle="Each set could not all hold at once; the cheapest one to relax was negotiated with its owner" />
          <div className="divide-y divide-line">
            {data.conflicts.map((c, i) => (
              <div key={i} className="flex flex-wrap items-start gap-4 px-5 py-3.5">
                <div className="w-52 shrink-0">
                  {c.case && (
                    <Link to={`/requests/${c.case}`} className="font-mono text-[12px] text-brand hover:underline">
                      {c.case}
                    </Link>
                  )}
                  <span className="text-[12px] text-ink-3"> · round {c.round ?? 1}</span>
                  <p className="mt-0.5 line-clamp-2 text-[12px] text-ink-3">{c.sender ? `${c.sender}: ` : ""}{c.request}</p>
                </div>
                <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2">
                  {c.constraints.map((id, j) => (
                    <span key={id} className="flex items-center gap-2">
                      {j > 0 && <X size={13} className="text-bad" />}
                      <button onClick={() => { setViewId("conflicts"); pick(id); }} className="max-w-xs rounded-md border border-line px-2.5 py-1.5 text-left text-[12px] hover:border-brand">
                        {byId.get(id)?.title ?? id}
                      </button>
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </>
  );
}

function KindDot({ n }: { n: GraphNode }) {
  const color = n.kind === "constraint" ? tierColor(n.tier) : KIND[n.kind]?.color ?? "var(--ink-3)";
  return <span className={clsx("mt-1 size-2.5 shrink-0", n.kind === "constraint" ? "rotate-45" : n.kind === "room" ? "rounded-[2px]" : "rounded-full")} style={{ background: color }} />;
}

function Legend({ nodes, edges }: { nodes: GraphNode[]; edges: Edge[] }) {
  const kinds = [...new Set(nodes.map((n) => n.kind))];
  const rels = [...new Set(edges.map((e) => e.rel))];
  const hasConstraints = kinds.includes("constraint");
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-line px-4 py-3 text-[11.5px] text-ink-2">
      {kinds.map((k) => (
        <span key={k} className="flex items-center gap-1.5">
          <svg width="14" height="14" viewBox="-7 -7 14 14">
            {k === "constraint" ? (
              <path d="M0 -6 L6 0 L0 6 L-6 0 Z" fill="var(--ink-3)" />
            ) : k === "room" ? (
              <rect x="-5" y="-5" width="10" height="10" rx="1.5" fill={KIND[k].color} />
            ) : k === "group" ? (
              <rect x="-6" y="-4.5" width="12" height="9" rx="4" fill={KIND[k].color} />
            ) : k === "request" || k === "rule" ? (
              <path d="M0 -6 L5.5 4.5 L-5.5 4.5 Z" fill={KIND[k].color} />
            ) : (
              <circle r="5" fill={KIND[k]?.color ?? "var(--ink-3)"} />
            )}
          </svg>
          {KIND[k]?.label ?? k}
        </span>
      ))}
      {hasConstraints && (
        <span className="flex items-center gap-1.5">
          <span className="text-ink-3">constraint tier:</span>
          {[1, 2, 3, 4, 5].map((t) => (
            <span key={t} className="flex items-center gap-1" title={TIER[t]?.name}>
              <span className="size-2.5 rotate-45" style={{ background: tierColor(t) }} />
              {t}
            </span>
          ))}
          <span className="ml-1 text-ink-3">hollow = preference</span>
        </span>
      )}
      <span className="h-3 w-px bg-line" />
      {rels.map((r) => {
        const st = EDGE_STYLE[r] ?? { color: "var(--ink-3)", label: r };
        return (
          <span key={r} className="flex items-center gap-1.5">
            <svg width="22" height="8">
              <line x1="0" y1="4" x2="22" y2="4" stroke={st.color} strokeWidth="2" strokeDasharray={st.dash} />
            </svg>
            {st.label}
          </span>
        );
      })}
    </div>
  );
}

/** What a node is, in a sentence. */
function summary(n: GraphNode, out: Map<string, GraphNode[]>, into: Map<string, GraphNode[]>): string {
  const count = (m: Map<string, GraphNode[]>, r: string) => m.get(r)?.length ?? 0;
  const get = (k: string) => n.details.find(([a]) => a === k)?.[1];
  switch (n.kind) {
    case "stakeholder": {
      const owns = count(out, "owns");
      const boss = out.get("reports_to")?.[0];
      return `${n.title} owns ${owns} constraint${owns === 1 ? "" : "s"}${get("Teaches") ? ` and teaches ${get("Teaches")}` : ""}; has given way ${n.given ?? 0} time${n.given === 1 ? "" : "s"}. ${
        boss ? `If a clash with them cannot be agreed, it goes to ${boss.title}.` : "Escalations stop here."
      }`;
    }
    case "constraint":
      return `${n.hard ? "It is a requirement: it can only be relaxed by negotiation or someone with authority." : "It is a preference: the solver drops it if it cannot be kept, and tells the owner."}${
        count(out, "conflicts_with") ? ` It clashed with ${count(out, "conflicts_with")} other constraint${count(out, "conflicts_with") === 1 ? "" : "s"}.` : ""
      }`;
    case "session":
      return `${n.title} for ${get("Sections") ?? "–"}, taught by ${get("Teacher") ?? "–"}, ${get("When") ?? "not placed"}. ${count(into, "references")} constraint${count(into, "references") === 1 ? "" : "s"} touch it.`;
    case "room":
      return `${n.title}, ${get("Type")} room for ${get("Capacity")}. ${count(into, "references")} constraint${count(into, "references") === 1 ? "" : "s"} touch it.`;
    case "group":
      return `${n.title}, ${get("Students")} students, attending ${count(into, "attended_by")} classes. A change to any of them is sent to this section.`;
    case "request":
      return `A request from ${get("From") ?? "someone"} (${get("Status") ?? "–"}), read into ${count(into, "derived_from")} constraint${count(into, "derived_from") === 1 ? "" : "s"}.`;
    default:
      return n.title;
  }
}

function Details({
  node,
  edges,
  byId,
  onPick,
  onClose,
}: {
  node: GraphNode;
  edges: Edge[];
  byId: Map<string, GraphNode>;
  onPick: (id: string) => void;
  onClose: () => void;
  data: GraphData;
}) {
  const out = new Map<string, GraphNode[]>();
  const into = new Map<string, GraphNode[]>();
  for (const e of edges) {
    if (e.source === node.id && byId.has(e.target)) out.set(e.rel, [...(out.get(e.rel) ?? []), byId.get(e.target)!]);
    if (e.target === node.id && byId.has(e.source)) into.set(e.rel, [...(into.get(e.rel) ?? []), byId.get(e.source)!]);
  }
  // each neighbour listed once, under the first relation that names it ("Owns" before "Touched by")
  const groups: [string, GraphNode[]][] = [];
  const seen = new Set<string>();
  const add = (label: string, ns: GraphNode[]) => {
    const fresh = ns.filter((n) => !seen.has(n.id));
    fresh.forEach((n) => seen.add(n.id));
    if (fresh.length) groups.push([label, fresh]);
  };
  for (const [r, ns] of out) add(PHRASE[r]?.[0] ?? r, ns);
  for (const [r, ns] of into) if (r !== "conflicts_with") add(PHRASE[r]?.[1] ?? r, ns);
  const color = node.kind === "constraint" ? tierColor(node.tier) : KIND[node.kind]?.color;
  return (
    <Card>
      <div className="flex items-start justify-between gap-3 border-b border-line px-5 py-3.5">
        <div className="min-w-0">
          <p className="flex items-center gap-1.5 text-[11px] tracking-wide uppercase" style={{ color }}>
            <KindDot n={node} /> {KIND[node.kind]?.label ?? node.kind}
            {node.kind === "constraint" && node.tier != null && <span className="text-ink-3 normal-case">· Tier {node.tier}, {TIER[node.tier]?.name}</span>}
          </p>
          <h3 className="mt-1 text-[15px] font-semibold leading-snug">{node.title}</h3>
        </div>
        <button onClick={onClose} className="rounded p-1 text-ink-3 hover:bg-panel-2 hover:text-ink" title="Close">
          <X size={15} />
        </button>
      </div>
      <div className="space-y-4 p-5 text-[13px]">
        <p className="leading-relaxed text-ink-2">{summary(node, out, into)}</p>
        {node.details.length > 0 && (
          <dl className="grid grid-cols-[110px_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-[12.5px]">
            {node.details.map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="text-ink-3">{k}</dt>
                <dd className="min-w-0 break-words">
                  {/^R-\d+/.test(v) ? (
                    <Link to={`/requests/${v}`} className="font-mono text-brand hover:underline">
                      {v}
                    </Link>
                  ) : k === "Role" ? (
                    ROLE_LABEL[v.replace(/ /g, "_") as Role] ?? v
                  ) : (
                    v
                  )}
                </dd>
              </div>
            ))}
          </dl>
        )}
        {groups.map(([label, ns]) => (
          <div key={label}>
            <p className="mb-1.5 text-[11.5px] font-medium text-ink-3">
              {label} <span className="font-normal">({ns.length})</span>
            </p>
            <div className="space-y-1">
              {ns.slice(0, 12).map((n) => (
                <button key={n.id} onClick={() => onPick(n.id)} className="flex w-full items-start gap-2 rounded-md px-2 py-1.5 text-left hover:bg-panel-2">
                  <KindDot n={n} />
                  <span className="min-w-0 flex-1 text-[12.5px] leading-snug">
                    {n.title}
                    {n.kind === "session" && (
                      <span className="block text-[11px] text-ink-3">{n.details.find(([k]) => k === "When")?.[1]}</span>
                    )}
                  </span>
                  <ArrowRight size={12} className="mt-1 shrink-0 text-ink-3" />
                </button>
              ))}
              {ns.length > 12 && <p className="px-2 text-[11.5px] text-ink-3">and {ns.length - 12} more</p>}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

function Overview({ data, byId, onPick }: { data: GraphData; byId: Map<string, GraphNode>; onPick: (id: string) => void }) {
  const count = (k: string) => data.nodes.filter((n) => n.kind === k).length;
  const people = data.nodes.filter((n) => n.kind === "stakeholder" && n.given != null).sort((a, b) => (b.given ?? 0) - (a.given ?? 0));
  const tiers = [0, 1, 2, 3, 4, 5].map((t) => ({ t, n: data.nodes.filter((n) => n.kind === "constraint" && n.tier === t).length })).filter((x) => x.n);
  const busiest = data.nodes
    .filter((n) => n.kind === "session" || n.kind === "room")
    .map((n) => ({ n, refs: data.edges.filter((e) => e.target === n.id && e.rel === "references").length }))
    .filter((x) => x.refs > 0)
    .sort((a, b) => b.refs - a.refs)
    .slice(0, 5);
  return (
    <>
      <Card>
        <CardHeader icon={Network} title="At a glance" subtitle="Click a node, or anything below, for details" />
        <div className="grid grid-cols-3 gap-2 p-4 text-center">
          {[
            ["stakeholder", "people"],
            ["constraint", "constraints"],
            ["request", "requests"],
            ["session", "classes"],
            ["room", "rooms"],
            ["group", "sections"],
          ].map(([k, l]) => (
            <div key={k} className="rounded-md border border-line py-2">
              <p className="font-serif text-[20px] font-semibold">{count(k)}</p>
              <p className="text-[11px] text-ink-3">{l}</p>
            </div>
          ))}
        </div>
        {tiers.length > 0 && (
          <div className="border-t border-line px-4 py-3">
            <p className="mb-2 text-[11.5px] font-medium text-ink-3">Constraints in force, by tier</p>
            <div className="flex h-2.5 overflow-hidden rounded-full bg-panel-2">
              {tiers.map(({ t, n }) => (
                <div key={t} style={{ width: `${(n / tiers.reduce((a, x) => a + x.n, 0)) * 100}%`, background: tierColor(t) }} title={`Tier ${t}: ${n}`} />
              ))}
            </div>
            <div className="mt-1.5 flex flex-wrap gap-x-3 text-[11px] text-ink-3">
              {tiers.map(({ t, n }) => (
                <span key={t}>
                  T{t} {TIER[t]?.name}: <span className="text-ink">{n}</span>
                </span>
              ))}
            </div>
          </div>
        )}
      </Card>
      {people.length > 0 && (
        <Card>
          <CardHeader title="Who has given way" subtitle="Concessions so far; fewer means they are asked first next time" />
          <div className="space-y-0.5 p-3">
            {people.slice(0, 8).map((n) => (
              <button key={n.id} onClick={() => onPick(n.id)} className="flex w-full items-center gap-3 rounded-md px-2 py-1.5 text-left text-[12.5px] hover:bg-panel-2">
                <span className="min-w-0 flex-1 truncate">{n.title}</span>
                <span className="flex gap-0.5">
                  {Array.from({ length: Math.max(1, n.given ?? 0) }).map((_, i) => (
                    <span key={i} className={clsx("h-3 w-1.5 rounded-sm", n.given ? "bg-brand" : "bg-panel-2")} />
                  ))}
                </span>
                <span className="w-5 text-right font-mono text-ink-3">{n.given ?? 0}</span>
              </button>
            ))}
          </div>
        </Card>
      )}
      {busiest.length > 0 && (
        <Card>
          <CardHeader title="Most constrained" subtitle="Classes and rooms the most constraints touch: where clashes are likeliest" />
          <div className="space-y-0.5 p-3">
            {busiest.map(({ n, refs }) => (
              <button key={n.id} onClick={() => onPick(n.id)} className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-[12.5px] hover:bg-panel-2">
                <KindDot n={byId.get(n.id) ?? n} />
                <span className="min-w-0 flex-1 truncate">{n.title}</span>
                <span className="font-mono text-ink-3">{refs}</span>
              </button>
            ))}
          </div>
        </Card>
      )}
    </>
  );
}
