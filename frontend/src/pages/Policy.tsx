import clsx from "clsx";
import { Search } from "lucide-react";
import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Card, CardHeader, Label, PageHeader, Skeleton, Tabs } from "../components/ui";
import PolicyDocuments from "./PolicyDocuments";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useApi, useDebounced } from "../lib/hooks";
import { how } from "../lib/meta";
import type { Rule } from "../lib/types";

const MODES = ["bm25", "dense", "hybrid"];
const SAMPLES = ["Can I teach at 1pm?", "Move my lecture to week 8", "Shift my practical to another lab", "Guest lecturer wants a different day", "Book the auditorium"];

function where(r: Rule): string {
  const pages = r.pages.length ? ` · p.${r.pages.join(", ")}` : "";
  return `${r.source}${pages} · ${how(r.method)}`;
}

function Rules() {
  const { data: rules } = useApi(() => api.handbook(), []);
  const [q, setQ] = useState("Move my lecture to week 8");
  const dq = useDebounced(q, 220);
  const [mode, setMode] = useState<string | undefined>(undefined); // undefined: the policy agent's retriever
  const { data: hits } = useApi(() => (dq.trim() ? api.search(dq, mode) : Promise.resolve(null)), [dq, mode]);
  const maxScore = Math.max(0.001, ...(hits?.results.map((r) => r.score ?? 0) ?? [0]));
  const [open, setOpen] = useState<string | null>(null);
  const bySource = new Map<string, Rule[]>();
  for (const r of rules ?? []) bySource.set(r.source, [...(bySource.get(r.source) ?? []), r]);

  return (
    <>
      <Card className="mb-6">
        <div className="p-5">
          <div className="flex items-center gap-3 rounded-md border border-line bg-panel px-3.5 focus-within:border-brand/60">
            <Search size={18} className="text-ink-3" />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask like a faculty member would…" className="h-11 flex-1 bg-transparent text-[14.5px] outline-none" />
            <div className="flex rounded border border-line text-[12px]" title="BM25 matches words; dense matches meanings; hybrid fuses both (reciprocal rank fusion)">
              {MODES.map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={clsx("px-2 py-1", (mode ?? hits?.mode) === m ? "bg-panel-2 text-ink" : "text-ink-3 hover:text-ink-2")}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {SAMPLES.map((s) => (
              <button key={s} onClick={() => setQ(s)} className="rounded-full border border-line px-3 py-1 text-[12px] text-ink-2 hover:border-brand/40">
                {s}
              </button>
            ))}
          </div>
          {hits && (
            <p className="mt-4 flex flex-wrap items-center gap-1.5 text-[12px] text-ink-3">
              Query tokens:
              {hits.tokens.map((t) => (
                <span key={t} className="rounded bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-ink-2">
                  {t}
                </span>
              ))}
              <span className="ml-1">(clock times become h13, h14…; used by BM25)</span>
            </p>
          )}
        </div>
        <div className="space-y-2 border-t border-line p-5">
          {hits?.results.map((r, i) => (
            <div key={r.id} className={clsx("rounded-md border p-4", i === 0 ? "border-brand/40" : "border-line")}>
              <div className="flex items-center gap-3">
                <span className="grid size-6 shrink-0 place-items-center rounded bg-panel-2 font-mono text-[11px]">{i + 1}</span>
                <span className="text-[13.5px] font-medium">
                  {r.number && `${r.number} `}
                  {r.title}
                </span>
                <span className="font-mono text-[11.5px] text-ink-3">{r.id}</span>
                <div className="ml-auto flex w-36 items-center gap-2">
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-panel-2">
                    <div className="h-full bg-brand" style={{ width: `${((r.score ?? 0) / maxScore) * 100}%` }} />
                  </div>
                  <span className="w-9 text-right font-mono text-[11px] text-ink-3">{r.score?.toFixed(hits?.mode === "bm25" ? 2 : 3)}</span>
                </div>
              </div>
              <p className="mt-2 text-[13px] leading-relaxed text-ink-2">{r.text}</p>
              <p className="mt-2 text-[12px] text-ink-3">{where(r)}</p>
            </div>
          ))}
          {hits && !hits.results.length && <p className="py-6 text-center text-[13px] text-ink-3">No rule matches, so the agent would see nothing and could not deny.</p>}
        </div>
      </Card>

      <Card>
        <CardHeader title="All rules" subtitle={`${rules?.length ?? 0} rules; each keeps the document and page it came from, so citations can be checked`} />
        {rules ? (
          <div className="divide-y divide-line">
            {[...bySource.entries()].map(([source, list]) => (
              <div key={source} className="px-5 py-4">
                <Label>
                  {source} · {list.length} rule{list.length === 1 ? "" : "s"}
                </Label>
                <div>
                  {list.map((r) => (
                    <button key={r.id} onClick={() => setOpen(open === r.id ? null : r.id)} className="block w-full rounded-md px-2 py-2 text-left hover:bg-panel-2/60">
                      <div className="flex items-center gap-3">
                        <span className="w-10 shrink-0 font-mono text-[12px] text-ink-3">{r.number}</span>
                        <span className="flex-1 text-[14px]">{r.title}</span>
                        <span className="font-mono text-[11.5px] text-ink-3">{r.id}</span>
                      </div>
                      {open === r.id && (
                        <div className="pl-[52px] pt-2">
                          <p className="text-[13px] leading-relaxed text-ink-2">{r.text}</p>
                          <p className="mt-1.5 text-[12px] text-ink-3">{where(r)}</p>
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <Skeleton className="m-5 h-48" />
        )}
      </Card>
    </>
  );
}

/** The handbook and its documents in one place: every rule the policy agent can cite, for everyone; and, for
 * the timetable office, the documents those rules are read from (added and removed in the Documents tab). */
export default function Policy() {
  const { sees } = useAuth();
  const [params, setParams] = useSearchParams();
  const manage = sees("documents");
  const tab = manage && params.get("tab") === "documents" ? "documents" : "rules";
  return (
    <>
      <PageHeader
        title="Handbook & policies"
        subtitle={
          tab === "documents"
            ? "The institute's regulations, circulars and notices. The policy agent retrieves and cites their rules."
            : "Every rule the policy agent can cite, with the document and page it comes from."
        }
      />
      {manage && (
        <div className="mb-5">
          <Tabs
            tabs={[
              { id: "rules", label: "Rules" },
              { id: "documents", label: "Documents" },
            ]}
            value={tab}
            onChange={(t) => setParams(t === "rules" ? {} : { tab: t }, { replace: true })}
          />
        </div>
      )}
      {tab === "documents" ? <PolicyDocuments embedded /> : <Rules />}
    </>
  );
}
