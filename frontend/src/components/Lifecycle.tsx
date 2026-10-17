import clsx from "clsx";
import { Check } from "lucide-react";
import { Link } from "react-router-dom";
import { ago, clock, EVENT_LABEL, STATUS, TONE_CLASS } from "../lib/meta";
import type { EventRow, Status } from "../lib/types";
import { LiveDot } from "./ui";

const MAIN: { status: Status; short: string }[] = [
  { status: "received", short: "Received" },
  { status: "classified", short: "Classified" },
  { status: "policy_checked", short: "Policy" },
  { status: "compiled", short: "Compiled" },
  { status: "solved", short: "Solved" },
  { status: "negotiating", short: "Negotiation" },
  { status: "fairness_audited", short: "Fairness" },
  { status: "awaiting_approval", short: "Approval" },
  { status: "published", short: "Published" },
];
const EXITS: Status[] = ["withdrawn", "refused", "denied", "clarification_requested", "escalated", "answered", "forwarded"];

/** The request lifecycle (proposal Figure 2), lit up from the event log. */
export function Lifecycle({ events, status, live }: { events: EventRow[]; status: Status; live?: boolean }) {
  const seen = new Set(events.map((e) => e.kind));
  const reached = MAIN.map((m) => seen.has(m.status) || (m.status === "received" && events.length > 0));
  const skippedNegotiation = seen.has("fairness_audited") && !seen.has("negotiating");
  const last = MAIN.reduce((acc, _m, i) => (reached[i] ? i : acc), 0);
  const exit = EXITS.find((x) => seen.has(x));
  const exitMeta = exit ? STATUS[exit] : null;

  return (
    <div>
      <ol className="flex items-start overflow-x-auto pb-1">
        {MAIN.map((m, i) => {
          const done = reached[i];
          const skipped = m.status === "negotiating" && skippedNegotiation;
          const current = i === last && !STATUS[status]?.terminal && live;
          return (
            <li key={m.status} className="flex min-w-[76px] flex-1 flex-col items-center">
              <div className="flex w-full items-center">
                <span className={clsx("h-px flex-1", i === 0 ? "opacity-0" : done ? "bg-brand" : "bg-line")} />
                <span
                  className={clsx(
                    "grid size-5 shrink-0 place-items-center rounded-full border text-[10px] font-semibold",
                    skipped ? "border-dashed border-ink-3 text-ink-3" : done ? "border-brand bg-brand text-panel" : "border-line bg-panel text-ink-3",
                    current && "animate-pulse",
                  )}
                >
                  {done && !skipped ? <Check size={11} strokeWidth={3} /> : i + 1}
                </span>
                <span className={clsx("h-px flex-1", i === MAIN.length - 1 ? "opacity-0" : reached[i + 1] ? "bg-brand" : "bg-line")} />
              </div>
              <p className={clsx("mt-1.5 text-center text-[11.5px]", done && !skipped ? "text-ink" : "text-ink-3")}>
                {m.short}
                {skipped && <span className="block text-[10.5px]">not needed</span>}
              </p>
            </li>
          );
        })}
      </ol>
      {exitMeta && (
        <p className="mt-3 flex items-center gap-2 text-[13px] text-ink-2">
          <exitMeta.icon size={15} className={TONE_CLASS[exitMeta.tone].text} />
          <span className="font-medium text-ink">Stopped safely: {exitMeta.label.toLowerCase()}.</span>
          Nothing reached the timetable.
        </p>
      )}
    </div>
  );
}

const DETAIL_KEYS = ["route", "action", "verdict", "reason", "to", "version", "approver", "rounds", "choice", "constraints", "superseded", "moved", "cited", "obligations", "seconds", "solver_seconds", "gini_after"];

function detail(e: EventRow): string {
  const parts: string[] = [];
  for (const k of DETAIL_KEYS) {
    const v = e[k];
    if (v === undefined || v === null || v === "" || (Array.isArray(v) && !v.length)) continue;
    parts.push(`${k.replace(/_/g, " ")}: ${Array.isArray(v) ? v.join(", ") : typeof v === "number" ? +v.toFixed(3) : String(v)}`);
  }
  if (e.kind === "negotiation_message" && e.message && typeof e.message === "object") {
    const m = e.message as { to: string; offers: unknown[] };
    parts.push(`to ${m.to} with ${m.offers.length} option(s)`);
  }
  if (e.kind === "negotiation_reply" && e.reply && typeof e.reply === "object") {
    const r = e.reply as { decision: string; choice?: string };
    parts.push(`${r.decision}${r.choice ? ` ${r.choice}` : ""}`);
  }
  if (e.kind === "negotiation_conflict" && Array.isArray(e.mus)) parts.push(`MUS: ${(e.mus as string[]).join(", ")}`);
  return parts.join(" · ");
}

/** Every event for a request, with the time spent between them. */
export function EventTimeline({ events }: { events: EventRow[] }) {
  return (
    <ol className="relative space-y-0">
      {events.map((e, i) => {
        const prev = events[i - 1];
        const dt = prev ? (new Date(e.at).getTime() - new Date(prev.at).getTime()) / 1000 : 0;
        const meta = STATUS[e.kind as Status];
        const tone = e.kind === "error" ? "bad" : meta?.tone ?? (e.kind.startsWith("negotiation") ? "warn" : "info");
        return (
          <li key={e.n} className="relative flex gap-3.5 pb-4 pl-1">
            {i < events.length - 1 && <span className="absolute left-[8.5px] top-5 h-full w-px bg-line" />}
            <span className="relative mt-1.5 grid size-[13px] shrink-0 place-items-center rounded-full bg-panel">
              <span className={clsx("size-[7px] rounded-full", TONE_CLASS[tone].dot)} />
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-baseline gap-x-2">
                <p className="text-[13.5px] font-medium">{EVENT_LABEL[e.kind] ?? e.kind}</p>
                <span className="font-mono text-[11px] text-ink-3">{clock(e.at)}</span>
                {dt > 0.05 && <span className="font-mono text-[10.5px] text-ink-3">+{dt.toFixed(2)}s</span>}
              </div>
              {detail(e) && <p className="mt-0.5 break-words text-[12.5px] text-ink-3">{detail(e)}</p>}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

export function ActivityFeed({ events, compact }: { events: EventRow[]; compact?: boolean }) {
  return (
    <div>
      {[...events].reverse().map((e) => {
        const meta = STATUS[e.kind as Status];
        const tone = e.kind === "error" ? "bad" : meta?.tone ?? (e.kind.startsWith("negotiation") ? "warn" : "info");
        return (
          <div key={e.n} className="flex items-center gap-2.5 rounded-md px-3 py-1.5 hover:bg-panel-2/60">
            <span className={clsx("size-1.5 shrink-0 rounded-full", TONE_CLASS[tone].dot)} />
            <span className="min-w-0 flex-1 truncate text-[13px]">
              {EVENT_LABEL[e.kind] ?? e.kind}
              {!compact && detail(e) && <span className="text-ink-3"> · {detail(e)}</span>}
            </span>
            {e.case && (
              <Link to={`/requests/${e.case}`} className="shrink-0 font-mono text-[11px] text-ink-3 hover:text-brand">
                {e.case}
              </Link>
            )}
            <span className="w-14 shrink-0 text-right text-[11px] text-ink-3">{ago(e.at)}</span>
          </div>
        );
      })}
      {!events.length && (
        <p className="flex items-center gap-2 px-3 py-6 text-[13px] text-ink-3">
          <LiveDot tone="muted" /> Waiting for activity…
        </p>
      )}
    </div>
  );
}
