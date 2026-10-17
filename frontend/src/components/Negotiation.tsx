import clsx from "clsx";
import { ArrowRight, BadgeCheck, Check, CircleSlash, Clock, MapPin, ShieldCheck } from "lucide-react";
import { pct } from "../lib/meta";
import type { ConstraintView, DiffRow, MessageView, OfferView } from "../lib/types";
import { Avatar, Badge, Mark, TierBadge } from "./ui";

export function ConstraintCard({ c, highlight, index = 0 }: { c: ConstraintView; highlight?: boolean; index?: number }) {
  return (
    <div data-index={index} className={clsx("rounded-md border p-3.5", highlight ? "border-bad/35 bg-bad/[0.04]" : "border-line bg-panel")}>
      <div className="flex flex-wrap items-center gap-2">
        <TierBadge tier={c.tier} />
        <Badge tone={c.hard ? "bad" : "muted"}>{c.hard ? "required" : "if possible"}</Badge>
      </div>
      <p className="mt-2 text-[13.5px] leading-relaxed">{c.text}</p>
      {c.tier_who && <p className="mt-1 text-[12.5px] text-ink-3">Who can change it: {c.tier_who.charAt(0).toLowerCase() + c.tier_who.slice(1)}.</p>}
      <div className="mt-2 flex flex-wrap items-center gap-3 text-[11.5px] text-ink-3">
        {c.owner && <span>owner: {c.owner_name}</span>}
        {c.source.request && <span>from {c.source.request}</span>}
        <span className="font-mono opacity-70">{c.id}</span>
        {c.source.rule && <span>rule: {c.source.rule}</span>}
        {c.justification !== "none" && (
          <span className="flex items-center gap-1">
            <ShieldCheck size={12} /> justification {c.justification} (reason kept private)
          </span>
        )}
      </div>
    </div>
  );
}

const norm = (t: string) => t.toLowerCase().replace(/\s+/g, " ").replace(/[.\s]+$/, "").trim();

/** A grounded explanation: each sentence of the message with the facts it rests on. Where a sentence
 * is its fact word for word (template messages) the fact is not repeated; where the model put it in
 * its own words the fact is shown under it, to compare. Sentences with no fact behind them are struck
 * through; facts the message did not use are listed at the end. */
export function GroundedExplanation({ m }: { m: MessageView }) {
  const supported = m.claims.filter((c) => c.supported).length;
  const byId = Object.fromEntries(m.facts.map((f) => [f.id, f]));
  const used = new Set(m.claims.flatMap((c) => c.facts));
  const unused = m.facts.filter((f) => !used.has(f.id));
  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <Badge tone={m.faithfulness >= 0.999 ? "ok" : "warn"}>
          <BadgeCheck size={12} /> {supported} of {m.claims.length} sentences traced to a fact · {pct(m.faithfulness)}
        </Badge>
        <Badge tone="muted">{m.explanation_mode === "template" ? "written from a template" : m.explanation_mode}</Badge>
        {m.leaks.length > 0 && <Badge tone="bad">leak: {m.leaks.join(", ")}</Badge>}
      </div>
      <ol className="space-y-1">
        {m.claims.map((c, i) => {
          const facts = c.facts.map((id) => byId[id]).filter(Boolean);
          const reworded = facts.filter((f) => !norm(c.text).includes(norm(f.text)) && !norm(f.text).includes(norm(c.text)));
          return (
            <li key={i} className="flex gap-2.5 rounded-md px-2 py-1.5 text-[13px] leading-relaxed">
              {c.supported ? <Check size={14} className="mt-1 shrink-0 text-ok" /> : <CircleSlash size={14} className="mt-1 shrink-0 text-bad" />}
              <div className="min-w-0">
                <p className={clsx(!c.supported && "text-ink-3 line-through decoration-bad/60")}>
                  {c.text}
                  {c.facts.map((f) => (
                    <span key={f} className="ml-1.5 rounded bg-panel-2 px-1 align-middle font-mono text-[10px] text-ink-3">
                      {f}
                    </span>
                  ))}
                </p>
                {!c.supported && <p className="text-[12px] text-bad">No fact behind this sentence.</p>}
                {reworded.map((f) => (
                  <p key={f.id} className="mt-0.5 border-l-2 border-line pl-2 text-[12px] text-ink-3">
                    fact: {f.text}
                  </p>
                ))}
              </div>
            </li>
          );
        })}
      </ol>
      {unused.length > 0 && (
        <div className="mt-3 border-t border-line pt-2">
          <p className="mb-1 text-[11.5px] uppercase tracking-wider text-ink-3">Available but not used in the message</p>
          <ul className="space-y-0.5">
            {unused.map((f) => (
              <li key={f.id} className="text-[12px] text-ink-3">
                <span className="mr-1.5 font-mono text-[10.5px]">{f.id}</span>
                {f.text}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export function OfferCard({
  offer,
  selected,
  onSelect,
  accepted,
}: {
  offer: OfferView;
  selected?: boolean;
  onSelect?: () => void;
  accepted?: boolean;
}) {
  const on = selected || accepted;
  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={!onSelect}
      className={clsx(
        "w-full rounded-md border p-3.5 text-left transition-colors",
        on ? "border-brand bg-brand/[0.06]" : "border-line bg-panel",
        onSelect ? "hover:border-brand/60" : "cursor-default",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-2 text-[13px] font-semibold">
          <span className={clsx("grid size-5 place-items-center rounded-full border text-[11px]", on ? "border-brand bg-brand text-panel" : "border-line text-ink-2")}>
            {accepted ? <Check size={11} strokeWidth={3} /> : offer.key}
          </span>
          Option {offer.key}
        </span>
        {offer.verified ? (
          <span className="flex items-center gap-1 text-[11.5px] text-ok">
            <ShieldCheck size={12} /> checked by solver
          </span>
        ) : (
          <span className="text-[11.5px] text-warn">invented by the LLM</span>
        )}
      </div>
      <div className="mt-2.5 space-y-2">
        {offer.placements.map((p) => (
          <div key={p.session}>
            <p className="text-[13px] font-medium">{p.session_name.replace(/^the /, "")}</p>
            <p className="mt-0.5 flex flex-wrap items-center gap-x-3 text-[12.5px] text-ink-2">
              <span className="flex items-center gap-1">
                <Clock size={12} className="text-ink-3" /> {p.day}
                {p.date && ` ${new Date(`${p.date}T00:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short" })}`} {p.time}
              </span>
              {p.date && p.weeks && p.weeks > 1 && <span className="text-ink-3">from week {p.week}, {p.weeks} weeks</span>}
              <span className="flex items-center gap-1">
                <MapPin size={12} className="text-ink-3" /> {p.room}
              </span>
            </p>
          </div>
        ))}
      </div>
      <p className="mt-2.5 border-t border-line pt-2 text-[11.5px] text-ink-3">
        {offer.moved === 0
          ? "Nothing else moves"
          : `${offer.moved} other class${offer.moved === 1 ? " moves" : "es move"}`}
      </p>
    </button>
  );
}

export function MessageBubble({ m, reply }: { m: MessageView; reply?: { decision: string | null; choice?: string; text?: string } }) {
  return (
    <div className="space-y-3">
      <div className="flex gap-3">
        <div className="grid size-8 shrink-0 place-items-center rounded-full bg-panel-2">
          <Mark size={16} />
        </div>
        <div className="min-w-0 flex-1 rounded-lg border border-line bg-panel p-4">
          <p className="mb-2 text-xs text-ink-3">
            {m.explanation_mode === "choice" ? "Choice of time" : `Round ${m.round}`} · to{" "}
            <span className="font-medium text-ink-2">{m.to_name}</span>
          </p>
          <p className="whitespace-pre-line text-[13.5px] leading-relaxed">{m.text.split("\n\nOptions:")[0]}</p>
          <div className="mt-3 grid gap-2 sm:grid-cols-3">
            {m.offers.map((o) => (
              <OfferCard key={o.key} offer={o} accepted={reply?.decision === "accept" && reply.choice === o.key} />
            ))}
          </div>
        </div>
      </div>
      {reply && (
        <div className="flex justify-end gap-3">
          <div className="max-w-[80%] rounded-lg border border-line bg-panel-2 px-4 py-3">
            <p className="text-xs text-ink-3">{m.to_name}</p>
            <p className="mt-1 text-[13.5px]">
              {reply.text || (reply.decision === "no_reply" ? "No reply before the deadline." : reply.decision)}
            </p>
            <p className="mt-1.5">
              <Badge tone={reply.decision === "accept" ? "ok" : reply.decision === "counter" ? "warn" : "bad"}>
                {reply.decision === "accept" ? `accepted ${reply.choice}` : reply.decision}
              </Badge>
            </p>
          </div>
          <Avatar name={m.to_name} id={m.to} />
        </div>
      )}
    </div>
  );
}

export function DiffTable({ rows }: { rows: DiffRow[] }) {
  if (!rows.length) {
    return (
      <p className="flex items-center gap-2 py-4 text-sm text-ink-3">
        <CircleSlash size={15} /> No class moves.
      </p>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[560px] text-[13px]">
        <thead>
          <tr className="border-b border-line text-left text-[12px] text-ink-3">
            <th className="py-2 pr-3 font-medium">Class</th>
            <th className="py-2 pr-3 font-medium">Teacher</th>
            <th className="py-2 pr-3 font-medium">Before</th>
            <th className="py-2 pr-3" />
            <th className="py-2 font-medium">After</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.session} data-index={i} className="border-b border-line/60">
              <td className="py-2.5 pr-3">
                <p className="font-medium">{r.session_name.replace(/^the /, "")}</p>
              </td>
              <td className="py-2.5 pr-3 text-ink-2">{r.faculty_name}</td>
              <td className={clsx("py-2.5 pr-3 text-ink-3", r.before && "line-through decoration-bad/50")}>
                {r.before ? `${r.before.day} ${r.before.time} · ${r.before.room}` : r.extra ? "New" : "—"}
              </td>
              <td className="py-2.5 pr-3 text-ink-3">
                <ArrowRight size={14} />
              </td>
              <td className={clsx("py-2.5 font-medium", r.after || !r.before ? "text-ok" : "text-bad")}>
                {r.after ? `${r.after.day} ${r.after.time} · ${r.after.room}` : r.before ? "Cancelled this week (make-up owed)" : "—"}
                {r.extra && !r.before && <span className="ml-2 text-[12px] font-normal text-ink-3">extra class, this week only</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
