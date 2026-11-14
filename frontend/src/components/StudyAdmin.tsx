import clsx from "clsx";
import { Ban, Download, FlaskConical, Plus, RotateCcw, Trash2, Users, X } from "lucide-react";
import { useState } from "react";
import { useApi } from "../lib/hooks";
import { num, pct } from "../lib/meta";
import { study, type H3, type Kappa, type StudyAdmin as Admin } from "../lib/study";
import { Badge, Button, Card, CardHeader, Label, Meter } from "./ui";

/** The coordinator's view of the human study: codes, progress, agreement and the pilot's ratings. */
export function StudyAdmin() {
  const { data, refresh, error } = useApi(() => study.admin(), [], 5000);
  const [codes, setCodes] = useState<string[]>([]);
  const [busy, setBusy] = useState<string | null>(null);

  const add = async (kind: "team" | "pilot") => {
    setBusy(kind);
    try {
      const got = await study.addParticipants(kind, 1);
      setCodes((c) => [...got.codes, ...c]);
      await refresh();
    } finally {
      setBusy(null);
    }
  };
  const exportAll = async () => {
    const blob = new Blob([JSON.stringify(await study.export(), null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `study-export-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <Card>
      <CardHeader
        icon={FlaskConical}
        title="Human study"
        subtitle={<>Judge validation (team) and the anonymised faculty pilot. Participants open <span className="font-mono">{window.location.origin}/study</span> and enter their code.</>}
        action={
          <div className="flex flex-wrap gap-2">
            <Button size="sm" icon={Plus} loading={busy === "team"} onClick={() => add("team")}>Team code</Button>
            <Button size="sm" icon={Plus} loading={busy === "pilot"} onClick={() => add("pilot")}>Pilot code</Button>
            <Button size="sm" variant="ghost" icon={Download} onClick={exportAll} disabled={!data?.participants.length}>Export</Button>
          </div>
        }
      />
      {error && <p className="p-5 text-[13.5px] text-bad">{error.message}</p>}
      {data && (
        <div className="space-y-6 p-5">
          {codes.length > 0 && (
            <p className="rounded-md bg-ok/10 px-3 py-2 text-[13.5px]">
              New code{codes.length > 1 ? "s" : ""}: <span className="font-mono font-semibold">{codes.join(", ")}</span>. Hand it to the participant; it is the only link to their answers.
            </p>
          )}
          {!data.items ? (
            <p className="text-[13.5px] text-ink-2">
              No study material yet. Build it from a negotiation report with explanations recorded:{" "}
              <code className="font-mono">uv run python -m evaluation.study build runs/negotiation-study-source.json</code>
            </p>
          ) : (
            <p className="text-[13px] text-ink-3">
              {data.items.claims} claims · {data.items.replies} replies · {data.items.rules ?? 0} rule answers · {data.items.ratings} messages ({data.items.pilot_ratings} in the pilot set) · from{" "}
              <span className="font-mono">{data.items.source}</span> · inbox replies read by {data.reply_parser}
            </p>
          )}
          {data.items && data.analysis && (
            <div className="rounded-md border border-line p-3.5">
              <p className="text-[13px] font-medium">Shared between the team</p>
              <p className="text-[12px] text-ink-3">
                Each item goes to two team raters, rotating through every pair, so with {data.analysis.participants.team || "N"} raters each does about{" "}
                {data.analysis.participants.team > 2 ? `${Math.round(200 / data.analysis.participants.team)}%` : "all"} of a task. Agreement needs both labels.
              </p>
              <div className="mt-2.5 grid gap-3 sm:grid-cols-3">
                {(["ratings", "claims", "replies"] as const).map((t) => {
                  const c = data.analysis!.coverage[t];
                  return (
                    <div key={t}>
                      <div className="flex justify-between text-[12.5px]">
                        <span className="capitalize text-ink-2">{t}</span>
                        <span className="font-mono text-ink-3">
                          {c.double}/{c.items} labelled twice
                        </span>
                      </div>
                      <Meter value={c.double} max={c.items || 1} tone={c.double === c.items && c.items ? "ok" : "brand"} className="mt-1" />
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div>
            <Label>Participants</Label>
            {data.participants.length ? (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[560px] text-[13px]">
                  <thead>
                    <tr className="border-b border-line text-left text-[12px] text-ink-3">
                      <th className="py-2 font-medium">Code</th>
                      <th className="py-2 font-medium">Group</th>
                      <th className="py-2 font-medium">Consent</th>
                      <th className="py-2 font-medium">Ratings</th>
                      <th className="py-2 font-medium">Claims</th>
                      <th className="py-2 font-medium">Replies</th>
                      <th className="py-2" />
                    </tr>
                  </thead>
                  <tbody>
                    {data.participants.map((p) => (
                      <tr key={p.code} className={clsx("border-b border-line/60", p.revoked && "text-ink-3")}>
                        <td className={clsx("py-2 font-mono", p.revoked && "line-through")}>{p.code}</td>
                        <td className="py-2">{p.kind === "pilot" ? `pilot · as ${p.persona_name}` : "team"}</td>
                        <td className="py-2">
                          {p.answers_deleted ? <Badge tone="bad">withdrawn</Badge> : p.revoked ? <Badge tone="warn">revoked</Badge> : p.consented ? <Badge tone="ok">given</Badge> : <Badge>not yet</Badge>}
                        </td>
                        {(["ratings", "claims", "replies"] as const).map((t) => (
                          <td key={t} className="py-2 font-mono text-ink-2">{p.progress[t] && !p.answers_deleted ? `${p.progress[t]!.done}/${p.progress[t]!.total}` : "—"}</td>
                        ))}
                        <td className="py-2 text-right">
                          <ParticipantActions code={p.code} revoked={!!p.revoked} deleted={!!p.answers_deleted} onDone={refresh} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="flex items-center gap-2 text-[13.5px] text-ink-3"><Users size={15} /> No codes issued yet. The brief plans two team raters and 5-10 pilot faculty.</p>
            )}
          </div>

          {data.analysis && <Analysis a={data.analysis} />}

          <ClearAll
            participants={data.participants.length}
            onExport={exportAll}
            onDone={async () => {
              setCodes([]);
              await refresh();
            }}
          />
        </div>
      )}
    </Card>
  );
}

/** Revoke a code (answers kept out of every result, can be restored), or withdraw it (answers deleted for good). */
function ParticipantActions({ code, revoked, deleted, onDone }: { code: string; revoked: boolean; deleted: boolean; onDone: () => Promise<void> }) {
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      setAsking(false);
      await onDone();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  if (deleted) return <span className="text-[12px] text-ink-3">answers deleted</span>;
  if (revoked)
    return (
      <Button size="sm" variant="ghost" icon={RotateCcw} loading={busy} onClick={() => run(() => study.restore(code))}>
        Restore
      </Button>
    );
  if (!asking)
    return (
      <Button size="sm" variant="ghost" icon={Ban} onClick={() => setAsking(true)}>
        Revoke
      </Button>
    );
  return (
    <div className="flex flex-wrap items-center justify-end gap-1.5">
      <span className="text-[12px] text-ink-2">Revoke {code}?</span>
      <Button size="sm" loading={busy} onClick={() => run(() => study.revoke(code, false))} title="The code stops working; their answers are kept but left out of every result">
        Keep answers
      </Button>
      <Button
        size="sm"
        variant="danger"
        disabled={busy}
        onClick={() => window.confirm(`Delete every answer from ${code}? This cannot be undone.`) && run(() => study.revoke(code, true))}
        title="Withdrawal: the code stops working and all their labels and live records are deleted"
      >
        Delete answers
      </Button>
      <Button size="sm" variant="ghost" disabled={busy} onClick={() => setAsking(false)}>
        Cancel
      </Button>
      {error && <span className="w-full text-right text-[12px] text-bad">{error}</span>}
    </div>
  );
}

function KappaList({ title, rows, note }: { title: string; rows: Record<string, Kappa>; note: string }) {
  const entries = Object.entries(rows);
  return (
    <div>
      <p className="text-[13px] font-medium">{title}</p>
      <p className="text-[12px] text-ink-3">{note}</p>
      {entries.length ? (
        <ul className="mt-1.5 space-y-1 text-[13px]">
          {entries.map(([who, k]) => (
            <li key={who} className="flex items-center gap-2">
              <span className="font-mono text-ink-2">{who}</span>
              <span className="ml-auto font-mono">κ {k.kappa === null ? "—" : num(k.kappa, 2)}</span>
              <span className="text-[12px] text-ink-3">n={k.n}</span>
              {k.kappa !== null && <Badge tone={k.kappa >= 0.6 ? "ok" : "warn"}>{k.kappa >= 0.6 ? "≥ 0.6" : "< 0.6"}</Badge>}
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-1.5 text-[13px] text-ink-3">No labels yet.</p>
      )}
    </div>
  );
}

function H3Block({ title, h }: { title: string; h: H3 }) {
  return (
    <div>
      <p className="text-[13px] font-medium">{title}</p>
      <p className="text-[12px] text-ink-3">{h.raters} rater{h.raters === 1 ? "" : "s"}, {h.pairs} paired messages (same scenario, grounded vs free)</p>
      <table className="mt-1.5 w-full text-[13px]">
        <thead>
          <tr className="text-left text-[12px] text-ink-3">
            <th className="font-medium" />
            <th className="font-medium">Grounded</th>
            <th className="font-medium">Free</th>
            <th className="font-medium">Wilcoxon p</th>
          </tr>
        </thead>
        <tbody>
          {(["clarity", "acceptability"] as const).map((k) => (
            <tr key={k}>
              <td className="capitalize text-ink-2">{k}</td>
              <td className="font-mono">{h[k].grounded === null ? "—" : num(h[k].grounded!, 2)}</td>
              <td className="font-mono">{h[k].free === null ? "—" : num(h[k].free!, 2)}</td>
              <td className="font-mono">{h[k].wilcoxon ? num(h[k].wilcoxon!.p, 3) : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Analysis({ a }: { a: NonNullable<Admin["analysis"]> }) {
  const ratingKappa = Object.fromEntries(
    Object.entries(a.ratings.kappa).flatMap(([pair, v]) => [
      [`${pair} · clarity`, { n: v.n, kappa: v.clarity ?? null }],
      [`${pair} · acceptability`, { n: v.n, kappa: v.acceptability ?? null }],
    ]),
  ) as Record<string, Kappa>;
  const u = a.live.understood;
  return (
    <div className="grid gap-6 border-t border-line pt-5 md:grid-cols-2">
      <KappaList title="Claims: human vs faithfulness verifier" rows={a.claims.human_vs_verifier} note="Cohen's κ per team rater (proposal: report only if ≥ 0.6)" />
      <KappaList title="Claims: human vs human" rows={a.claims.human_vs_human} note="Agreement between the two team raters" />
      <KappaList title="Replies: human vs recorded call" rows={a.replies.human_vs_recorded} note="What the reply does, as labelled vs as the system read it" />
      <KappaList title="Message ratings: rater vs rater" rows={ratingKappa} note="Quadratic-weighted κ; add the LLM judge with evaluation.judge" />
      {a.rules && (
        <>
          <KappaList title="Rule answers: human vs RAG judge" rows={a.rules.human_vs_judge} note="Declines / consistent with the rule / not, as labelled vs as the Gemini judge said" />
          <KappaList title="Rule answers: human vs human" rows={a.rules.human_vs_human} note="Agreement between the two team raters" />
        </>
      )}
      <H3Block title="H3 · pilot faculty ratings" h={a.ratings.h3_pilot} />
      <H3Block title="H3 · team ratings" h={a.ratings.h3_team} />
      <div className="md:col-span-2">
        <p className="text-[13px] font-medium">Live sessions in the portal</p>
        <div className="mt-2 grid gap-4 sm:grid-cols-4">
          <Fig label="Requests checked" v={`${a.live.requests_checked}`} sub={`yes ${u.yes ?? 0} · partly ${u.partly ?? 0} · no ${u.no ?? 0}`} />
          <Fig label="Own-words replies" v={`${a.live.replies}`} sub={`reading confirmed ${pct(a.live.reply_reading_confirmed)}`} />
          <Fig label="Messages rated live" v={`${a.live.message_ratings}`} />
          <Fig label="Live clarity / acceptability" v={`${a.live.clarity === null ? "—" : num(a.live.clarity, 1)} / ${a.live.acceptability === null ? "—" : num(a.live.acceptability, 1)}`} />
          <Fig label="Timetable answers right" v={answered(a.live.timetable_answers)} sub={split(a.live.timetable_answers)} />
          <Fig label="Rule answers right" v={answered(a.live.rules_answers)} sub={split(a.live.rules_answers)} />
          <Fig label="Closing questionnaires" v={`${a.live.survey?.n ?? 0}`} />
          <Fig label="Pilots who finished every step" v={`${Object.values(a.live.steps_done ?? {}).filter((s) => s.length >= 6).length} of ${Object.keys(a.live.steps_done ?? {}).length}`} />
        </div>
      </div>
      {!!a.live.survey?.n && (
        <div className="md:col-span-2">
          <p className="text-[13px] font-medium">Closing questionnaire (mean, 1-5)</p>
          <ul className="mt-2 grid gap-x-6 gap-y-1 text-[13px] sm:grid-cols-2">
            {Object.entries(a.live.survey).filter(([k]) => k !== "n").map(([k, v]) => (
              <li key={k} className="flex justify-between gap-3 border-b border-line/60 py-1">
                <span className="text-ink-2">{SURVEY_LABEL[k] ?? k}</span>
                <span className="font-mono">{v === null ? "—" : num(v as number, 2)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

const SURVEY_LABEL: Record<string, string> = {
  understood: "Understood what I wrote",
  timetable: "Timetable answers correct",
  rules: "Rule answers trustworthy",
  clash: "Clash message clear",
  next: "Knew what would happen next",
  use: "Would use it instead of e-mail",
};

const answered = (c?: Record<string, number>) => {
  const n = Object.values(c ?? {}).reduce((t, x) => t + x, 0);
  return n ? `${Math.round((100 * (c?.yes ?? 0)) / n)}%` : "—";
};
const split = (c?: Record<string, number>) => `yes ${c?.yes ?? 0} · partly ${c?.partly ?? 0} · no ${c?.no ?? 0}`;

function Fig({ label, v, sub }: { label: string; v: string; sub?: string }) {
  return (
    <div>
      <p className="text-[12px] text-ink-3">{label}</p>
      <p className="mt-0.5 font-serif text-xl font-semibold tabular-nums">{v}</p>
      {sub && <p className="text-[12px] text-ink-3">{sub}</p>}
    </div>
  );
}

const CLEAR_PHRASE = "delete all study data";

/** GitHub-style: the coordinator types a phrase before every participant, label and live record is deleted. */
function ClearAll({ participants, onExport, onDone }: { participants: number; onExport: () => Promise<void>; onDone: () => Promise<void> }) {
  const [open, setOpen] = useState(false);
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const close = () => {
    setOpen(false);
    setTyped("");
    setError(null);
  };
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-bad/30 p-3.5">
        <div>
          <p className="text-[13px] font-medium">Start over</p>
          <p className="text-[12px] text-ink-3">Delete every participant code, label and live-session record. The study material stays.</p>
        </div>
        <Button size="sm" variant="danger" icon={Trash2} onClick={() => setOpen(true)} disabled={!participants}>
          Clear all study data
        </Button>
      </div>
      {open && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-ink/40 p-4" onClick={close}>
          <div role="dialog" aria-modal="true" aria-labelledby="clear-title" className="w-full max-w-md rounded-lg border border-line bg-panel shadow-xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between border-b border-line px-5 py-3.5">
              <h3 id="clear-title" className="text-[14.5px] font-semibold">
                Clear all study data?
              </h3>
              <button onClick={close} className="grid size-7 place-items-center rounded-md text-ink-3 hover:bg-panel-2" aria-label="Close">
                <X size={15} />
              </button>
            </div>
            <div className="space-y-3 px-5 py-4 text-[13.5px]">
              <p className="rounded-md bg-bad/10 px-3 py-2 text-bad">This cannot be undone.</p>
              <p className="text-ink-2">
                All {participants} participant code{participants === 1 ? "" : "s"}, every label and rating they gave, and every live-session record will be deleted. Codes start again
                from T-1 and P-01. The 100 claims, 100 replies and 110 messages stay.
              </p>
              <button onClick={() => void onExport()} className="text-[13px] font-medium text-brand hover:underline">
                Export everything first
              </button>
              <label className="block">
                <span className="text-ink-2">
                  To confirm, type <span className="rounded bg-panel-2 px-1.5 py-0.5 font-mono text-[12.5px] text-ink">{CLEAR_PHRASE}</span> below
                </span>
                <input
                  autoFocus
                  value={typed}
                  onChange={(e) => setTyped(e.target.value)}
                  className="mt-2 h-9 w-full rounded-md border border-line bg-panel px-3 font-mono text-[13.5px] outline-none focus:border-bad/60"
                />
              </label>
              {error && <p className="text-[12.5px] text-bad">{error}</p>}
            </div>
            <div className="border-t border-line px-5 py-3.5">
              <Button
                variant="danger"
                className="w-full"
                disabled={typed.trim().toLowerCase() !== CLEAR_PHRASE}
                loading={busy}
                onClick={async () => {
                  setBusy(true);
                  setError(null);
                  try {
                    await study.clear(typed);
                    close();
                    await onDone();
                  } catch (e) {
                    setError((e as Error).message);
                  } finally {
                    setBusy(false);
                  }
                }}
              >
                I understand, delete all study data
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
