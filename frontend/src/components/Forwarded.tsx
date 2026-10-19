import clsx from "clsx";
import { ArrowRight, Check, Mail, Send, Wrench } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { ago } from "../lib/meta";
import type { CaseDetail, CaseSummary } from "../lib/types";
import { Badge, Button, Card, CardHeader } from "./ui";

/** A message forwarded to the timetable office: nothing in it reaches the solver by itself, so a person
 * answers it, or acts on it: the office's own request made of it goes through the whole pipeline. */
export function ForwardedPanel({ c, onHandled }: { c: CaseDetail; onHandled?: (msg: string) => void }) {
  const h = c.answer;
  // a refused request: the refusal stands, but the office may still answer it or make the change itself
  const refused = c.status === "refused" && c.forwarded_to !== "coordinator";
  return (
    <Card className="overflow-hidden">
      <div className={clsx("flex items-start gap-4 px-5 py-4", !h && "bg-info/[0.06]", c.can_handle && "border-b border-line")}>
        <span className={clsx("grid size-10 shrink-0 place-items-center rounded-full", h ? "bg-ok/12 text-ok" : "bg-info/15 text-info")}>
          {h ? <Check size={18} /> : <Mail size={18} />}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-[15px] font-semibold">
              {h
                ? `Answered by ${h.by_name}`
                : refused
                  ? "Refused: you can still answer it, or make the change yourself"
                  : c.can_handle
                    ? "Passed to you for a reply"
                    : "Waiting for a reply from the timetable office"}
            </h3>
            {h && <Badge tone="muted">{ago(h.at)}</Badge>}
          </div>
          <p className="mt-1 max-w-3xl text-[13.5px] leading-relaxed text-ink-2">
            {h
              ? `“${h.note}”`
              : refused
                ? `The system refused it (${c.parse?.refusal ?? "no authority over it"}), so nothing reached the timetable. If it should happen, make it your request: it goes through the policy check, the solver and your approval like any other. Its sender is told either way.`
                : "This is not a change the system can make by itself, so a person answers it or acts on it."}
          </p>
          {h?.acting && (
            <Link to={`/requests/${h.acting}`} className="mt-1.5 inline-flex items-center gap-1 text-[13px] font-medium text-brand">
              Follow request {h.acting} <ArrowRight size={14} />
            </Link>
          )}
        </div>
      </div>
      {c.can_handle && <HandleForm c={c} onHandled={onHandled} />}
    </Card>
  );
}

function HandleForm({ c, onHandled }: { c: CaseDetail; onHandled?: (msg: string) => void }) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const send = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.handle(c.id, note.trim());
      setNote("");
      onHandled?.(`Reply sent to ${c.sender_name}.`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  // the office's own request made of the message: what is written in the box, else the message itself
  const act = async () => {
    setBusy(true);
    setError(null);
    try {
      const { id } = await api.act(c.id, note.trim());
      setNote("");
      onHandled?.(`Acting on it as request ${id}; ${c.sender_name} has been told.`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="bg-panel-2/40 p-5">
      <textarea
        value={note}
        onChange={(e) => setNote(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && (e.ctrlKey || e.metaKey) && note.trim() && void send()}
        rows={3}
        placeholder={`Your reply to ${c.sender_name}, or the change to make of it (e.g. "Move the Algorithms lecture of G-02 on Tuesday to the afternoon")`}
        className="w-full resize-none rounded-md border border-line bg-panel px-3 py-2 text-[14px] outline-none focus:border-brand/60"
      />
      <div className="mt-2 flex items-center justify-end gap-3">
        {error && <p className="mr-auto text-[13px] text-bad">{error}</p>}
        <Button icon={Wrench} loading={busy} onClick={act}>
          {note.trim() ? "Make this change" : "Act on the message"}
        </Button>
        <Button variant="primary" icon={Send} loading={busy} disabled={!note.trim()} onClick={send}>
          Send reply
        </Button>
      </div>
    </div>
  );
}

/** The office's list of forwarded messages still waiting for its reply (on the Approvals page). */
export function ForwardedList({ cases }: { cases: CaseSummary[] }) {
  return (
    <Card className="mb-8 overflow-hidden border-info/30">
      <CardHeader
        icon={Mail}
        title={`Messages passed to you (${cases.length})`}
        subtitle="Requests the system cannot act on by itself. Reply to each one, or act on it as your own request."
      />
      <ul className="divide-y divide-line">
        {cases.map((c) => (
          <li key={c.id}>
            <Link to={`/requests/${c.id}`} className="group flex items-center gap-4 px-5 py-3.5 hover:bg-panel-2/60">
              <span className="grid size-8 shrink-0 place-items-center rounded-full bg-info/12 text-info">
                <Mail size={15} />
              </span>
              <div className="min-w-0 flex-1">
                <p className="truncate text-[14px]">“{c.text}”</p>
                <p className="mt-0.5 text-[12.5px] text-ink-3">
                  {c.sender_name} · {ago(c.received_at)} · <span className="font-mono">{c.id}</span>
                </p>
              </div>
              <span className="flex shrink-0 items-center gap-1 text-[13px] font-medium text-brand">
                Reply <ArrowRight size={14} className="transition-transform group-hover:translate-x-0.5" />
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </Card>
  );
}
