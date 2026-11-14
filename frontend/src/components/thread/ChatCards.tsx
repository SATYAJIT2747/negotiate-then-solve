import clsx from "clsx";
import { ArrowRight, BookOpen, CalendarClock, CalendarSearch, HelpCircle, Info, MessageCircleQuestion, Send } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../../lib/auth";
import { effectiveStatus } from "../../lib/requests";
import { firstSlots, slotsOf, type Slot } from "../../lib/slots";
import type { CalendarView, CaseSummary, ChatMessage, ConfirmAs, InstanceView } from "../../lib/types";
import { useWorkspace, type LookupMessage } from "../../lib/workspace";
import { PLAIN_STATUS, STATUS } from "../../lib/meta";
import { StudyAnswerCheck } from "../StudyBanner";
import { Button, LiveDot, StatusBadge } from "../ui";
import { CardFrame, Tag, YouBubble } from "./Bubble";

/** The assistant chat's messages, one card per kind (``CHAT_CARDS``): a new kind of message is a new entry
 * there, and every chat that shows ChatMessages draws it the same way. */

export interface ChatContext {
  inst: InstanceView | null;
  cal: CalendarView | null;
  /** an answer is being sent: the cards' buttons wait */
  busy: boolean;
  onConfirm: (n: number, as: ConfirmAs) => void;
  /** a request card's "it's a question": take the request back and answer it as a question */
  onAsQuestion: (n: number) => void;
  /** the person's requests by number, for the status on a request's card */
  cases: Map<string, CaseSummary>;
}

type Of<K extends ChatMessage["kind"]> = Extract<ChatMessage, { kind: K }>;
type Renderers = { [K in ChatMessage["kind"]]: (m: Of<K>, x: ChatContext) => ReactNode };

export const CHAT_CARDS: Renderers = {
  text: (m) => <YouBubble compact text={m.text} note={m.read_as && `Read as: ${m.read_as}`} />,
  lookup: (m, x) => (
    <>
      <LookupCard m={m} inst={x.inst} cal={x.cal} />
      <StudyAnswerCheck n={m.n} step="timetable" answer={m.answer ?? ""} />
    </>
  ),
  rule: (m) => (
    <>
      <RuleCard answer={m.answer} rules={m.rules} />
      <StudyAnswerCheck n={m.n} step="rules" answer={m.answer} />
    </>
  ),
  confirm: (m, x) => <ConfirmCard n={m.n} text={m.part ? m.text : undefined} answered={m.answered} busy={x.busy} onConfirm={x.onConfirm} />,
  request: (m, x) => (
    <RequestCard id={m.case} summary={x.cases.get(m.case)} cases={x.cases} part={m.part} reread={!!m.reread} busy={x.busy} onAsQuestion={() => x.onAsQuestion(m.n)} />
  ),
  other: (m) => <CardFrame label="Assistant">{m.text}</CardFrame>,
  change: (m) => <ChangeNotice text={m.text} weeks={m.weeks} from={m.from_name} />,
};

/** One chat message, drawn by its kind. */
export function ChatMessageView({ m, x }: { m: ChatMessage; x: ChatContext }) {
  const draw = CHAT_CARDS[m.kind] as (m: ChatMessage, x: ChatContext) => ReactNode;
  return <>{draw(m, x)}</>;
}

const SHOWN_SLOTS = 4; // more than this: "Show on the timetable" shows them all

const TONE: Record<Slot["tone"], string> = {
  free: "border-ok/40 bg-ok/8",
  busy: "border-line bg-panel text-ink-3",
  class: "border-brand/40 bg-brand/6",
  cancelled: "border-bad/40 bg-bad/5 line-through decoration-bad/60",
  moved: "border-warn/40 bg-warn/5",
  extra: "border-brand/40 bg-brand/6",
};

/** A timetable answer: the words, then at most 4 slots grouped by date, then "Show on the timetable". */
export function LookupCard({ m, inst, cal }: { m: LookupMessage; inst: InstanceView | null; cal: CalendarView | null }) {
  const { showAnswer } = useWorkspace();
  const answer = m.answer ?? "";
  const groups = m.view && inst && cal ? slotsOf(m.view, inst, cal) : [];
  const { shown, more } = firstSlots(groups, SHOWN_SLOTS);
  // "Dr. Iyer teaches in week 13: Compiler Design practical, Monday 2–4 pm (Lab 3); …": with the slots below,
  // only the lead-in ("Dr. Iyer teaches in week 13."); the slots carry the details
  const lead = shown.length && answer.includes(": ") ? `${answer.slice(0, answer.indexOf(": "))}.` : answer;
  return (
    <CardFrame label="Lookup" icon={CalendarSearch}>
      <p>{lead}</p>
      {shown.length > 0 && (
        <div className="mt-2 space-y-2">
          {shown.map((g) => (
            <div key={g.day}>
              <p className="mb-1 text-[11px] font-medium text-ink-3">{g.date}</p>
              <div className="flex flex-wrap gap-1.5">
                {g.slots.map((s, i) => (
                  <div key={i} className={clsx("rounded-md border px-2 py-0.5 text-[12px]", TONE[s.tone])}>
                    <span className="font-medium">
                      {s.title ? `${s.title} · ` : ""}
                      {s.time}
                    </span>
                    {s.note && <span className="ml-1.5 text-ink-3">{s.note}</span>}
                    {s.sub && <p className="text-[11px] text-ink-3">{s.sub}</p>}
                  </div>
                ))}
              </div>
            </div>
          ))}
          {more > 0 && <p className="text-[11.5px] text-ink-3">and {more} more</p>}
          <button onClick={() => showAnswer(m)} className="inline-flex items-center gap-1 text-[12px] font-medium text-brand hover:underline">
            Show on the timetable <ArrowRight size={13} />
          </button>
        </div>
      )}
    </CardFrame>
  );
}

/** A rule question's answer and every rule it cites; a cited rule opens beside the chat. */
export function RuleCard({ answer, rules }: { answer: string; rules: { id: string; title: string }[] }) {
  const { openRule } = useWorkspace();
  return (
    <CardFrame label="Rule" icon={BookOpen}>
      <p className="whitespace-pre-wrap">{answer}</p>
      {rules.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {rules.map((r) => (
            <Tag key={r.id} title="Read this rule" onClick={() => openRule(r.id)}>
              <span className="font-mono">{r.id}</span> · {r.title}
            </Tag>
          ))}
        </div>
      )}
    </CardFrame>
  );
}

const ANSWERED: Record<ConfirmAs, string> = {
  request: "Sent as a request.",
  question: "Read as a question.",
  cancel: "Not sent.",
};

/** "Send this as a request?": nothing is sent until the person chooses. ``text``: the request part of a message
 * whose questions were answered above it (shown, and no "It's a question": they already were). */
export function ConfirmCard({ n, text, answered, busy, onConfirm }: {
  n: number;
  text?: string;
  answered?: ConfirmAs;
  busy: boolean;
  onConfirm: (n: number, as: ConfirmAs) => void;
}) {
  return (
    <CardFrame label="Request?" icon={HelpCircle} tone={answered ? undefined : "warn"}>
      <p>{text ? "Part of your message looks like a request. Send it to the timetable office?" : "This looks like a request. Send it to the timetable office?"}</p>
      {text && <p className="mt-1.5 border-l-2 border-line pl-2 text-[12px] text-ink-2">{text}</p>}
      {answered ? (
        <p className="mt-1.5 text-[11.5px] text-ink-3">{ANSWERED[answered]}</p>
      ) : (
        <div className="mt-2 flex flex-wrap gap-1.5">
          <Button variant="primary" size="sm" disabled={busy} onClick={() => onConfirm(n, "request")}>
            Send as a request
          </Button>
          {!text && (
            <Button size="sm" disabled={busy} onClick={() => onConfirm(n, "question")}>
              It's a question
            </Button>
          )}
          <Button variant="ghost" size="sm" disabled={busy} onClick={() => onConfirm(n, "cancel")}>
            Cancel
          </Button>
        </div>
      )}
    </CardFrame>
  );
}

/** A change to this person's timetable that someone else's request made, once published: what it means for them
 * (the classes by their times, why), from whom, and the way to the week on the timetable beside the chat. */
function ChangeNotice({ text, weeks, from }: { text: string; weeks: number[]; from?: string }) {
  const { goToWeek } = useWorkspace();
  return (
    <CardFrame label={from ? `Timetable change · from ${from}'s request` : "Timetable change"} icon={CalendarClock}>
      <div className="rounded-lg border border-info/30 bg-info/5 px-3 py-2">
        <p className="leading-relaxed">{text}</p>
        {weeks.length > 0 && (
          <button onClick={() => goToWeek(weeks[0])} className="mt-1.5 inline-flex items-center gap-1 text-[12px] font-medium text-brand hover:underline">
            Show week {weeks[0]} on the timetable <ArrowRight size={13} />
          </button>
        )}
      </div>
    </CardFrame>
  );
}

/** The requests a message was split into, one small block each: what it asks, in full (the chat is narrow, so
 * no number or badge beside it: the number is in its tooltip); where it stands only when the parts differ (the
 * card's own status says it when they agree). */
function Parts({ parts, plain }: { parts: CaseSummary[]; plain: boolean }) {
  const same = new Set(parts.map((p) => p.status)).size === 1;
  return (
    <div className="mt-2 space-y-1.5">
      <p className="text-[11px] text-ink-3">Sent as {parts.length} requests</p>
      {parts.map((p) => {
        const st = (plain ? PLAIN_STATUS[p.status] : STATUS[p.status]) ?? STATUS.received;
        return (
          <Link
            key={p.id}
            to={`/requests/${p.id}`}
            title={`${p.id} · open this request`}
            className="group flex items-start gap-2 rounded-md border border-line bg-panel-2/40 px-2.5 py-1.5 transition-colors hover:border-brand/50"
          >
            <span className="min-w-0 flex-1">
              <span className="block text-[12.5px] leading-snug text-ink">{p.text}</span>
              {!same && (
                <span className="mt-0.5 flex items-center gap-1.5 text-[11px] text-ink-3">
                  <LiveDot tone={st.tone} pulse={p.running} /> {st.label}
                </span>
              )}
            </span>
            <ArrowRight size={12} className="mt-1 shrink-0 text-ink-3 transition-transform group-hover:translate-x-0.5 group-hover:text-brand" />
          </Link>
        );
      })}
    </div>
  );
}

/** Where switching a request to a question no longer makes sense: it was answered, or is over without a change. */
const SETTLED = new Set(["answered", "withdrawn", "refused", "denied", "published"]);

/** A request the chat sent at once, followed live: what it asks, how far it has got, whether it waits on the
 * person; it opens as a conversation, and its details (how the system handled it) open beside the chat. If it was
 * meant as a question, the "?" in its corner takes it back and answers it instead (``onAsQuestion``). */
export function RequestCard({ id, summary, cases, part, reread, busy, onAsQuestion }: {
  id: string;
  summary?: CaseSummary;
  /** every request of the person by number: the parts of a split message */
  cases?: Map<string, CaseSummary>;
  part?: boolean;
  reread?: boolean;
  busy?: boolean;
  onAsQuestion?: () => void;
}) {
  const { openDetails } = useWorkspace();
  const { can } = useAuth(); // the office sees the pipeline's step names, everyone else plain statuses
  const parts = (summary?.parts ?? []).map((p) => cases?.get(p)).filter((p): p is CaseSummary => !!p);
  const status = summary && cases ? effectiveStatus(summary, cases) : summary?.status;
  const waits = !reread && !!summary && (summary.waits_on_you || status === "clarification_requested" || parts.some((p) => p.waits_on_you));
  const switchable = !reread && !!onAsQuestion && !parts.length && !(status && SETTLED.has(status));
  return (
    <CardFrame
      label={part ? "Request · sent from your message" : "Request · sent"}
      icon={Send}
      tone={waits ? "warn" : undefined}
      action={
        switchable && (
          <button
            onClick={onAsQuestion}
            disabled={busy}
            title="Meant as a question? Take this request back and answer it instead"
            aria-label="It's a question: take the request back and answer it"
            className="flex items-center gap-1 rounded px-1 py-0.5 text-ink-3 hover:bg-panel-2 hover:text-brand disabled:opacity-40"
          >
            <MessageCircleQuestion size={13} />
          </button>
        )
      }
    >
      <div className={clsx(!waits && "rounded-lg border border-line px-3 py-2")}>
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-[12px] text-ink-2">{id}</span>
          {summary && status && <StatusBadge status={status} live={summary.running || parts.some((p) => p.running)} plain={!can("approve")} />}
        </div>
        {summary?.text && <p className="mt-1 line-clamp-2 text-[12.5px] text-ink-2">{summary.text}</p>}
        {parts.length > 1 && <Parts parts={parts} plain={!can("approve")} />}
        {reread && <p className="mt-1 text-[12px] text-ink-3">Taken back and answered as a question below.</p>}
        {waits && <p className="mt-1 text-[12px] font-medium text-warn">Waiting on your answer</p>}
        <div className="mt-1.5 flex flex-wrap items-center gap-3">
          <Link to={`/requests/${id}`} className="inline-flex items-center gap-1 text-[12px] font-medium text-brand hover:underline">
            {waits ? "Answer it" : "Open request"} <ArrowRight size={13} />
          </Link>
          <button onClick={() => openDetails(id)} className="inline-flex items-center gap-1 text-[12px] text-ink-3 hover:text-ink">
            <Info size={12} /> Details
          </button>
        </div>
      </div>
    </CardFrame>
  );
}
