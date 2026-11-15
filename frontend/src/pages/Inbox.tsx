import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowRight, BadgeCheck, BellRing, Gavel, MessageSquareReply, Bot, CalendarClock, CalendarDays, Check, CheckCheck, Clock3, FileText, Inbox as InboxIcon, MessageSquareDashed, MessagesSquare, ShieldAlert, Undo2, X } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ClarifyForm } from "../components/Clarify";
import { EscalationPanel } from "../components/Escalation";
import { ForwardedPanel } from "../components/Forwarded";
import { ApprovalCard } from "./Approvals";
import { GroundedExplanation, OfferCard, SwapDetails } from "../components/Negotiation";
import { Badge, Button, Card, CardHeader, Collapse, EmptyState, LiveDot, PageHeader, Skeleton, Tabs, Toast } from "../components/ui";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useApi } from "../lib/hooks";
import { ago, PLAIN_STATUS } from "../lib/meta";
import { study as studyApi } from "../lib/study";
import type { FeedItem, InboxItem, Status } from "../lib/types";
import { useStudySession } from "../components/StudyBanner";
import { RatingForm } from "../components/Rating";
import { Withdraw } from "../components/RequestThread";

const KIND_ICON = { negotiation: MessagesSquare, change: CalendarClock, request: FileText, action: BellRing, alert: ShieldAlert } as const;
const KIND_LABEL = { negotiation: "Negotiation", change: "Timetable change", request: "Your request", action: "Waiting on you", alert: "Refused request" } as const;
/** What waits on the person, in a word or two: the row's tag and icon say what kind of item it is, so the
 * row's main line can be the message itself (the heading, "A change is waiting for your approval", stays
 * as the row's tooltip). */
const ACTION_TAG = { approve: "To approve", reply: "To answer", decide: "To decide" } as const;
const ACTION_ICON = { approve: BadgeCheck, reply: MessageSquareReply, decide: Gavel } as const;
function tagOf(i: FeedItem): string {
  const report = i.title.startsWith("A report");
  if (i.answered) return report ? "Report · answered" : i.kind === "alert" ? "Refused · answered" : "Answered";
  if (i.kind === "action" && report) return "Report to look into";
  if (i.kind === "action") return i.action ? ACTION_TAG[i.action] : "Waiting on you";
  if (i.kind === "alert") return "Refused";
  if (i.kind === "negotiation") return i.title;
  if (i.kind === "request") return i.status && PLAIN_STATUS[i.status as Status] ? `Your request · ${PLAIN_STATUS[i.status as Status].label}` : "Your request";
  return KIND_LABEL[i.kind];
}
/** The row's main line: what the sender wrote for something to act on, else what the item says. */
/** Each kind of message has its own colour, so the list reads at a glance without a heading per row. */
type Hue = "brand" | "violet" | "warn" | "teal" | "ok" | "bad" | "info";
const HUE: Record<Hue, { text: string; bg: string }> = {
  brand: { text: "text-brand", bg: "bg-brand/12" },
  violet: { text: "text-violet", bg: "bg-violet/12" },
  warn: { text: "text-warn", bg: "bg-warn/15" },
  teal: { text: "text-teal", bg: "bg-teal/12" },
  ok: { text: "text-ok", bg: "bg-ok/12" },
  bad: { text: "text-bad", bg: "bg-bad/12" },
  info: { text: "text-info", bg: "bg-info/12" },
};
function hueOf(i: FeedItem): Hue {
  if (i.kind === "action") return i.action === "approve" ? "brand" : i.action === "decide" ? "warn" : "violet";
  return ({ negotiation: "teal", change: "ok", alert: "bad", request: "info" } as const)[i.kind];
}
const lineOf = (i: FeedItem) => ((i.kind === "action" || i.kind === "alert") && i.request) || i.text || i.request || i.title;
const ACTION_LINK = { approve: [null, "Open the request"], reply: [null, "Reply"], decide: [null, "Decide"] } as const;

/** The inbox is also where the office approves, decides and replies: "Needs you" lists what waits on the
 * person, and each such item is acted on here, in the pane beside the list. */
export default function Inbox() {
  const { user } = useAuth();
  const { data, refresh } = useApi(() => api.feed(), [], 2000);
  const { data: inst } = useApi(() => api.instance(), []);
  const { data: auto, refresh: refreshAuto } = useApi(() => api.autopilot(), []);
  // the tab and the open item live in the address (?view=needs&item=forward:R-00011), so a refresh, the back
  // button or a shared link comes back to the same place
  const [params, setParams] = useSearchParams();
  const asked = params.get("view");
  const view: "all" | "needs" | "unread" = asked === "needs" || asked === "unread" ? asked : "all";
  const sel = params.get("item");
  const place = (next: { view?: typeof view; item?: string | null }) =>
    setParams(
      (p) => {
        const q = new URLSearchParams(p);
        if (next.view !== undefined) (next.view === "all" ? q.delete("view") : q.set("view", next.view));
        if (next.item !== undefined) (next.item ? q.set("item", next.item) : q.delete("item"));
        return q;
      },
      { replace: true },
    );
  const setSel = (item: string | null) => place({ item });
  const [toast, setToast] = useState<string | null>(null);
  const teaches = ["hod", "faculty", "guest_faculty"].includes(user?.role ?? "");

  // the unread items read while the Unread tab is open stay in its list until the tab changes: opening one
  // reads it, and it should not vanish from under the reader (nor take the first item's place on its own)
  const [readHere, setReadHere] = useState<Set<string>>(new Set());
  useEffect(() => setReadHere(new Set()), [view]);
  useEffect(() => {
    if (view === "unread" && sel) setReadHere((r) => (r.has(sel) ? r : new Set(r).add(sel)));
  }, [view, sel]);

  const all = data ?? [];
  const unread = (i: (typeof all)[number]) => i.unread && !i.needs_you; // what needs you is counted there
  const items = all.filter((i) => (view === "needs" ? i.needs_you : view === "unread" ? unread(i) || readHere.has(i.id) : true));
  // open the first item, except on Unread, where opening it would read it at once: there the reader picks
  useEffect(() => {
    if (!sel && items.length && view !== "unread") setSel(items[0].id);
  }, [items, sel, view]); // eslint-disable-line react-hooks/exhaustive-deps
  const current = all.find((i) => i.id === sel) ?? null;

  // opening an item reads it
  useEffect(() => {
    if (current && current.unread && !current.needs_you) void api.feedSeen({ [current.id]: current.n }).then(refresh);
  }, [current?.id, current?.n]); // eslint-disable-line react-hooks/exhaustive-deps

  const readAll = async () => {
    await api.feedSeen(Object.fromEntries(all.filter((i) => i.unread && !i.needs_you).map((i) => [i.id, i.n])));
    await refresh();
  };
  const done = async (msg: string) => {
    setToast(msg);
    await refresh();
  };

  return (
    <>
      <PageHeader
        title="Inbox"
        subtitle="Everything the timetable office sends you: changes to your timetable, news of your requests, and messages that need your answer."
        actions={
          <div className="flex items-center gap-4">
            {all.some((i) => i.unread && !i.needs_you) && (
              <Button size="sm" variant="ghost" icon={CheckCheck} onClick={readAll}>
                Mark all read
              </Button>
            )}
            {(teaches || user?.role === "exam_cell") && auto && (
              <button
                onClick={async () => {
                  await api.setAutopilot(user!.id, !auto[user!.id]);
                  await refreshAuto();
                }}
                title="Let the simulator answer your messages automatically"
                className="flex items-center gap-2.5 text-[13px] text-ink-2"
              >
                <Bot size={15} className="text-ink-3" /> Answer automatically
                <span className={clsx("relative h-[18px] w-8 rounded-full transition-colors", auto[user!.id] ? "bg-ok" : "bg-line")}>
                  <span className="absolute top-[3px] size-3 rounded-full bg-panel transition-[left]" style={{ left: auto[user!.id] ? 17 : 3 }} />
                </span>
              </button>
            )}
          </div>
        }
      />
      <div className="grid gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
        <Card className="overflow-hidden">
          <div className="border-b border-line p-3">
            <Tabs
              tabs={[
                { id: "all", label: "All" },
                { id: "needs", label: "Needs you", count: all.filter((i) => i.needs_you).length },
                { id: "unread", label: "Unread", count: all.filter(unread).length },
              ]}
              value={view}
              onChange={(v) => {
                place({ view: v, item: null });
              }}
            />
          </div>
          <div className="max-h-[680px] divide-y divide-line overflow-y-auto">
            {items.map((i) => {
              const Icon = (i.kind === "action" && i.action && ACTION_ICON[i.action]) || KIND_ICON[i.kind];
              const hue = HUE[hueOf(i)];
              return (
                <button
                  key={i.id}
                  onClick={() => setSel(i.id)}
                  title={i.title}
                  className={clsx("relative flex w-full gap-3 px-4 py-3 text-left", sel === i.id ? "bg-panel-2" : "hover:bg-panel-2/50")}
                >
                  {sel === i.id && <span className="absolute inset-y-0 left-0 w-0.5 bg-brand" />}
                  <span
                    aria-label={i.title}
                    className={clsx(
                      "mt-0.5 grid size-[30px] shrink-0 place-items-center rounded-full",
                      hue.bg,
                      hue.text,
                    )}
                  >
                    <Icon size={15} />
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className={clsx("line-clamp-2 text-[13.5px] leading-snug", i.unread ? "font-semibold text-ink" : "text-ink-2")}>{lineOf(i)}</p>
                    <div className="mt-1 flex items-center gap-1.5 text-[12px] text-ink-3">
                      <span className={clsx("shrink-0 font-medium", hue.text)}>{tagOf(i)}</span>
                      {i.from_name && i.kind !== "negotiation" && <span className="min-w-0 truncate">· {i.from_name}</span>}
                      <span className="ml-auto flex shrink-0 items-center gap-1.5">
                        {i.needs_you ? <LiveDot tone="warn" /> : i.unread && <span className="size-2 rounded-full bg-brand" />}
                        {ago(i.at)}
                      </span>
                    </div>
                  </div>
                </button>
              );
            })}
            {!items.length && (
              <EmptyState
                icon={InboxIcon}
                title={view === "needs" ? "Nothing waiting on you" : view === "unread" ? "All read" : "Your inbox is empty"}
                text="Changes to your timetable, replies to your requests and messages that need your answer arrive here."
              />
            )}
          </div>
        </Card>

        <AnimatePresence mode="wait">
          {current && inst ? (
            <motion.div key={current.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.12 }}>
              {current.kind === "negotiation" && current.message ? (
                <Thread
                  item={current.message}
                  mine={current.message.to === user?.id}
                  canSimulate={current.message.to === user?.id || user?.role === "coordinator"}
                  days={inst.calendar.days}
                  slots={inst.slots}
                  onDone={done}
                />
              ) : (current.kind === "action" || (current.kind === "alert" && (current.action || current.answered))) && current.case ? (
                <ActionPane item={current} onDone={done} />
              ) : (
                <Notice item={current} onDone={done} />
              )}
            </motion.div>
          ) : (
            <Card className="hidden place-items-center lg:grid">
              <EmptyState icon={MessageSquareDashed} title="Select a message" />
            </Card>
          )}
        </AnimatePresence>
      </div>
      <Toast message={toast} onDone={() => setToast(null)} />
    </>
  );
}

/** Something waiting on the person, acted on in place: a change to approve, an escalation to decide, a
 * message passed to the office to answer. */
function ActionPane({ item, onDone }: { item: FeedItem; onDone: (msg: string) => Promise<void> }) {
  const { data: c, refresh } = useApi(() => api.case(item.case!), [item.case, item.n]); // n: someone else acted on it
  if (!c) return <Skeleton className="h-64" />;
  const settled = async (msg: string) => {
    await refresh();
    await onDone(msg);
  };
  const body =
    item.action === "approve" && c.status === "awaiting_approval" ? (
      <ApprovalCard c={c} onDone={settled} compact />
    ) : item.action === "decide" && c.status === "escalated" && c.outcome?.escalation ? (
      <EscalationPanel c={c} onDecided={(msg) => void settled(msg)} />
    ) : (item.action === "reply" || item.answered) && (c.forwarded_to || c.status === "refused") ? (
      <ForwardedPanel c={c} onHandled={(msg) => void settled(msg)} />
    ) : (
      <Notice item={item} onDone={onDone} />
    );
  return (
    <div className="space-y-3">
      {body}
      <Link to={`/requests/${item.case}`} className="inline-flex items-center gap-1 text-[13px] text-brand hover:underline">
        Open request {item.case} in full <ArrowRight size={13} />
      </Link>
    </div>
  );
}

/** A notice, a request update or something to act on: what it says, and where to go from it. */
function Notice({ item, onDone }: { item: FeedItem; onDone: (msg: string) => Promise<void> }) {
  const link = item.action ? ACTION_LINK[item.action] : null;
  // news of the person's own request: they may take it back while nothing it asked for is published
  const own = item.kind === "request" && !!item.case;
  const { data: c, refresh } = useApi(() => (own ? api.case(item.case!) : Promise.resolve(null)), [item.case, item.n, own]);
  return (
    <Card>
      <CardHeader
        title={item.title}
        subtitle={
          <span className="flex flex-wrap items-center gap-2">
            <Badge tone={item.needs_you ? "warn" : item.kind === "change" ? "info" : "muted"}>{KIND_LABEL[item.kind]}</Badge>
            {item.from_name && <span>from {item.from_name}</span>}
            <span>· {ago(item.at)}</span>
            {item.version != null && <span>· timetable version {item.version}</span>}
          </span>
        }
      />
      <div className="space-y-5 p-5">
        {item.text && <p className="whitespace-pre-line text-[14px] leading-relaxed">{item.text}</p>}
        {item.request && (
          <div className="rounded-md border border-line bg-panel-2 px-4 py-3 text-[13.5px]">
            <p className="text-[12px] text-ink-3">{item.kind === "request" ? "You wrote" : "The request behind it"}</p>
            <p className="mt-1">“{item.request}”</p>
          </div>
        )}
        {item.status === "clarification_requested" && item.case && item.kind === "request" && (
          <QuestionFor id={item.case} onSent={() => void onDone("Answer sent; your request is being looked at again.")} />
        )}
        <div className="flex flex-wrap gap-2">
          {link && (
            <Link to={link[0] ?? `/requests/${item.case}`}>
              <Button variant="primary" icon={ArrowRight}>
                {link[1]}
              </Button>
            </Link>
          )}
          {item.case && (
            <Link to={`/requests/${item.case}`}>
              <Button variant="ghost">Open request {item.case}</Button>
            </Link>
          )}
          {own && c?.can_withdraw && (
            <span className="ml-auto">
              <Withdraw
                id={c.id}
                onDone={() => {
                  void refresh();
                  void onDone(`You withdrew ${c.id}; the timetable is as it was before it.`);
                }}
              />
            </span>
          )}
          {item.kind === "change" && (
            <Link to="/timetable">
              <Button variant="ghost" icon={CalendarDays}>
                See the timetable
              </Button>
            </Link>
          )}
        </div>
      </div>
    </Card>
  );
}

function Thread({
  item,
  mine,
  canSimulate,
  days,
  slots,
  onDone,
}: {
  item: InboxItem;
  mine: boolean;
  canSimulate: boolean;
  days: string[];
  slots: { slot: number; label: string }[];
  onDone: (msg: string) => Promise<void>;
}) {
  const [choice, setChoice] = useState<string | null>(null);
  const [mode, setMode] = useState<"pick" | "counter">("pick");
  const [cDays, setCDays] = useState<string[]>([]);
  const [from, setFrom] = useState(5);
  const [to, setTo] = useState(7);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const m = item.message;
  const answered = !!item.reply;
  // the sender's own request fits in more than one way: they pick a time, there is nothing to give up
  const choosing = m.explanation_mode === "choice";
  // a colleague asks to swap classes: a yes or no, shown as the two classes before and after
  const swap = m.explanation_mode === "swap" && m.offers.length === 1;
  const asker = swap ? m.offers[0].placements.find((p) => p.faculty !== item.to)?.faculty_name : null;

  const act = async (label: string, fn: () => Promise<unknown>, msg: string) => {
    setBusy(label);
    setError(null);
    try {
      await fn();
      await onDone(msg);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader
          title={
            choosing
              ? mine ? "Choose a time for your request" : `A time for ${item.to_name} to choose`
              : swap
                ? mine ? `${asker ?? "A colleague"} asks to swap classes` : `A swap for ${item.to_name} to agree to`
              : mine ? "A clash with your timetable" : `A clash for ${item.to_name}`
          }
          subtitle={
            <span className="flex flex-wrap items-center gap-2">
              {item.case && (
                <Link to={`/requests/${item.case}`} className="font-mono text-brand hover:underline">
                  {item.case}
                </Link>
              )}
              <span className="flex items-center gap-1">
                <Clock3 size={12} /> reply by {new Date(item.deadline).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              </span>
            </span>
          }
        />
        <div className="space-y-5 p-5">
          {swap ? (
            <div className="max-w-xl space-y-3">
              <p className="text-[14px] leading-relaxed">{m.text.split("\n\n")[0]}</p>
              <SwapDetails offer={m.offers[0]} me={item.to} />
            </div>
          ) : (
            <>
              <p className="whitespace-pre-line text-[14px] leading-relaxed">{m.text.split("\n\nOptions:")[0]}</p>
              <div className="grid gap-3 md:grid-cols-3">
                {m.offers.map((o) => (
                  <OfferCard
                    key={o.key}
                    offer={o}
                    selected={choice === o.key && mode === "pick"}
                    accepted={item.reply?.decision === "accept" && item.reply.choice === o.key}
                    onSelect={mine && !answered ? () => { setChoice(o.key); setMode("pick"); } : undefined}
                  />
                ))}
              </div>
            </>
          )}

          {answered ? (
            <div className="flex items-center gap-2.5 rounded-md border border-line bg-panel-2 px-4 py-3 text-[13.5px]">
              <Check size={16} className="text-ok" />
              <span>
                Answered{item.answered_by === "simulator" ? " by the simulator" : item.answered_by === "deadline" ? " (deadline passed)" : ""}:{" "}
                <span className="font-medium">
                  {item.reply?.decision}
                  {item.reply?.choice ? ` ${item.reply.choice}` : ""}
                </span>
                {item.reply?.text ? ` · “${item.reply.text}”` : ""}
              </span>
            </div>
          ) : (
            <div className="space-y-4">
              {mine && (
                <AnimatePresence>
                  {mode === "counter" && (
                    <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden">
                      <div className="rounded-md border border-line bg-panel-2 p-4">
                        <p className="text-[13px] font-medium">Which days and times would work instead?</p>
                        <div className="mt-3 flex flex-wrap gap-2">
                          {days.map((d) => (
                            <button
                              key={d}
                              onClick={() => setCDays((xs) => (xs.includes(d) ? xs.filter((x) => x !== d) : [...xs, d]))}
                              className={clsx("rounded-md border px-3 py-1 text-[13px]", cDays.includes(d) ? "border-brand bg-brand text-panel" : "border-line bg-panel text-ink-2")}
                            >
                              {d}
                            </button>
                          ))}
                        </div>
                        <div className="mt-3 flex flex-wrap items-center gap-2 text-[13px]">
                          from
                          <select value={from} onChange={(e) => setFrom(+e.target.value)} className="rounded-lg border border-line bg-panel px-2 py-1">
                            {slots.map((s) => (
                              <option key={s.slot} value={s.slot}>
                                {s.label}
                              </option>
                            ))}
                          </select>
                          until the end of
                          <select value={to} onChange={(e) => setTo(+e.target.value)} className="rounded-lg border border-line bg-panel px-2 py-1">
                            {slots.map((s) => (
                              <option key={s.slot} value={s.slot}>
                                {s.label}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              )}
              <div className="flex flex-wrap items-center gap-2">
                {mine && swap && (
                  <>
                    <Button
                      variant="primary"
                      icon={Check}
                      loading={busy === "accept"}
                      onClick={() =>
                        act(
                          "accept",
                          () => api.reply(item.id, { decision: "accept", choice: "A", text: "I agree to the swap." }),
                          "You agreed. The swap goes to the timetable coordinator for approval.",
                        )
                      }
                    >
                      Agree to swap
                    </Button>
                    <Button
                      variant="danger"
                      icon={X}
                      loading={busy === "reject"}
                      onClick={() =>
                        act("reject", () => api.reply(item.id, { decision: "reject", text: "I don't agree to the swap." }), "Declined. Your colleague is told the swap was not made.")
                      }
                    >
                      Decline
                    </Button>
                  </>
                )}
                {mine && !swap && (
                  <>
                    <Button
                      variant="primary"
                      icon={Check}
                      disabled={!choice || mode !== "pick"}
                      loading={busy === "accept"}
                      onClick={() =>
                        act(
                          "accept",
                          () => api.reply(item.id, { decision: "accept", choice: choice!, text: `Option ${choice} works for me.` }),
                          choosing ? `Chose option ${choice}. It goes to the timetable office for approval.` : `Accepted option ${choice}. The solver re-solves now.`,
                        )
                      }
                    >
                      {choice ? `${choosing ? "Choose" : "Accept"} option ${choice}` : "Pick an option"}
                    </Button>
                    {choosing ? null : mode === "counter" ? (
                      <Button
                        icon={Undo2}
                        disabled={!cDays.length || to < from}
                        loading={busy === "counter"}
                        onClick={() =>
                          act(
                            "counter",
                            () =>
                              api.reply(item.id, {
                                decision: "counter",
                                counter_days: cDays,
                                counter_slots: Array.from({ length: to - from + 1 }, (_, i) => from + i),
                                text: `${cDays.join(", ")} from ${slots[from].label} would work.`,
                              }),
                            "Counter-offer sent. Your constraint is replaced and the solver tries again.",
                          )
                        }
                      >
                        Send counter-offer
                      </Button>
                    ) : (
                      <Button icon={Undo2} onClick={() => setMode("counter")}>
                        Suggest another time
                      </Button>
                    )}
                    {!choosing && (
                      <Button
                        variant="danger"
                        icon={X}
                        loading={busy === "reject"}
                        onClick={() => act("reject", () => api.reply(item.id, { decision: "reject", text: "None of these work for me." }), "Declined. The negotiator will ask again or move on.")}
                      >
                        Decline
                      </Button>
                    )}
                  </>
                )}
                {canSimulate && (
                  <Button variant="ghost" icon={Bot} loading={busy === "sim"} className="ml-auto" onClick={() => act("sim", () => api.simulate(item.id), "The simulated stakeholder answered.")}>
                    Simulate a reply
                  </Button>
                )}
              </div>
              {error && <p className="rounded-md bg-bad/10 px-3 py-2 text-sm text-bad">{error}</p>}
              {mine && <OwnWords item={item} onDone={onDone} />}
            </div>
          )}
        </div>
      </Card>
      {mine && <RateMessage item={item} />}
      {!choosing && (
        <Collapse title="Why these options (the facts behind the message)">
          <GroundedExplanation m={m} />
        </Collapse>
      )}
    </div>
  );
}

/** Reply in your own words: the reply parser reads it into one answer, which the sender confirms before it is sent. */
function OwnWords({ item, onDone }: { item: InboxItem; onDone: (msg: string) => Promise<void> }) {
  const study = useStudySession();
  const [text, setText] = useState("");
  const [reading, setReading] = useState<{ reading: string; parser: string | null } | null>(null);
  const [rejected, setRejected] = useState(false);
  const [busy, setBusy] = useState<"check" | "send" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const check = async () => {
    setBusy("check");
    setError(null);
    setRejected(false);
    try {
      setReading(await api.replyText(item.id, text, true));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };
  const send = async () => {
    setBusy("send");
    setError(null);
    try {
      const got = await api.replyText(item.id, text, false);
      if (study) await studyApi.live({ kind: "reply", item: item.id, text, reading: got.reading, confirmed: true });
      await onDone("Reply sent. The negotiator continues with your answer.");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };
  const wrong = async () => {
    setRejected(true);
    if (study && reading) await studyApi.live({ kind: "reply", item: item.id, text, reading: reading.reading, confirmed: false });
  };

  return (
    <div className="rounded-md border border-line p-4">
      <p className="text-[13px] font-medium">Or reply in your own words</p>
      <textarea
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          setReading(null);
        }}
        rows={2}
        placeholder="e.g. Wednesday works for me, but not before 11."
        className="mt-2 w-full resize-none rounded-md border border-line bg-panel p-2.5 text-[13.5px] outline-none focus:border-brand/60"
      />
      {!reading ? (
        <Button size="sm" className="mt-2" disabled={!text.trim()} loading={busy === "check"} onClick={check}>
          Check how it is read
        </Button>
      ) : (
        <div className="mt-2 space-y-2">
          <p className="rounded-md bg-panel-2 px-3 py-2 text-[13.5px]">
            <span className="text-ink-3">The system reads this as: </span>
            {reading.reading}
          </p>
          {rejected ? (
            <p className="text-[13px] text-ink-2">Thanks, noted. Rephrase it above, or answer with the buttons.</p>
          ) : (
            <div className="flex flex-wrap gap-2">
              <Button size="sm" variant="primary" icon={Check} loading={busy === "send"} onClick={send}>
                Yes, send it
              </Button>
              <Button size="sm" icon={X} onClick={wrong}>
                No, that is not what I meant
              </Button>
            </div>
          )}
          {reading.parser && <p className="text-[11.5px] text-ink-3">Read by: {reading.parser}</p>}
        </div>
      )}
      {error && <p className="mt-2 text-[13px] text-bad">{error}</p>}
    </div>
  );
}

/** Pilot participants rate the message they received, as in the study's rating task. */
function RateMessage({ item }: { item: InboxItem }) {
  const study = useStudySession();
  const [saved, setSaved] = useState(false);
  const [busy, setBusy] = useState(false);
  if (!study) return null;
  return (
    <Card>
      <CardHeader title="Study: rate this message" subtitle="How it reads to you, whatever you answered" />
      <div className="p-5">
        {saved ? (
          <p className="flex items-center gap-2 text-[13.5px] text-ok">
            <Check size={15} /> Saved. Thank you.
          </p>
        ) : (
          <RatingForm
            busy={busy}
            submitLabel="Save rating"
            onSave={async (value) => {
              setBusy(true);
              await studyApi.live({ kind: "rating", item: item.id, text: item.message.text, value });
              setBusy(false);
              setSaved(true);
            }}
          />
        )}
      </div>
    </Card>
  );
}

/** The question a request of yours was sent back with, and the box to answer it. */
function QuestionFor({ id, onSent }: { id: string; onSent: () => void }) {
  const { data: c } = useApi(() => api.case(id), [id]);
  if (!c) return null;
  return <ClarifyForm caseId={c.id} sender={c.sender} question={c.reply || "Could you give more detail?"} onSent={onSent} />;
}
