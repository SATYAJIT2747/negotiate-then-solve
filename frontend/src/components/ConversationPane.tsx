import clsx from "clsx";
import { ArrowLeft, Eraser, Send, Undo2, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useSearchParams } from "react-router-dom";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { requestExamples } from "../lib/examples";
import { useApi } from "../lib/hooks";
import type { AskAnswer, InboxItem } from "../lib/types";
import { ColleagueThread, groupOf, type Conversation } from "./Conversations";
import { RequestThread } from "./RequestThread";
import { ReplyBubble, YouBubble } from "./thread/Bubble";
import { Button, Skeleton } from "./ui";

/** Which conversation is open, its details, and what the composer under it does: answer the open request's
 * question, reply to options sent to this person, or (otherwise) write a new request. Starting to write
 * swaps the open thread for a blank new request, with a way back to it. */
export function useConversation(
  convs: Conversation[],
  refresh: () => void,
  { autoOpen = true, urlKey }: { autoOpen?: boolean; /** keep the open conversation in ?<urlKey>= across a refresh */ urlKey?: string } = {},
) {
  const { user } = useAuth();
  const [params, setParams] = useSearchParams();
  const [selected, setSelected] = useState<string | null>(() => (urlKey ? params.get(urlKey) : null)); // a case id, or "other:…"
  useEffect(() => {
    if (!urlKey || (params.get(urlKey) ?? null) === selected) return;
    setParams(
      (p) => {
        const q = new URLSearchParams(p);
        if (selected) q.set(urlKey, selected);
        else q.delete(urlKey);
        return q;
      },
      { replace: true },
    );
  }, [selected]); // eslint-disable-line react-hooks/exhaustive-deps
  const [picked, setPicked] = useState(false); // the person chose (or asked for a new request): don't choose for them
  const [back, setBack] = useState<string | null>(null); // the thread to go back to from a new request
  useEffect(() => {
    if (!autoOpen || picked || selected || !convs.length) return;
    const first = convs.find((c) => groupOf(c) === "needs") ?? convs.find((c) => groupOf(c) === "progress") ?? convs.find((c) => c.own);
    if (first) setSelected(first.key); // what needs them, else what is under way, else their latest request
  }, [autoOpen, convs, picked, selected]);

  const conv = convs.find((c) => c.key === selected) ?? null;
  const caseId = selected && !selected.startsWith("other:") ? selected : null;
  const { data: detail, refresh: refreshDetail } = useApi(() => (caseId ? api.case(caseId) : Promise.resolve(null)), [caseId], caseId ? 1500 : undefined);
  const shown = detail && detail.id === caseId ? detail : null;
  // opening a conversation reads it (things waiting on an answer stay unread until answered)
  useEffect(() => {
    const unread = (conv?.feed ?? []).filter((f) => f.unread && !f.needs_you);
    if (unread.length) void api.feedSeen(Object.fromEntries(unread.map((f) => [f.id, f.n]))).then(refresh);
  }, [conv?.key, conv?.feed.map((f) => `${f.id}:${f.n}`).join()]); // eslint-disable-line react-hooks/exhaustive-deps

  const answering = shown?.status === "clarification_requested" && shown.sender === user?.id;
  const offer: InboxItem | null =
    (conv && !conv.own
      ? conv.feed.map((f) => f.message).find((m) => m && m.to === user?.id && !m.reply)
      : shown?.inbox?.find((i) => i.to === user?.id && !i.reply)) ?? null;
  const replying = !answering && !!offer;

  return {
    selected,
    conv,
    caseId,
    shown,
    back,
    answering,
    offer,
    replying,
    /** Open a conversation (null: a blank new request). */
    open: (key: string | null) => {
      setPicked(true);
      setBack(null);
      setSelected(key);
    },
    /** Leave the open thread for a new request, unless it is waiting on an answer from this person. */
    startNew: () => {
      if (!selected || answering || replying) return;
      setBack(selected);
      setPicked(true);
      setSelected(null);
    },
    goBack: () => {
      setSelected(back);
      setBack(null);
    },
    changed: () => {
      void refreshDetail();
      refresh();
    },
  };
}

export type ConversationState = ReturnType<typeof useConversation>;

/** Classes attached to the message being written (the home page's timetable clicks). */
export interface Attachments {
  items: { key: string; label: string }[];
  remove: (key: string) => void;
  /** The message as sent: the text with the attached classes named in it. */
  compose: (text: string) => string;
  clear: () => void;
}

/** The week the timetable beside the composer shows, so a request can say when in a click. */
export interface WhenHint {
  week: number;
  days: { code: string; name: string; date?: string }[];
}

/** The open conversation and the composer under it. With nothing open, the composer is the full new-request
 * page: examples to start from, the week and days shown on the timetable to add, Undo and Clear. */
export function ConversationPane({ s, convs, attachments, when, wide = false, composerIn, onAnswer }: {
  s: ConversationState;
  convs: Conversation[];
  attachments?: Attachments;
  when?: WhenHint;
  wide?: boolean;
  /** Draw the message box here instead of under the conversation (a layout on trial: the box under the timetable). */
  composerIn?: HTMLElement | null;
  /** A timetable question was answered (e.g. to shade the free hours on the timetable). */
  onAnswer?: (a: AskAnswer) => void;
}) {
  const [text, setTextRaw] = useState("");
  const [history, setHistory] = useState<string[]>([]); // the text before each click-made change, for Undo
  // a change made by a click (an example, a week or day, Clear): Undo takes it back; typing is not recorded
  const edit = (next: string) => {
    setHistory((h) => [...h.slice(-49), text]);
    setTextRaw(next);
  };
  const setText = (next: string) => {
    setTextRaw(next);
    if (!next) setHistory([]);
  };
  const undo = () => {
    if (!history.length) return;
    setTextRaw(history[history.length - 1]);
    setHistory((h) => h.slice(0, -1));
    box.current?.focus();
  };
  const { data: inst } = useApi(() => api.instance(), []);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { user } = useAuth();
  const [makeup, setMakeup] = useState(false); // classes missed in an absence: make-up classes, or cancelled
  const teaches = !!user && ["faculty", "hod", "guest_faculty"].includes(user.role);
  const [reading, setReading] = useState<{ text: string; reading: string } | null>(null); // how a reply was read
  const box = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = box.current;
    if (!el || !composerIn) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [text, composerIn]);
  useEffect(() => setReading(null), [s.selected, text]);
  const attached = attachments?.items ?? [];
  // timetable questions answered in this session, kept until the page reloads (questions are not requests)
  const [answers, setAnswers] = useState<{ q: string; a: AskAnswer }[]>([]);
  const backTitle = useMemo(() => convs.find((c) => c.key === s.back)?.title ?? s.back, [convs, s.back]);

  /** A reply to the options in the person's own words: read first, sent once they confirm the reading. */
  const reply = async (confirm: boolean) => {
    if (!s.offer) return;
    setSending(true);
    setError(null);
    try {
      if (!confirm) {
        const got = await api.replyText(s.offer.id, text.trim(), true);
        setReading({ text: text.trim(), reading: got.reading });
      } else if (reading) {
        await api.replyText(s.offer.id, reading.text, false);
        setText("");
        setReading(null);
        s.changed();
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  };

  const send = async () => {
    const typed = text.trim();
    if (!typed) return;
    if (s.replying) return reply(false);
    if (!s.answering && !attached.length) {
      // a question about the timetable is answered from it; anything else goes on as a request
      setSending(true);
      setError(null);
      try {
        const got = await api.ask(typed);
        if (got.kind === "timetable") {
          setAnswers((x) => [...x, { q: typed, a: got }]);
          onAnswer?.(got);
          setText("");
          return;
        }
      } catch {
        /* the question service failing must not stop a request from being sent */
      } finally {
        setSending(false);
      }
    }
    const t = attachments ? attachments.compose(typed) : typed;
    setSending(true);
    setError(null);
    try {
      if (s.answering && s.shown) await api.clarify(s.shown.id, t);
      else s.open((await api.submit(t, teaches && makeup)).id);
      setText("");
      attachments?.clear();
      s.changed();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  };

  const [allExamples, setAllExamples] = useState(false);
  const composing = !s.selected && !s.answering && !s.replying; // a new request: the full composer
  const examples = composing ? requestExamples(user, inst) : [];
  const shownExamples = allExamples ? examples : examples.slice(0, 3);
  /** Add a phrase ("in week 7", "on Tuesday in week 7") to the end of the text, or take it out again. */
  const has = (phrase: string) => text.toLowerCase().includes(phrase.toLowerCase());
  const togglePhrase = (phrase: string) => {
    if (has(phrase)) edit(text.replace(new RegExp(`\\s*${phrase.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`, "i"), "").trim());
    else edit(text.trim() ? `${text.trim().replace(/[.!?]$/, "")} ${phrase}.` : `${phrase.charAt(0).toUpperCase()}${phrase.slice(1)}: `);
    box.current?.focus();
  };
  const column = wide ? "mx-auto w-full max-w-3xl" : "";
  const docked = !!composerIn; // the box on its own under the timetable: one line that grows as you type
  const makeupToggle = teaches && !s.answering && !s.replying;
  const makeupTip = "If you are away in some weeks: on, the classes you miss move to other times that week; off, they are cancelled";
  const chip = "rounded-full border px-2.5 py-1 text-[11.5px] transition-colors";
  const place = (box: ReactNode) => (composerIn ? createPortal(box, composerIn) : box);
  return (
    <>
      <div className="flex-1 overflow-y-auto p-4">
        <div className={column}>
          {s.conv && !s.conv.own ? (
            <ColleagueThread c={s.conv} onChanged={s.changed} />
          ) : s.caseId && !s.shown ? (
            <Skeleton className="h-40" />
          ) : s.shown ? (
            <RequestThread c={s.shown} onChanged={s.changed} />
          ) : (
            <div className="space-y-4">
              {answers.length > 0 && (
                <div className="space-y-3">
                  {answers.map((x, i) => (
                    <div key={i} className="space-y-1.5">
                      <YouBubble text={x.q} />
                      <ReplyBubble>{x.a.answer}</ReplyBubble>
                    </div>
                  ))}
                  <button onClick={() => setAnswers([])} className="text-[11.5px] text-ink-3 hover:text-ink">
                    Clear these answers
                  </button>
                </div>
              )}
              <div className="space-y-1.5">
                {s.back && (
                  <button onClick={s.goBack} className="mb-2 flex max-w-full items-center gap-1 text-[12px] text-ink-3 hover:text-brand">
                    <ArrowLeft size={12} className="shrink-0" />
                    <span className="truncate">Back to: {backTitle}</span>
                  </button>
                )}
                <p className="text-[14px] font-medium">New request</p>
                <p
                  className="text-[12.5px] leading-snug text-ink-3"
                  title="An absence, a class to move, a question about the rules or about the timetable (“when is Dr. Iyer free this week?”). Click a class on the timetable to cancel, move or swap it, or to name it here."
                >
                  Write as you would to the timetable office, or click a class on the timetable.
                </p>
              </div>
              {when && (
                <div className="flex flex-wrap items-center gap-1">
                  <span className="mr-0.5 text-[11.5px] text-ink-3">Add:</span>
                    {[{ phrase: `in week ${when.week}`, label: `Week ${when.week}` }, ...when.days.map((d) => ({ phrase: `on ${d.name} in week ${when.week}`, label: d.date ? `${d.name.slice(0, 3)} ${d.date}` : d.name }))].map(
                      ({ phrase, label }) => (
                        <button
                          key={phrase}
                          onClick={() => togglePhrase(phrase)}
                          title={has(phrase) ? "Take it out of the request" : `Add “${phrase}”`}
                          className={clsx(
                            "rounded-full border px-2 py-0.5 text-[11.5px] transition-colors",
                            has(phrase) ? "border-brand bg-brand/10 text-brand" : "border-line hover:bg-panel-2",
                          )}
                        >
                          {label}
                        </button>
                      ),
                    )}
                </div>
              )}
              {examples.length > 0 && (
                <div>
                  <p className="mb-1 text-[11.5px] text-ink-3">Examples</p>
                  <ul className="space-y-0.5">
                    {shownExamples.map((ex) => (
                      <li key={ex}>
                        <button onClick={() => edit(ex)} title={ex} className="block w-full truncate text-left text-[12.5px] text-ink-2 hover:text-brand">
                          “{ex}”
                        </button>
                      </li>
                    ))}
                  </ul>
                  {examples.length > 3 && (
                    <button onClick={() => setAllExamples((v) => !v)} className="mt-1 text-[11.5px] text-ink-3 hover:text-ink">
                      {allExamples ? "Fewer" : `${examples.length - 3} more`}
                    </button>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {place(
      <div className={docked ? "" : "border-t border-line p-3"}>
        <div className={column}>
          {s.replying && !reading && <p className="mb-2 text-[12px] text-brand">Pick an option above, or reply in your own words.</p>}
          {reading && (
            <div className="mb-2 rounded-md border border-brand/40 bg-brand/5 px-3 py-2 text-[12.5px]">
              <p>
                <span className="text-ink-3">We read this as:</span> {reading.reading}
              </p>
              <div className="mt-1.5 flex gap-2">
                <Button variant="primary" size="sm" loading={sending} onClick={() => void reply(true)}>
                  Send this reply
                </Button>
                <Button variant="ghost" size="sm" onClick={() => setReading(null)}>
                  Change it
                </Button>
              </div>
            </div>
          )}
          {s.answering && (
            <p className="mb-2 flex items-center gap-2 text-[12px] text-warn">
              <span className="min-w-0 flex-1 truncate">Answering: {s.shown?.reply}</span>
              <button onClick={s.startNew} className="flex shrink-0 items-center gap-1 text-ink-3 hover:text-ink" title="Write a new request instead">
                <X size={12} /> New request instead
              </button>
            </p>
          )}
          {attached.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1">
              {attached.map((a) => (
                <span key={a.key} className="flex items-center gap-1 rounded border border-brand/40 bg-brand/10 py-0.5 pl-1.5 pr-1 text-[11.5px]">
                  {a.label}
                  <button onClick={() => attachments!.remove(a.key)} className="text-ink-3 hover:text-ink" aria-label={`Remove ${a.label}`}>
                    <X size={11} />
                  </button>
                </span>
              ))}
            </div>
          )}
          <div
            className={clsx(
              "flex gap-2 border border-line bg-panel focus-within:border-brand/60",
              docked ? "items-center rounded-xl py-1.5 pl-4 pr-1.5 shadow-sm" : "items-end rounded-lg px-3 py-2",
            )}
          >
            <textarea
              ref={box}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                }
              }}
              onFocus={s.startNew}
              rows={docked ? 1 : composing ? 3 : 2}
              title="Enter to send · Shift+Enter for a new line"
              placeholder={
                s.answering
                  ? "Your answer…"
                  : s.replying
                    ? "e.g. “B works” or “Friday after 2 pm would suit me”"
                    : attached.length
                      ? "What should happen to these classes?"
                      : s.selected
                        ? "Start a new request…"
                        : docked
                          ? "Ask or request something, or click a class to cancel, reschedule or swap it…"
                          : "Ask or request something…"
              }
              className={clsx(
                "max-h-40 flex-1 resize-none bg-transparent text-[13.5px] leading-relaxed outline-none placeholder:text-ink-3",
                docked && "py-1",
              )}
            />
            {docked && makeupToggle && (
              <label className={clsx(chip, "shrink-0 cursor-pointer", makeup ? "border-brand bg-brand/10 text-brand" : "border-line text-ink-3 hover:bg-panel-2")} title={makeupTip}>
                <input type="checkbox" checked={makeup} onChange={(e) => setMakeup(e.target.checked)} className="sr-only" />
                Make-up classes{makeup ? " on" : ""}
              </label>
            )}
            <Button variant="primary" size="sm" icon={Send} loading={sending} disabled={!text.trim()} onClick={send} aria-label="Send" />
          </div>
          <div className={clsx("flex flex-wrap items-center justify-between gap-2 text-[11px] text-ink-3", docked ? "mt-1 px-2 empty:hidden" : "mt-1.5")}>
            <p className="flex items-center gap-2">
              {composing && (text || history.length > 0) && (
                <>
                  <button onClick={undo} disabled={!history.length} className="flex items-center gap-0.5 hover:text-ink disabled:opacity-40" title="Undo the last example, week or clear">
                    <Undo2 size={11} /> Undo
                  </button>
                  <button onClick={() => edit("")} disabled={!text} className="flex items-center gap-0.5 hover:text-ink disabled:opacity-40" title="Clear the text (Undo brings it back)">
                    <Eraser size={11} /> Clear
                  </button>
                </>
              )}
            </p>
            {!docked && makeupToggle && (
              <label className="flex cursor-pointer items-center gap-1.5" title={makeupTip}>
                <input type="checkbox" checked={makeup} onChange={(e) => setMakeup(e.target.checked)} className="accent-brand" />
                Hold make-up classes
              </label>
            )}
          </div>
          {error && <p className="mt-1 text-[12.5px] text-bad">{error}</p>}
        </div>
      </div>
      )}
    </>
  );
}
