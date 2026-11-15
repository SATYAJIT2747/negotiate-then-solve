import { BookOpen, CalendarSearch, Eraser, Send, Sparkles, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Skeleton, useOutside } from "./ui";
import { ChatMessageView, type ChatContext } from "./thread/ChatCards";
import { Composer, MakeupToggle } from "./thread/Composer";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { chatExamples } from "../lib/examples";
import { useApi } from "../lib/hooks";
import { PREFS, usePref } from "../lib/prefs";
import type { CalendarView, ConfirmAs, InstanceView } from "../lib/types";
import { useWorkspace, type LookupMessage } from "../lib/workspace";

export type { LookupMessage };

/** The assistant beside the timetable (pages/Home): one chat for questions and requests (CHAT-PLAN.md). A
 * timetable question is answered with the times it is about, shaded on the timetable; a question about the rules
 * with the rules it cites (each opens beside the chat); anything else is offered as a request and sent only when
 * the person confirms. A follow-up ("and Friday?") is read with the messages before it and shown as it was read.
 * Requests made with clicks on the timetable appear here too, as cards that follow their status.
 *
 * What the rest of the screen tells it comes through the workspace (lib/workspace): a class clicked on the
 * timetable to name in the message, a nudge to read the chat again; it hands back answers to shade. The cards
 * are drawn by kind (thread/ChatCards). */
export function ChatPanel({ inst, cal, actions }: {
  inst: InstanceView | null;
  cal: CalendarView | null;
  /** buttons for the panel's header (the page's: show or hide the timetable) */
  actions?: ReactNode;
}) {
  const { user } = useAuth();
  const ws = useWorkspace();
  const teaches = !!user && ["faculty", "hod", "guest_faculty"].includes(user.role);
  // polled, so a request card's status follows its request
  const { data: messages, refresh, loading } = useApi(() => api.chat(), [], 4000);
  const { data: mine, refresh: refreshMine } = useApi(() => api.cases("mine"), [], 4000);
  const [text, setText] = useState("");
  const [makeupDefault] = usePref(PREFS.makeupDefault);
  const [makeup, setMakeup] = useState(makeupDefault); // ticked to start with if chosen on the Settings page
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [examples, setExamples] = useState(false);
  const box = useRef<HTMLTextAreaElement>(null);
  const end = useRef<HTMLDivElement>(null);
  const seen = useRef<number | null>(null); // the newest message already on screen, so only new answers shade
  const shown = useMemo(() => messages ?? [], [messages]);
  const groups = useMemo(() => chatExamples(user, inst), [user, inst]);
  const suggestions = useMemo(() => groups.flatMap((g) => g.items), [groups]);
  const cases = useMemo(() => new Map((mine ?? []).map((c) => [c.id, c])), [mine]);

  // the newest message in view (braces: an effect must return nothing or a clean-up function, and newer
  // browsers' scrollIntoView returns a Promise, which React would then call as one)
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [shown.length]);

  // a new timetable answer is shaded on the timetable beside the chat at once; older ones on "Show"
  useEffect(() => {
    if (!messages) return;
    const newest = messages.length ? messages[messages.length - 1].n : 0;
    if (seen.current !== null && newest > seen.current) {
      const answer = [...messages].reverse().find((m): m is LookupMessage => m.kind === "lookup" && m.n > seen.current!);
      if (answer) ws.showAnswer(answer);
    }
    seen.current = newest;
  }, [messages]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!ws.mention) return;
    const add = ws.mention.text;
    setText((t) => (t.trim() ? `${t.trim().replace(/[.!?]$/, "")} (${add})` : add));
    box.current?.focus();
  }, [ws.mention?.key]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!ws.chatTick) return;
    void refresh();
    void refreshMine();
  }, [ws.chatTick]); // eslint-disable-line react-hooks/exhaustive-deps

  const run = async (work: () => Promise<unknown>) => {
    setSending(true);
    setError(null);
    try {
      await work();
      await Promise.all([refresh(), refreshMine()]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  };
  const send = () => {
    const t = text.trim();
    if (!t || sending) return;
    void run(async () => {
      await api.chatSend(t, teaches && makeup);
      setText("");
      setExamples(false);
    });
  };
  const confirm = (n: number, as: ConfirmAs) => void run(() => api.chatConfirm(n, as));
  const asQuestion = (n: number) => void run(() => api.chatAsQuestion(n));
  const clear = () => {
    if (window.confirm("Clear the chat? Your requests stay in the Requests tab.")) void run(() => api.chatClear());
  };
  const pick = (ex: string) => {
    setText(ex);
    setExamples(false);
    box.current?.focus();
  };
  const x: ChatContext = { inst, cal, busy: sending, onConfirm: confirm, onAsQuestion: asQuestion, cases };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center justify-between border-b border-line px-3 py-2">
        <p className="text-[13.5px] font-semibold">Assistant</p>
        <div className="flex items-center gap-3">
          {shown.length > 0 && (
            <button onClick={clear} className="flex items-center gap-1 text-[11.5px] text-ink-3 hover:text-ink" title="Clear the chat (your requests stay)">
              <Eraser size={11} /> Clear
            </button>
          )}
          {actions}
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-3 py-3">
        {loading && !messages ? (
          <Skeleton className="h-40" />
        ) : !shown.length ? (
          <Welcome hasExamples={suggestions.length > 0} open={examples} onExamples={() => setExamples((v) => !v)} />
        ) : (
          shown.map((m) => <ChatMessageView key={m.n} m={m} x={x} />)
        )}
        <div ref={end} />
      </div>

      <div className="border-t border-line px-3 pb-3 pt-2">
        <Composer
          value={text}
          onChange={setText}
          onSend={send}
          sending={sending}
          inputRef={box}
          placeholder="Ask or request something…"
          error={error}
          above={examples && suggestions.length > 0 && <Examples groups={groups} onPick={pick} onClose={() => setExamples(false)} />}
          footerLeft={
            suggestions.length > 0 && (
              <button
                onClick={() => setExamples((v) => !v)}
                onMouseDown={(e) => e.stopPropagation()}
                aria-expanded={examples}
                className={examples ? "flex items-center gap-0.5 text-brand" : "flex items-center gap-0.5 hover:text-ink"}
                title="Questions and requests to start from"
              >
                <Sparkles size={11} /> Examples
              </button>
            )
          }
          footerRight={teaches && <MakeupToggle checked={makeup} onChange={setMakeup} />}
        />
      </div>
    </div>
  );
}

const GROUP_ICON: Record<string, typeof BookOpen> = { Timetable: CalendarSearch, Rules: BookOpen, Requests: Send };

/** The empty chat: what it is for; the examples to start from are under the Examples button below the box. */
function Welcome({ hasExamples, open, onExamples }: { hasExamples: boolean; open: boolean; onExamples: () => void }) {
  return (
    <div className="space-y-2 py-2">
      <p className="text-[14px] font-medium">What would you like to know or change?</p>
      <p className="text-[12px] text-ink-3">
        Ask about the timetable or the rules, or make a request: a request is sent straight away, and if it was meant as a
        question, the ? on its card answers it instead. Click a class on the timetable to cancel, move or swap it.
      </p>
      {hasExamples && (
        <button onClick={onExamples} onMouseDown={(e) => e.stopPropagation()} className="flex items-center gap-1 text-[12px] font-medium text-brand hover:underline">
          <Sparkles size={12} /> {open ? "Hide examples" : "See examples"}
        </button>
      )}
    </div>
  );
}

/** Examples to start from, by kind (timetable, rules, requests), opened from the Examples button under the box;
 * the close button, Esc or a click anywhere else puts them away. */
function Examples({ groups, onPick, onClose }: { groups: { title: string; items: string[] }[]; onPick: (ex: string) => void; onClose: () => void }) {
  const box = useOutside<HTMLDivElement>(onClose);
  return (
    <div ref={box} className="relative mb-2 max-h-72 space-y-3 overflow-y-auto rounded-lg border border-line bg-panel p-2.5">
      <button onClick={onClose} className="absolute right-1.5 top-1.5 grid size-6 place-items-center rounded text-ink-3 hover:bg-panel-2 hover:text-ink" aria-label="Hide the examples" title="Hide the examples">
        <X size={13} />
      </button>
      {groups.map((g) => {
        const Icon = GROUP_ICON[g.title] ?? Sparkles;
        return (
          <div key={g.title} className="space-y-1">
            <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-ink-3">
              <Icon size={12} /> {g.title}
            </p>
            {g.items.map((ex) => (
              <button
                key={ex}
                onClick={() => onPick(ex)}
                className="block w-full rounded-md px-2 py-1 text-left text-[12.5px] leading-snug text-ink-2 hover:bg-panel-2 hover:text-ink"
              >
                {ex}
              </button>
            ))}
          </div>
        );
      })}
    </div>
  );
}
