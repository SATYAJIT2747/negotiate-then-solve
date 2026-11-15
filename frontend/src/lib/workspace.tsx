import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { DetailsDrawer } from "../components/DetailsDrawer";
import { RulePanel } from "../components/thread/RulePanel";
import type { ChatMessage } from "./types";

/** A timetable answer in the chat: what the timetable beside it can shade. */
export type LookupMessage = Extract<ChatMessage, { kind: "lookup" }>;

/** What the parts of one screen share (the timetable, the chat and the cards in it): a class named in the
 * message being written, a nudge to read the chat again, an answer to shade on the timetable, and the side
 * panels any card can open (a request's details, a rule's text). Without a provider every action does nothing,
 * so a part can be used on its own. */
export interface Workspace {
  /** Words to add to the message being written; ``key`` changes each time. */
  mention: { text: string; key: number } | null;
  name: (text: string) => void;
  /** Changes when the chat should be read again (a request was just sent from elsewhere on the screen). */
  chatTick: number;
  refreshChat: () => void;
  /** A timetable answer to show on the timetable, and the latest one asked for (``key`` changes each time). */
  showAnswer: (m: LookupMessage) => void;
  answer: { m: LookupMessage; key: number } | null;
  openDetails: (caseId: string) => void;
  openRule: (ruleId: string) => void;
  /** Show a week on the timetable beside the chat (a timetable change's card); ``key`` changes each time. */
  goToWeek: (week: number) => void;
  weekAsked: { week: number; key: number } | null;
}

const noop = () => {};
const Ctx = createContext<Workspace>({ mention: null, name: noop, chatTick: 0, refreshChat: noop, showAnswer: noop, answer: null, openDetails: noop, openRule: noop, goToWeek: noop, weekAsked: null });

export const useWorkspace = () => useContext(Ctx);

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [mention, setMention] = useState<Workspace["mention"]>(null);
  const [chatTick, setChatTick] = useState(0);
  const [details, setDetails] = useState<string | null>(null);
  const [rule, setRule] = useState<string | null>(null);
  const [answer, setAnswer] = useState<Workspace["answer"]>(null);
  const showAnswer = useCallback((m: LookupMessage) => setAnswer((a) => ({ m, key: (a?.key ?? 0) + 1 })), []);
  const [weekAsked, setWeekAsked] = useState<Workspace["weekAsked"]>(null);
  const goToWeek = useCallback((week: number) => setWeekAsked((x) => ({ week, key: (x?.key ?? 0) + 1 })), []);
  const name = useCallback((text: string) => setMention((m) => ({ text, key: (m?.key ?? 0) + 1 })), []);
  const refreshChat = useCallback(() => setChatTick((n) => n + 1), []);
  const closeDetails = useCallback(() => setDetails(null), []);
  const closeRule = useCallback(() => setRule(null), []);
  const value = useMemo<Workspace>(
    () => ({ mention, name, chatTick, refreshChat, showAnswer, answer, openDetails: setDetails, openRule: setRule, goToWeek, weekAsked }),
    [mention, name, chatTick, refreshChat, showAnswer, answer, goToWeek, weekAsked],
  );
  return (
    <Ctx.Provider value={value}>
      {children}
      <DetailsDrawer id={details} onClose={closeDetails} />
      <RulePanel id={rule} onClose={closeRule} />
    </Ctx.Provider>
  );
}
