import { useCallback, useSyncExternalStore } from "react";

/** Choices a person makes for this browser (the sidebar folded, the timetable beside the chat, …), kept in
 * localStorage and read live: changing one on the Settings page changes the sidebar or the home page at once,
 * since every reader subscribes to the same store. Private browsing (no storage): kept for this page only. */

const EVENT = "nts-pref";
const memory = new Map<string, string>(); // when localStorage cannot be used

function read(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return memory.get(key) ?? null;
  }
}

function write(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    memory.set(key, value);
  }
  window.dispatchEvent(new CustomEvent(EVENT, { detail: key }));
}

function subscribe(onChange: () => void) {
  window.addEventListener(EVENT, onChange);
  window.addEventListener("storage", onChange); // another tab changed it
  return () => {
    window.removeEventListener(EVENT, onChange);
    window.removeEventListener("storage", onChange);
  };
}

/** An on/off choice under ``key``: ``fallback`` until the person chooses. ``on``/``off``: how it is stored
 * (the keys already in people's browsers use "1"/"0" or "shown"/"hidden"). */
export function useFlag(key: string, fallback: boolean, { on = "1", off = "0" }: { on?: string; off?: string } = {}) {
  const raw = useSyncExternalStore(subscribe, () => read(key));
  const value = raw === on ? true : raw === off ? false : fallback;
  const set = useCallback((v: boolean) => write(key, v ? on : off), [key, on, off]);
  return [value, set] as const;
}

/** The choices the Settings page offers, in one place: their keys, how they are stored, and their defaults. */
export const PREFS = {
  sidebarCollapsed: { key: "nts.sidebar.collapsed", fallback: true }, // folded until opened: more room for the chat
  researchShown: { key: "nts.sidebar.research", fallback: false }, // the project's own tools stay out of the way
  timetableShown: { key: "home.timetable", fallback: true, on: "shown", off: "hidden" }, // beside the chat on Home
  makeupDefault: { key: "nts.chat.makeup", fallback: false }, // "Hold make-up classes" ticked to start with
} as const;

type Pref = (typeof PREFS)[keyof typeof PREFS];

/** One of PREFS, read live. */
export function usePref(p: Pref) {
  return useFlag(p.key, p.fallback, "on" in p ? { on: p.on, off: p.off } : {});
}
