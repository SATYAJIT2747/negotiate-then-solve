import { ApiError, currentUserId, currentWorld } from "./api";

/** Human study (evaluation.study): a participant is known only by a code, sent as X-Study. */

const CODE_KEY = "nts.study";

export type StudyTask = "claims" | "replies" | "ratings" | "rules";

/** A step of the pilot's live session (evaluation.study.LIVE_STEPS). */
export interface LiveStep {
  step: string;
  title: string;
  text: string;
}

export interface StudySession {
  code: string;
  kind: "team" | "pilot";
  persona: string | null;
  persona_name: string | null;
  consented: string | null;
  tasks: StudyTask[];
  progress: Partial<Record<StudyTask, { done: number; total: number }>>;
  items_ready: boolean;
  consent_text: string[];
  reply_labels: string[];
  rule_labels: string[];
  /** the pilot's tour of the system, and the steps this participant has finished */
  live_steps: LiveStep[];
  steps_done: string[];
  /** the closing statements, each rated 1-5 */
  survey: { key: string; text: string }[];
}

export type StudyItem =
  | { id: string; claim: string; facts: { id: string; text: string }[] }
  | { id: string; message: string; reply: string }
  | { id: string; query: string; answer: string; rules: { id: string; title: string; text: string }[] }
  | { id: string; text: string };

export interface NextItem {
  item: StudyItem | null;
  done: number;
  total: number;
}

export interface Practice {
  colleague: string;
  course: string;
  session: string;
  task: string;
}

export interface Kappa {
  n: number;
  kappa: number | null;
  agreement?: number;
  ok?: boolean;
}

export interface H3 {
  raters: number;
  pairs: number;
  clarity: { grounded: number | null; free: number | null; wilcoxon: { n: number; p: number; median_diff: number } | null };
  acceptability: { grounded: number | null; free: number | null; wilcoxon: { n: number; p: number; median_diff: number } | null };
}

export interface StudyAnalysis {
  coverage: Record<StudyTask, { items: number; double: number; single: number }>;
  participants: { team: number; pilot: number; pilot_consented: number };
  claims: { items: number; human_vs_verifier: Record<string, Kappa>; human_vs_human: Record<string, Kappa> };
  replies: { items: number; human_vs_recorded: Record<string, Kappa>; human_vs_human: Record<string, Kappa> };
  rules?: { items: number; human_vs_judge: Record<string, Kappa>; human_vs_human: Record<string, Kappa> };
  ratings: {
    items: number;
    pilot_items: number;
    kappa: Record<string, { n: number; clarity?: number | null; acceptability?: number | null }>;
    h3_pilot: H3;
    h3_team: H3;
  };
  live: {
    requests_checked: number;
    understood: Record<string, number>;
    replies: number;
    reply_reading_confirmed: number | null;
    message_ratings: number;
    clarity: number | null;
    acceptability: number | null;
    timetable_answers?: Record<string, number>;
    rules_answers?: Record<string, number>;
    survey?: { n: number } & Record<string, number | null>;
    steps_done?: Record<string, string[]>;
  };
}

export interface StudyAdmin {
  items: null | { claims: number; replies: number; ratings: number; rules?: number; pilot_ratings: number; built: string; source: string };
  participants: (Pick<StudySession, "code" | "kind" | "persona" | "persona_name" | "consented" | "progress"> & { created: string; revoked?: string; answers_deleted?: string })[];
  analysis: StudyAnalysis | null;
  reply_parser: string | null;
}

export function studyCode(): string | null {
  try {
    return localStorage.getItem(CODE_KEY);
  } catch {
    return null;
  }
}

export function setStudyCode(code: string | null): void {
  try {
    if (code) localStorage.setItem(CODE_KEY, code);
    else localStorage.removeItem(CODE_KEY);
  } catch {
    /* storage unavailable: the session lasts until reload */
  }
}

const TASK_KEY = "nts.study.task";
const STEP_KEY = "nts.study.step";

/** The live-session step a pilot participant opened last: the checks they answer in the portal belong to it. */
export function studyStep(): string | null {
  try {
    return localStorage.getItem(STEP_KEY);
  } catch {
    return null;
  }
}

export function setStudyStep(step: string | null): void {
  try {
    if (step) localStorage.setItem(STEP_KEY, step);
    else localStorage.removeItem(STEP_KEY);
  } catch {
    /* the checks fall back to their own step */
  }
}

/** What a pilot participant was asked to do in the portal, shown in the banner there. */
export function studyTask(): string | null {
  try {
    return localStorage.getItem(TASK_KEY);
  } catch {
    return null;
  }
}

export function setStudyTask(task: string | null): void {
  try {
    if (task) localStorage.setItem(TASK_KEY, task);
    else localStorage.removeItem(TASK_KEY);
  } catch {
    /* the banner just won't show it */
  }
}

async function call<T>(path: string, init: RequestInit = {}, code = studyCode()): Promise<T> {
  const headers = new Headers(init.headers);
  const user = currentUserId();
  if (code) headers.set("X-Study", code);
  if (user) headers.set("X-User", user);
  headers.set("X-World", currentWorld());
  if (init.body) headers.set("Content-Type", "application/json");
  const res = await fetch(`/api/study${path}`, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

const post = <T>(path: string, body?: unknown) =>
  call<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const study = {
  me: (code?: string) => call<StudySession>("/me", {}, code ?? studyCode()),
  consent: () => post<StudySession>("/consent"),
  next: (task: StudyTask) => call<NextItem>(`/next/${task}`),
  label: (task: StudyTask, item: string, value: Record<string, unknown>) => post<{ ok: boolean }>("/labels", { task, item, value }),
  practice: () => post<Practice>("/practice"),
  live: (body: {
    kind: "understood" | "reply" | "rating" | "answer" | "survey";
    step?: string;
    n?: number;
    case?: string;
    item?: string;
    text?: string;
    reading?: string;
    confirmed?: boolean;
    value?: string | Record<string, unknown>;
    comment?: string;
  }) => post<{ ok: boolean }>("/live", body),
  admin: () => call<StudyAdmin>("/admin"),
  addParticipants: (kind: "team" | "pilot", n: number) => post<{ codes: string[] }>("/participants", { kind, n }),
  revoke: (code: string, deleteAnswers: boolean) => post<unknown>(`/participants/${code}/revoke`, { delete_answers: deleteAnswers }),
  restore: (code: string) => post<unknown>(`/participants/${code}/restore`),
  clear: (confirm: string) => post<{ removed: { participants: number; labels: number; live: number } }>("/clear", { confirm }),
  export: () => call<unknown>("/export"),
};
