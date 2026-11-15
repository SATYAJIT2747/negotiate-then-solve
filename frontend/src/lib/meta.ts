import {
  Ban,
  CheckCircle2,
  CircleDashed,
  Cpu,
  FileCheck2,
  Gavel,
  HelpCircle,
  Inbox,
  MessageSquareReply,
  MessagesSquare,
  Scale,
  Send,
  ShieldAlert,
  ShieldCheck,
  ScanText,
  TriangleAlert,
  Undo2,
  type LucideIcon,
} from "lucide-react";
import type { Role, Status } from "./types";

export type Tone = "brand" | "ok" | "warn" | "bad" | "info" | "muted";

export const STATUS: Record<Status, { label: string; tone: Tone; icon: LucideIcon; terminal?: boolean }> = {
  received: { label: "Received", tone: "muted", icon: Inbox },
  classified: { label: "Classified", tone: "info", icon: ScanText },
  policy_checked: { label: "Policy checked", tone: "info", icon: Gavel },
  compiled: { label: "Compiled", tone: "info", icon: FileCheck2 },
  solved: { label: "Solved", tone: "info", icon: Cpu },
  negotiating: { label: "Negotiating", tone: "warn", icon: MessagesSquare },
  fairness_audited: { label: "Fairness audited", tone: "info", icon: Scale },
  awaiting_approval: { label: "Awaiting approval", tone: "brand", icon: CircleDashed },
  published: { label: "Published", tone: "ok", icon: CheckCircle2, terminal: true },
  refused: { label: "Refused", tone: "bad", icon: Ban, terminal: true },
  denied: { label: "Denied", tone: "bad", icon: ShieldAlert, terminal: true },
  clarification_requested: { label: "Clarification", tone: "warn", icon: HelpCircle, terminal: true },
  escalated: { label: "Escalated", tone: "warn", icon: TriangleAlert, terminal: true },
  answered: { label: "Answered", tone: "ok", icon: MessageSquareReply, terminal: true },
  forwarded: { label: "Forwarded", tone: "muted", icon: Send, terminal: true },
  withdrawn: { label: "Withdrawn", tone: "muted", icon: Undo2, terminal: true },
};

export const ROLE_LABEL: Record<Role, string> = {
  coordinator: "Timetable coordinator",
  dean: "Dean",
  hod: "Head of Department",
  faculty: "Faculty",
  guest_faculty: "Guest faculty",
  lab_incharge: "Lab in-charge",
  exam_cell: "Exam cell",
  student: "Class representative",
};

/** Roles whose home is their week's timetable with the chat beside it (pages/Home: questions, rules, requests), and whose
 * sidebar keeps only the essentials. Everyone else opens on the dashboard. */
export const TIMETABLE_HOME: Role[] = ["faculty", "guest_faculty", "student"];

/** Roles whose Requests page is their own requests, each read as a conversation (pages/RequestsTable, RequestChat):
 * the timetable-home roles, and the head of department and the Dean, whose list also carries, on top, only the
 * requests that wait on their decision (the department's list is a link away). The timetable office runs the
 * pipeline, so it gets every request (pages/Requests). */
export const REQUESTS_AS_CHATS: Role[] = [...TIMETABLE_HOME, "hod", "dean"];

/** Roles whose home page is the request workspace (pages/Home): the timetable people live in plus their
 * conversations. The head of department teaches, so they get it too, with their department on top; the lab
 * in-charge and the exam cell look after rooms, so theirs opens on a room's week. The office and the Dean
 * open on the dashboard, a queue of what waits on them. */
export const WORKSPACE_HOME: Role[] = [...TIMETABLE_HOME, "hod", "lab_incharge", "exam_cell"];

/** Roles that look after rooms, whose timetable opens by room. */
export const ROOM_KEEPERS: Role[] = ["lab_incharge", "exam_cell"];

/** A request's status in the sender's words: where it stands for them, not which pipeline step ran.
 * Everyone but the timetable office sees these; the pipeline's own names stay in Technical details. */
export const PLAIN_STATUS: Record<Status, { label: string; tone: Tone }> = {
  received: { label: "In progress", tone: "info" },
  classified: { label: "In progress", tone: "info" },
  policy_checked: { label: "In progress", tone: "info" },
  compiled: { label: "In progress", tone: "info" },
  solved: { label: "In progress", tone: "info" },
  fairness_audited: { label: "In progress", tone: "info" },
  negotiating: { label: "Waiting for a colleague", tone: "warn" },
  awaiting_approval: { label: "Waiting for approval", tone: "brand" },
  escalated: { label: "Waiting for a decision", tone: "warn" },
  forwarded: { label: "With the timetable office", tone: "muted" },
  clarification_requested: { label: "Needs your answer", tone: "warn" },
  published: { label: "Done", tone: "ok" },
  answered: { label: "Done", tone: "ok" },
  refused: { label: "Couldn't be done", tone: "bad" },
  denied: { label: "Couldn't be done", tone: "bad" },
  withdrawn: { label: "Withdrawn", tone: "muted" },
};

/** How a request ended up, for the colour of its reply, its steps and its line in the list: done (green),
 * not done (red), waiting on someone (amber), still being worked on (blue) or withdrawn (grey). */
export type Outcome = "ok" | "bad" | "pending" | "progress" | "muted";

export function outcomeOf(status: Status | null | undefined): Outcome {
  if (status === "published" || status === "answered") return "ok";
  if (status === "denied" || status === "refused") return "bad";
  if (status === "withdrawn") return "muted";
  if (status && ["awaiting_approval", "escalated", "negotiating", "clarification_requested", "forwarded"].includes(status)) return "pending";
  return "progress";
}

/** A message split into several requests stands where its parts stand: waiting while any part waits or is
 * under way, done when every part is, not done when every part failed, else done (some parts failed). */
export function splitOutcome(parts: Status[]): Outcome {
  const o = parts.map(outcomeOf);
  if (o.some((x) => x === "pending" || x === "progress")) return "pending";
  if (o.every((x) => x === "bad")) return "bad";
  return "ok";
}

/** "Split into 2 requests · 2 need your answer" */
export function splitLabel(parts: Status[]): string {
  const asking = parts.filter((s) => s === "clarification_requested").length;
  const waiting = parts.filter((s) => outcomeOf(s) === "pending" || outcomeOf(s) === "progress").length - asking;
  const done = parts.filter((s) => outcomeOf(s) === "ok").length;
  const head = `Split into ${parts.length} requests`;
  if (asking) return `${head} · ${asking} need${asking === 1 ? "s" : ""} your answer`;
  if (waiting) return `${head} · ${waiting} in progress`;
  return `${head} · ${done} of ${parts.length} done`;
}

/** The colour of a pipeline step in a thread: where it ended well, badly, or waits on someone. */
export function stepOutcome(kind: string): Outcome | null {
  if (["published", "answered", "handled", "choice_made"].includes(kind)) return "ok";
  if (["denied", "refused", "approval_refused", "withdrawn"].includes(kind)) return kind === "withdrawn" ? "muted" : "bad";
  if (["awaiting_approval", "escalated", "negotiating", "forwarded", "choice_offered", "held"].includes(kind)) return "pending";
  return null;
}

export const OUTCOME_DOT: Record<Outcome, string> = {
  ok: "bg-ok",
  bad: "bg-bad",
  pending: "bg-warn",
  progress: "bg-info",
  muted: "bg-ink-3/50",
};

export const OUTCOME_TEXT: Record<Outcome, string> = {
  ok: "text-ok",
  bad: "text-bad",
  pending: "text-warn",
  progress: "text-info",
  muted: "text-ink-3",
};

/** Where a role's request workspace (the timetable and the chat) is: its home page, or New request for the
 * office and the Dean (whose home is the dashboard). */
export const workspacePath = (role: Role) => (WORKSPACE_HOME.includes(role) ? "/" : "/new");

/** Tiers in the words of the person using the service; the number and who may change it stay in the tooltip. */
export const TIER: Record<number, { name: string; who: string; tone: Tone }> = {
  0: { name: "Can't be changed", who: "Nobody, it is physically impossible otherwise", tone: "bad" },
  1: { name: "Institute rule", who: "Only the Dean or the academic council can make an exception", tone: "bad" },
  2: { name: "Agreed commitment", who: "Only the academic office can change it", tone: "warn" },
  3: { name: "Confirmed absence", who: "Kept, unless the person offers an alternative themselves", tone: "brand" },
  4: { name: "Teaching need", who: "The owner or the head of department can agree to change it", tone: "info" },
  5: { name: "Preference", who: "Kept where possible; the owner can give it up freely", tone: "muted" },
};

export const EVENT_LABEL: Record<string, string> = {
  received: "Request received",
  held: "Waiting for the request reader (busy or away)",
  released: "Read",
  classified: "Classified and parsed",
  policy_checked: "Policy checked",
  compiled: "Compiled into constraints",
  superseded: "Earlier constraint replaced",
  retargeted: "Class on the day named chosen",
  choice_offered: "Times offered to the sender",
  choice_made: "Time chosen",
  solved: "Solved with CP-SAT",
  negotiating: "Negotiation started",
  negotiation_conflict: "Conflict found (MUS)",
  negotiation_message: "Message sent",
  negotiation_reply: "Reply received",
  negotiation_escalated: "Escalation brief written",
  negotiation_options: "Options ranked (MCS)",
  fairness_audited: "Fairness audited",
  awaiting_approval: "Waiting for approval",
  published: "Published",
  refused: "Refused",
  denied: "Denied",
  clarification_requested: "Clarification requested",
  escalated: "Escalated",
  answered: "Answered",
  forwarded: "Forwarded",
  withdrawn: "Withdrawn by the sender",
  handled: "Answered by the timetable office",
  approval_refused: "Approval refused",
  bootstrap: "Timetable bootstrapped",
  seeded: "Demo history loaded",
  policy_document: "Policy document added",
  rollback: "Rolled back",
  duplicate_dropped: "Duplicate dropped",
  unknown_sender: "Unknown sender rejected",
  error: "Error",
  carried: "Carried into later weeks",
  resumed: "Picked up again after a restart",
  spelled_out: "Shorthand spelled out",
  model_reachable: "Model reachable",
  model_unreachable: "Model unreachable",
  swap_consent_requested: "Swap: colleague asked",
  swap_consent: "Swap: colleague answered",
};

export const TONE_CLASS: Record<Tone, { text: string; bg: string; ring: string; dot: string }> = {
  brand: { text: "text-brand", bg: "bg-brand/12", ring: "ring-brand/25", dot: "bg-brand" },
  ok: { text: "text-ok", bg: "bg-ok/12", ring: "ring-ok/25", dot: "bg-ok" },
  warn: { text: "text-warn", bg: "bg-warn/12", ring: "ring-warn/25", dot: "bg-warn" },
  bad: { text: "text-bad", bg: "bg-bad/12", ring: "ring-bad/25", dot: "bg-bad" },
  info: { text: "text-info", bg: "bg-info/12", ring: "ring-info/25", dot: "bg-info" },
  muted: { text: "text-ink-2", bg: "bg-ink-3/12", ring: "ring-ink-3/25", dot: "bg-ink-3" },
};

export const SAFETY_ICON = ShieldCheck;

export function ago(iso: string | undefined | null): string {
  if (!iso) return "";
  const t = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}`).getTime();
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return new Date(t).toLocaleDateString();
}

export function clock(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export const pct = (x: number | null | undefined, digits = 0) =>
  x === null || x === undefined ? "–" : `${(x * 100).toFixed(digits)}%`;

export const num = (x: number | null | undefined, digits = 2) =>
  x === null || x === undefined ? "–" : x.toFixed(digits);

export const secs = (x: number | null | undefined) =>
  x === null || x === undefined ? "–" : x < 1 ? `${Math.round(x * 1000)} ms` : `${x.toFixed(2)} s`;

export function initials(name: string): string {
  const parts = name.replace(/^Dr\.\s*/, "").split(/[\s,]+/).filter(Boolean);
  return (parts[0]?.[0] ?? "?").toUpperCase() + (parts[1]?.[0] ?? "").toUpperCase();
}

// Muted inks that read on paper and on the dark background alike.
const PALETTE = ["#4f6fa8", "#a0643a", "#4a8a6a", "#8a5a8f", "#b0873a", "#3f8290", "#a84f5c", "#6f7f3f", "#7a6a55", "#5a62a8", "#9a7040", "#4f7f86"];
export function colorFor(key: string): string {
  let h = 0x811c9dc5; // FNV-1a: similar keys ("C-011", "C-044") spread across the palette
  for (const ch of key) {
    h ^= ch.charCodeAt(0);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return PALETTE[h % PALETTE.length];
}

/** How the text of a policy document page was obtained, in words. */
export function how(method: string): string {
  if (method === "markdown") return "Markdown";
  if (method === "html") return "Web page";
  if (method === "text layer") return "PDF text layer";
  return method.replace(/^ocr: /, "OCR · ");
}
