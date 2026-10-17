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
