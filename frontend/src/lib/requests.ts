import type { CaseSummary, Role, Status } from "./types";

/** How far along each status is, so a message split into requests shows its least finished part. */
const ORDER: Status[] = ["received", "classified", "policy_checked", "compiled", "solved", "fairness_audited", "negotiating",
  "clarification_requested", "escalated", "awaiting_approval", "forwarded", "published", "answered", "refused", "denied", "withdrawn"];

/** Where a request stands. A message split into several requests is only "answered" (split) itself: it stands where
 * its least finished part does (one waiting for approval: waiting for approval), and is done when every part is. */
export function effectiveStatus(c: CaseSummary, byId: Map<string, CaseSummary>): Status {
  const parts = (c.parts ?? []).map((id) => byId.get(id)).filter((p): p is CaseSummary => !!p);
  if (!parts.length) return c.status;
  return parts.reduce((least, p) => (ORDER.indexOf(p.status) < ORDER.indexOf(least) ? p.status : least), parts[0].status);
}

/** The parts of a split message that are still being worked on. */
export const partsRunning = (c: CaseSummary, byId: Map<string, CaseSummary>) =>
  (c.parts ?? []).some((id) => byId.get(id)?.running);

/** Where a request stands, read the same way on every page that lists requests (All requests, the dashboard). */

/** An escalation addressed to this person: the brief names "coordinator", "dean", "hod" or the heads by id. */
export function decidesOn(c: Pick<CaseSummary, "status" | "escalated_to">, user: { id: string; role: Role } | null): boolean {
  if (c.status !== "escalated" || !user) return false;
  const to = c.escalated_to ?? "";
  if (user.role === "coordinator") return to === "coordinator";
  if (user.role === "dean") return to === "dean";
  if (user.role === "hod") return to === "hod" || to.split(", ").includes(user.id);
  return false;
}

/** The lists of All requests: what waits on the person's decision, what is still on its way (waiting on someone),
 * what was done, and what ended without a change. */
export type Stage = "needs" | "waiting" | "done" | "ended";

const DONE: Status[] = ["published", "answered"];
const ENDED: Status[] = ["refused", "denied", "withdrawn"];

export function stageOf(c: CaseSummary, user: { id: string; role: Role } | null, byId?: Map<string, CaseSummary>): Stage {
  if (decidesOn(c, user)) return "needs";
  if (byId && c.parts?.length) c = { ...c, status: effectiveStatus(c, byId) }; // a split message: where its parts are
  if (DONE.includes(c.status) || (c.status === "forwarded" && c.handled)) return "done";
  if (ENDED.includes(c.status)) return "ended";
  return "waiting";
}

/** Who a request still waits on, in a few words (null: on nobody, or it is being worked on). */
export function waitsOn(c: CaseSummary): string | null {
  if (c.held) return "the model that reads requests";
  switch (c.status) {
    case "escalated":
      return { coordinator: "the timetable office", dean: "the Dean", hod: "the head of department" }[c.escalated_to ?? ""] ?? "the head of department";
    case "awaiting_approval":
      return "the timetable office's approval";
    case "negotiating":
      return "a colleague's reply";
    case "clarification_requested":
      return "the sender's answer";
    case "forwarded":
      return c.handled ? null : c.forwarded_to === "office" ? "the office it was passed to" : "the timetable office";
    default:
      return null;
  }
}

/** What a request asks for, in words (System One's types). */
export const KIND_LABEL: Record<string, string> = {
  preference: "Preference",
  unavailability: "Away",
  swap: "Swap",
  room_issue: "Room issue",
  clash_report: "Clash report",
  policy_question: "Rule question",
  out_of_scope: "Other",
  move: "Move",
  cancel: "Cancel",
  extra_class: "Extra class",
  club_booking: "Club booking",
};

export const kindLabel = (t: string | null | undefined) => (t ? (KIND_LABEL[t] ?? t.replace(/_/g, " ")) : null);

/** Who sent it, by the part they play: teachers, class representatives, the rooms' keepers, the office. */
export type SenderGroup = "teachers" | "reps" | "rooms" | "office";
export const SENDER_GROUP: Record<Role, SenderGroup> = {
  faculty: "teachers",
  guest_faculty: "teachers",
  hod: "teachers",
  student: "reps",
  lab_incharge: "rooms",
  exam_cell: "rooms",
  coordinator: "office",
  dean: "office",
};
export const SENDER_LABEL: Record<SenderGroup, string> = { teachers: "Teachers", reps: "Class reps", rooms: "Labs and exams", office: "Office" };
