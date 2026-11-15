import type {
  AskAnswer,
  CaseDetail,
  CaseSummary,
  ChatMessage,
  ClubDecision,
  ConfirmAs,
  ClubRequest,
  EventRow,
  FeedItem,
  Experiment,
  GraphData,
  InboxItem,
  InstanceView,
  Ledger,
  Observability,
  Overview,
  ParsedInput,
  Persona,
  PolicyDocument,
  PolicyDocuments,
  ReplyView,
  Rule,
  SemPreference,
  SemStatus,
  SemTimetable,
  SemesterOverview,
  Timetable,
  Transparency,
  VersionRow,
  CalendarView,
  WeeklyChange,
} from "./types";

const USER_KEY = "nts.user";
const WORLD_KEY = "nts.world";

/* Who is signed in, and in which world, is kept per tab (sessionStorage), so several tabs can be open
 * side by side as different people. localStorage keeps the last choice, so a new tab starts there; a
 * duplicated tab copies its tab's choice. */
const memory: Record<string, string | null> = {};

function tabGet(key: string): string | null {
  if (key in memory) return memory[key];
  try {
    const v = sessionStorage.getItem(key) ?? localStorage.getItem(key);
    if (v !== null) sessionStorage.setItem(key, v); // fixed for this tab from now on
    return (memory[key] = v);
  } catch {
    return null;
  }
}

function tabSet(key: string, v: string | null): void {
  memory[key] = v;
  for (const store of [() => sessionStorage, () => localStorage]) {
    try {
      if (v) store().setItem(key, v);
      else store().removeItem(key);
    } catch {
      /* storage unavailable: the choice lasts until reload */
    }
  }
}

// ?world=<id> opens this tab in that world (with ?as=, see auth.tsx)
try {
  const w = new URLSearchParams(window.location.search).get("world");
  if (w) tabSet(WORLD_KEY, w);
} catch {
  /* no URL to read */
}

/** The world (department) this tab works in; sent with every request as X-World. */
export function currentWorld(): string {
  return tabGet(WORLD_KEY) || "demo";
}

export function setCurrentWorld(id: string): void {
  tabSet(WORLD_KEY, id);
}

/** A link that opens a new tab signed in as ``person``, in this tab's world. */
export function tabLinkAs(person: string): string {
  const q = new URLSearchParams({ as: person, world: currentWorld() });
  return `${window.location.origin}/?${q}`;
}

export interface WorldRow {
  id: string;
  name: string;
  kind: "demo" | "offerings";
  status: "ready" | "starting" | "error";
  error: string | null;
  source: string;
  people?: number;
  sections?: number;
  sessions?: number;
  rooms?: number;
  seeding?: boolean;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export function currentUserId(): string | null {
  return tabGet(USER_KEY);
}

export function setCurrentUserId(id: string | null): void {
  tabSet(USER_KEY, id);
}

async function call<T>(path: string, init: RequestInit = {}): Promise<T> {
  const user = currentUserId();
  const headers = new Headers(init.headers);
  if (user) headers.set("X-User", user);
  headers.set("X-World", currentWorld());
  if (init.body) headers.set("Content-Type", "application/json");
  const res = await fetch(`/api${path}`, { ...init, headers });
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

export const api = {
  personas: () => call<Persona[]>("/personas"),
  login: (person: string) => post<Persona>("/login", { person }),
  me: () => call<Persona>("/me"),
  overview: () => call<Overview>("/overview"),
  instance: () => call<InstanceView>("/instance"),
  /** makeup: hold make-up classes for the ones an absence misses (otherwise they are cancelled). */
  /** A question about the timetable ("my free slots this week"): answered, nothing stored; else "not_a_question". */
  ask: (text: string) => post<AskAnswer>("/ask", { text }),
  /** The person's chat (CHAT-PLAN.md): all of it, or only the messages after number ``after``. */
  chat: (after = 0) => call<ChatMessage[]>(`/chat?after=${after}`),
  /** A message to the chat; returns the new messages (yours, then the reply). */
  chatSend: (text: string, makeup = false) => post<ChatMessage[]>("/chat", { text, makeup }),
  /** The answer to "send this as a request?" (message ``n``); returns the new messages. */
  chatConfirm: (n: number, as: ConfirmAs) => post<ChatMessage[]>("/chat/confirm", { n, as }),
  /** A request the chat sent, taken back and answered as a timetable question instead. */
  chatAsQuestion: (n: number) => post<ChatMessage[]>("/chat/as-question", { n }),
  chatClear: () => call<{ cleared: number }>("/chat", { method: "DELETE" }),
  submit: (text: string, makeup = false) => post<{ id: string }>("/requests", { text, makeup }),
  /** A request made with clicks: ``text`` is what people read, ``intent`` what the pipeline acts on (no parser). */
  submitForm: (text: string, intent: Record<string, unknown>) => post<{ id: string }>("/requests/form", { text, intent }),
  /** several classes of a range of dates: cancelled together, or moved (the sender away then, one week per request) */
  submitRange: (text: string, intents: Record<string, unknown>[], makeup = false) =>
    post<{ id: string }>("/requests/form", { text, intents, makeup }),
  settings: () => call<{ max_range_weeks: number }>("/settings"),
  setSettings: (max_range_weeks: number) => post<{ max_range_weeks: number }>("/settings", { max_range_weeks }),
  clarify: (id: string, text: string) => post<{ id: string }>(`/cases/${id}/clarify`, { text }),
  cases: (scope: "mine" | "all") => call<CaseSummary[]>(`/cases?scope=${scope}`),
  case: (id: string) => call<CaseDetail>(`/cases/${id}`),
  events: (after = 0, caseId?: string) =>
    call<EventRow[]>(`/events?after=${after}${caseId ? `&case=${caseId}` : ""}&limit=300`),
  inbox: () => call<InboxItem[]>("/inbox"),
  /** Everything addressed to you, newest first. */
  feed: () => call<FeedItem[]>("/feed"),
  feedSeen: (items: Record<string, number>) => post<{ ok: boolean }>("/feed/seen", { items }),
  reply: (id: string, reply: Omit<ReplyView, "decision"> & { decision: "accept" | "reject" | "counter" }) =>
    post<InboxItem>(`/inbox/${id}/reply`, reply),
  /** A reply in the person's own words, read by the reply parser; preview shows the reading without sending. */
  replyText: (id: string, text: string, preview: boolean) =>
    post<{ reply: ReplyView & { decision: string }; reading: string; parser: string | null; item?: InboxItem }>(`/inbox/${id}/reply-text`, { text, preview }),
  simulate: (id: string) => post<InboxItem>(`/inbox/${id}/simulate`),
  autopilot: () => call<Record<string, boolean>>("/autopilot"),
  setAutopilot: (person: string, on: boolean) => post<Record<string, boolean>>("/autopilot", { person, on }),
  approvals: () => call<CaseDetail[]>("/approvals"),
  /** ``version`` is null for a requirement the timetable already met: confirmed, nothing published. */
  approve: (id: string) => post<{ version: number | null; notified: Record<string, string> }>(`/approvals/${id}/approve`),
  reject: (id: string, reason: string) => post<{ status: string }>(`/approvals/${id}/reject`, { reason }),
  /** Settle an escalation sent to this person: grant (re-solved, then to approval) or decline. */
  decide: (id: string, grant: boolean, note: string) => post<{ status: string }>(`/cases/${id}/decide`, { grant, note }),
  simulateDecision: (id: string) => post<{ status: string }>(`/cases/${id}/simulate-decision`),
  /** The coordinator answers a message forwarded to them; the answer becomes the sender's reply. */
  handle: (id: string, note: string) => post<CaseDetail>(`/cases/${id}/handle`, { note }),
  act: (id: string, text: string) => post<{ id: string; case: CaseDetail }>(`/cases/${id}/act`, { text }),
  /** The sender takes a request back; everything it set in motion is undone. */
  withdraw: (id: string) => post<CaseDetail>(`/cases/${id}/withdraw`, {}),
  /** "Delete from list": one of your own finished requests off your list (it stays in the records). */
  hide: (id: string) => post<{ id: string; hidden: boolean }>(`/cases/${id}/hide`, {}),
  versions: () => call<VersionRow[]>("/versions"),
  /** ``of``: whose weeks ``mine_changed`` marks (a teacher, a section or a room the page shows); the person's own without */
  calendar: (of?: { teacher?: string; group?: string; room?: string }) =>
    call<CalendarView>(`/calendar${of && Object.values(of).some(Boolean) ? `?${new URLSearchParams(Object.entries(of).filter(([, v]) => v) as [string, string][])}` : ""}`),
  weeklyChanges: () => call<WeeklyChange[]>("/weekly-changes"),
  timetable: (opts: { version?: number; week?: number | null } = {}) => {
    const q = new URLSearchParams();
    if (opts.version) q.set("version", String(opts.version));
    if (opts.week) q.set("week", String(opts.week));
    return call<Timetable>(`/timetable?${q}`);
  },
  rollback: (version: number) => post<{ version: number }>(`/versions/${version}/rollback`),
  ledger: () => call<Ledger>("/ledger"),
  graph: () => call<GraphData>("/graph"),
  handbook: () => call<Rule[]>("/handbook"),
  documents: () => call<PolicyDocuments>("/handbook/documents"),
  uploadDocument: (name: string, data: string) => post<PolicyDocument & { rules: Rule[] }>("/handbook/documents", { name, data }),
  removeDocument: (name: string) => call<{ removed: string }>(`/handbook/documents/${encodeURIComponent(name)}`, { method: "DELETE" }),
  search: (q: string, mode?: string) =>
    call<{ query: string; tokens: string[]; mode: string; results: Rule[] }>(
      `/handbook/search?q=${encodeURIComponent(q)}${mode ? `&mode=${mode}` : ""}`,
    ),
  semester: () => call<SemesterOverview>("/semester"),
  semesterPublic: () => call<SemesterOverview>("/semester/public"),
  loadOfferings: (body: { sample: true } | { name: string; data: string; populate?: boolean }) =>
    post<{ courses: number; cohorts: number; pools: number; warnings: string[] }>("/semester/offerings", body),
  setSizes: (sizes: Record<string, number>) => post<{ ok: boolean }>("/semester/sizes", { sizes }),
  addPreference: (text: string, preview = false) => post<ParsedInput>("/semester/preferences", { text, preview }),
  myPreferences: () => call<SemPreference[]>("/semester/preferences"),
  removePreference: (id: string) => call<{ removed: string }>(`/semester/preferences/${id}`, { method: "DELETE" }),
  build: (note = "") => post<SemStatus>("/semester/build", { note }),
  semesterStatus: () => call<SemStatus>("/semester/status"),
  semesterTimetable: (version?: number) => call<SemTimetable>(`/semester/timetable${version ? `?version=${version}` : ""}`),
  proposeChange: (text: string, preview = false) => post<ParsedInput>("/semester/changes", { text, preview }),
  publishSemester: (version: number) => post<{ published: number }>(`/semester/versions/${version}/publish`),
  club: (request: ClubRequest, submit: boolean) => post<ClubDecision>("/clubs", { request, submit }),
  clubs: () => call<ClubDecision[]>("/clubs"),
  decideClub: (id: string, approve: boolean) => post<ClubDecision>(`/clubs/${id}/decide`, { approve }),
  transparency: () => call<Transparency>("/transparency"),
  observability: () => call<Observability>("/observability"),
  experiments: () => call<Experiment[]>("/experiments"),
  reset: () => post<{ ok: boolean }>("/demo/reset"),
  worlds: () => call<WorldRow[]>("/worlds"),
  newWorld: (body: { preset: "campus" } | { name: string; filename: string; data: string }) =>
    post<{ id: string; name: string; status: string }>("/worlds", body),
  deleteWorld: (id: string) => call<{ deleted: string }>(`/worlds/${id}`, { method: "DELETE" }),
  /** The demo department's courses as an offering PDF (timetable office), saved as a download. */
  demoOfferingsPdf: async () => {
    const user = currentUserId();
    const res = await fetch("/api/semester/demo-offerings.pdf", { headers: { "X-World": currentWorld(), ...(user ? { "X-User": user } : {}) } });
    if (!res.ok) throw new ApiError(res.status, res.statusText);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(await res.blob());
    a.download = "DemoCourseOfferings.pdf";
    a.click();
    URL.revokeObjectURL(a.href);
  },
};
