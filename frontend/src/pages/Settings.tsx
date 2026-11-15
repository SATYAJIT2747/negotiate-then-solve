import { Bot, Cpu, Moon, PanelLeft, Send, Sun, Users } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { Button, Card, CardHeader, LiveDot, PageHeader, Skeleton, Switch } from "../components/ui";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useApi } from "../lib/hooks";
import { PREFS, usePref } from "../lib/prefs";
import { useTheme } from "../lib/theme";
import type { View } from "../lib/types";

const RESEARCH: View[] = ["graph", "health", "experiments", "study"];

/** Every choice in one place, for everyone. This browser's (the theme, how the screen is laid out, how a message
 * starts) take effect at once wherever they are used (lib/prefs); a teacher's "answer automatically" is kept by the
 * server. The coordinator also sets here what applies to everyone's requests. The model the system reads with is
 * shown, not set: it is chosen when the server starts. */
export default function Settings() {
  const { user, sees } = useAuth();
  const teaches = ["faculty", "guest_faculty", "hod"].includes(user?.role ?? "");
  return (
    <div className="max-w-2xl space-y-5">
      <PageHeader title="Settings" subtitle="Your choices for this browser, and what you look after for everyone." />
      <Appearance />
      <Layout research={RESEARCH.some(sees)} home={sees("home")} />
      {(teaches || user?.role === "exam_cell") && <Messages teaches={teaches} />}
      {sees("settings") && <ForEveryone />}
      <System />
    </div>
  );
}

/** A choice: what it is, a line on what it does, and its control. */
function Row({ title, text, children }: { title: string; text?: ReactNode; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4 px-5 py-3.5">
      <div className="min-w-0">
        <p className="text-[13.5px] font-medium">{title}</p>
        {text && <p className="mt-0.5 text-[12.5px] text-ink-3">{text}</p>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

function Section({ title, subtitle, icon, children }: { title: string; subtitle?: string; icon: typeof Sun; children: ReactNode }) {
  return (
    <Card>
      <CardHeader title={title} subtitle={subtitle} icon={icon} />
      <div className="divide-y divide-line">{children}</div>
    </Card>
  );
}

function Appearance() {
  const { theme, toggle } = useTheme();
  const seg = (on: boolean) =>
    `flex items-center gap-1.5 rounded px-2.5 py-1 text-[13px] ${on ? "bg-panel text-ink shadow-[0_0_0_1px_var(--line)]" : "text-ink-3 hover:text-ink"}`;
  return (
    <Section title="Appearance" icon={Sun}>
      <Row title="Theme" text="Also the sun and moon button beside your name.">
        <div className="inline-flex gap-0.5 rounded-md border border-line bg-panel-2 p-0.5">
          <button className={seg(theme === "light")} onClick={() => theme !== "light" && toggle()}>
            <Sun size={13} /> Light
          </button>
          <button className={seg(theme === "dark")} onClick={() => theme !== "dark" && toggle()}>
            <Moon size={13} /> Dark
          </button>
        </div>
      </Row>
    </Section>
  );
}

function Layout({ research, home }: { research: boolean; home: boolean }) {
  const [collapsed, setCollapsed] = usePref(PREFS.sidebarCollapsed);
  const [researchShown, setResearchShown] = usePref(PREFS.researchShown);
  const [timetable, setTimetable] = usePref(PREFS.timetableShown);
  return (
    <Section title="Layout" subtitle="Remembered in this browser." icon={PanelLeft}>
      <Row title="Narrow sidebar" text="Icons only, for more room. Ctrl B switches it anywhere.">
        <Switch on={collapsed} onChange={setCollapsed} label="Narrow sidebar" />
      </Row>
      {research && (
        <Row title="Show the research pages in the sidebar" text="Knowledge graph, System health, Experiments, Human study. Hidden, they stay a click away under Research.">
          <Switch on={researchShown} onChange={setResearchShown} label="Show the research pages" />
        </Row>
      )}
      {home && (
        <Row title="Timetable beside the chat" text="Off: the home page is the chat alone, at reading width.">
          <Switch on={timetable} onChange={setTimetable} label="Timetable beside the chat" />
        </Row>
      )}
    </Section>
  );
}

function Messages({ teaches }: { teaches: boolean }) {
  const { user } = useAuth();
  const [makeup, setMakeup] = usePref(PREFS.makeupDefault);
  const { data: auto, refresh } = useApi(() => api.autopilot(), []);
  const [busy, setBusy] = useState(false);
  const on = !!user && !!auto?.[user.id];
  return (
    <Section title="Requests and messages" icon={Send}>
      {teaches && (
        <Row title="Hold make-up classes" text="When you write that you are away: on, the classes you miss move to other times that week; off, they are cancelled. Where it starts in the chat; you can still change it for one message.">
          <Switch on={makeup} onChange={setMakeup} label="Hold make-up classes" />
        </Row>
      )}
      <Row
        title="Answer messages automatically"
        text={
          <>
            <Bot size={12} className="mr-1 inline" />
            {teaches
              ? "The simulator answers what is sent to you (a clash, a time to choose) as you would."
              : "The simulator decides the changes near the examinations sent to you, as the exam cell would."}{" "}
            Kept by the server; also in the Inbox.
          </>
        }
      >
        {auto ? (
          <Switch
            on={on}
            disabled={busy}
            label="Answer messages automatically"
            onChange={async (v) => {
              setBusy(true);
              try {
                await api.setAutopilot(user!.id, v);
                await refresh();
              } finally {
                setBusy(false);
              }
            }}
          />
        ) : (
          <Skeleton className="h-5 w-8" />
        )}
      </Row>
    </Section>
  );
}

/** What the coordinator sets for everyone's requests: the longest range of dates a request may cover (the range
 * builder, "A range of dates"). */
function ForEveryone() {
  const { data, refresh } = useApi(() => api.settings(), []);
  const [weeks, setWeeks] = useState("");
  const [saving, setSaving] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  useEffect(() => {
    if (data) setWeeks(String(data.max_range_weeks));
  }, [data]);
  const n = Number(weeks);
  const valid = Number.isInteger(n) && n >= 1 && n <= 52;
  const save = async () => {
    setSaving(true);
    setNote(null);
    try {
      await api.setSettings(n);
      await refresh();
      setNote("Saved.");
    } catch (e) {
      setNote((e as Error).message);
    } finally {
      setSaving(false);
    }
  };
  return (
    <Section title="For everyone" subtitle="What applies to everyone's requests." icon={Users}>
      <Row title="Longest range of dates in a request" text="Teachers can cancel or move several classes of a range of dates in one go. A range can cover at most this many weeks.">
        {!data ? (
          <Skeleton className="h-8 w-48" />
        ) : (
          <div className="flex items-center gap-2">
            <input
              type="number"
              min={1}
              max={52}
              value={weeks}
              onChange={(e) => setWeeks(e.target.value)}
              aria-label="Weeks"
              className="h-8 w-16 rounded-md border border-line bg-panel px-2.5 text-[13px] outline-none focus:border-brand/60"
            />
            <span className="text-[13px] text-ink-2">weeks</span>
            <Button size="sm" variant="primary" loading={saving} disabled={!valid || n === data.max_range_weeks} onClick={() => void save()}>
              Save
            </Button>
          </div>
        )}
      </Row>
      {note && <p className="px-5 pb-3 text-[12.5px] text-ink-3">{note}</p>}
    </Section>
  );
}

/** What the system runs on, shown and not set (the model is chosen when the server starts). */
function System() {
  const { data: ov } = useApi(() => api.overview(), [], 5000);
  const m = ov?.model;
  return (
    <Section title="System" subtitle="Shown here; chosen when the server starts." icon={Cpu}>
      <Row title="Reading requests and replies" text={m?.error && !m.up ? m.error : undefined}>
        {!m ? (
          <Skeleton className="h-5 w-40" />
        ) : (
          <span className="flex items-center gap-2 text-[13px]">
            <LiveDot tone={m.up === null ? "muted" : m.up ? "ok" : "warn"} pulse={false} />
            {m.up === null ? "Rule-based reader (no model)" : m.up ? (m.mode === "gemini" ? "Gemini" : "Fine-tuned model") : "Model offline · using rules"}
          </span>
        )}
      </Row>
      {ov && (
        <Row title="Timetable">
          <span className="text-[13px] text-ink-2">
            {ov.department} · semester {ov.semester}
            {ov.latest_version != null ? ` · latest version ${ov.latest_version}` : ""}
            {ov.published_version != null && ov.published_version !== ov.latest_version ? ` (semester timetable: v${ov.published_version})` : ""}
          </span>
        </Row>
      )}
    </Section>
  );
}
