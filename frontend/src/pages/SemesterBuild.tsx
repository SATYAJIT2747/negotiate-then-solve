import clsx from "clsx";
import { AlertTriangle, CheckCircle2, Download, FileUp, Hammer, Loader2, Search, Trash2, Upload } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { SemesterGrid } from "../components/SemesterGrid";
import { Badge, Button, Card, CardHeader, Collapse, EmptyState, Label, PageHeader, Skeleton, Tabs, Toast } from "../components/ui";
import { api, currentWorld } from "../lib/api";
import { useApi } from "../lib/hooks";
import type { ParsedInput, SemCourse, SemMeeting, SemesterOverview, SemNeed, SemTimetable } from "../lib/types";

type Tab = "offerings" | "preferences" | "timetable" | "changes";

function readFile(f: File): Promise<string> {
  return new Promise((res, rej) => {
    const reader = new FileReader();
    reader.onload = () => res(String(reader.result).split(",")[1] ?? "");
    reader.onerror = () => rej(reader.error);
    reader.readAsDataURL(f);
  });
}

function needText(n: SemNeed): string {
  if (n.kind === "lecture") {
    const secs = n.rooms.length > 1 ? `, ${n.rooms.length} parallel sections of ${n.rooms[0]}` : "";
    return `${n.pair ? "Lecture twice a week (day + 3)" : "Lecture once a week"}${secs}`;
  }
  if (n.kind === "tutorial") return `${n.label}${n.rooms.length > 1 ? ` of ~${n.rooms[0]}` : ""}`;
  return `${n.label} · ${n.room_type} lab${n.rooms.length > 1 ? ` × ${n.rooms.length}` : ""}`;
}

export default function SemesterBuild() {
  const { data: ov, refresh } = useApi(() => api.semester(), []);
  const [params, setParams] = useSearchParams();
  const tab = (params.get("tab") as Tab) || "offerings";
  const setTab = (t: Tab) => setParams(t === "offerings" ? {} : { tab: t }, { replace: true });
  const [toast, setToast] = useState<string | null>(null);
  const { data: status, refresh: refreshStatus } = useApi(() => api.semesterStatus(), [], 2000);
  const wasRunning = useRef(false);

  useEffect(() => {
    if (status?.running) wasRunning.current = true;
    else if (wasRunning.current) {
      wasRunning.current = false;
      void refresh();
      if (status?.error) setToast(status.error);
      else if (status?.version) setToast(`Version ${status.version} is ready to review`);
    }
  }, [status, refresh]);


  if (!ov) return <Skeleton className="h-96" />;
  return (
    <>
      <PageHeader
        title="Semester plan"
        subtitle={
          ov.loaded
            ? `The whole semester, built from the course offering document on the institute's 85-minute slots; separate from this week's timetable, which requests and negotiations change. From ${ov.source}: ${ov.courses?.length ?? 0} courses, ${ov.cohorts.length} programmes, ${ov.rooms.length} rooms. ${ov.published ? `Version ${ov.published} is published.` : "Nothing published yet."}`
            : "Load the course offering document, add preferences, build, review and publish."
        }
      />
      {status?.running && (
        <p className="mb-5 flex items-center gap-2 text-[13.5px] text-ink-2">
          <Loader2 size={15} className="animate-spin text-brand" />
          {status.kind === "change" ? "Re-solving around the published timetable" : "Building the timetable"}… {status.elapsed?.toFixed(0)} s
          (a full build takes about a minute)
        </p>
      )}
      <div className="mb-5">
        <Tabs
          tabs={[
            { id: "offerings", label: "1. Offerings" },
            { id: "preferences", label: "2. Preferences", count: ov.preferences?.length },
            { id: "timetable", label: "3. Timetable", count: ov.versions?.length },
            { id: "changes", label: "4. Changes during the semester" },
          ]}
          value={tab}
          onChange={setTab}
        />
      </div>
      {tab === "offerings" && <Offerings ov={ov} refresh={refresh} toast={setToast} />}
      {tab === "preferences" && <Preferences ov={ov} refresh={refresh} />}
      {tab === "timetable" && <TimetableTab ov={ov} refresh={refresh} running={!!status?.running} start={refreshStatus} toast={setToast} />}
      {tab === "changes" && <Changes ov={ov} refresh={refresh} running={!!status?.running} start={refreshStatus} toast={setToast} />}
      <Toast message={toast} onDone={() => setToast(null)} />
    </>
  );
}

// ---------------------------------------------------------------------------

function Offerings({ ov, refresh, toast }: { ov: SemesterOverview; refresh: () => Promise<void>; toast: (s: string) => void }) {
  const file = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [sizes, setSizes] = useState<Record<string, number>>({});

  const [populate, setPopulate] = useState(currentWorld() !== "demo");
  const load = async (body: { sample: true } | { name: string; data: string; populate?: boolean }) => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.loadOfferings(body);
      toast(
        `Read ${r.courses} courses, ${r.cohorts} programmes and ${r.pools} elective pools` +
          ("populate" in body && body.populate ? ". This world's people and weekly timetable are being rebuilt from it: the sign-in list updates in a minute." : ""),
      );
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const courses = useMemo(() => {
    const ql = q.toLowerCase();
    return (ov.courses ?? []).filter((c) => !ql || `${c.code} ${c.name} ${c.faculty.join(" ")}`.toLowerCase().includes(ql));
  }, [ov.courses, q]);

  const picker = (
    <>
      <input
        ref={file}
        type="file"
        className="hidden"
        accept=".pdf,.png,.jpg,.jpeg,.html,.htm,.txt,.md"
        onChange={async (e) => {
          const f = e.target.files?.[0];
          if (f) await load({ name: f.name, data: await readFile(f), populate });
          if (file.current) file.current.value = "";
        }}
      />
      <Button icon={Upload} loading={busy} onClick={() => file.current?.click()}>
        {ov.loaded ? "Replace document" : "Upload offering document"}
      </Button>
    </>
  );

  // for demonstrations: the demo department's courses straight away, or as a PDF to upload and watch being read
  const demo = ov.sample_is_demo && (
    <>
      <label className="flex cursor-pointer items-center gap-2 text-[13px] text-ink-2">
        <input
          type="checkbox"
          checked={!!ov.demo_loaded}
          disabled={busy}
          onChange={(e) => (e.target.checked ? void load({ sample: true }) : file.current?.click())}
        />
        Use the demo department's courses (no PDF needed)
      </label>
      <Button variant="ghost" size="sm" icon={Download} onClick={() => void api.demoOfferingsPdf()} title="The same courses as a course offering PDF, to upload">
        Demo offering PDF
      </Button>
    </>
  );

  if (!ov.loaded)
    return (
      <Card className="p-8 text-center">
        <FileUp size={24} className="mx-auto text-ink-3" />
        <p className="mt-3 text-[15px] font-medium">Start from the course offering document</p>
        <p className="mx-auto mt-1 max-w-lg text-[13.5px] text-ink-3">
          A PDF (or a scan or web page) listing each course's code, name, L-T-P-C credits and faculty under programme headings. Tutorials and labs are read
          from L-T-P; the lab type from the course code.
        </p>
        <div className="mt-5 flex justify-center gap-2">
          {ov.sample_available && !ov.sample_is_demo && (
            <Button variant="primary" loading={busy} onClick={() => load({ sample: true })}>
              {ov.sample_is_demo ? "Use the demo department's courses" : "Use the Monsoon 2026 offerings"}
            </Button>
          )}
          {picker}
        </div>
        {demo && <div className="mt-4 flex flex-wrap items-center justify-center gap-3">{demo}</div>}
        {error && <p className="mt-4 text-[13px] text-bad">{error}</p>}
      </Card>
    );

  const unscheduled = (ov.courses ?? []).filter((c) => !c.scheduled);
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        {picker}
        {demo}
        <label className="flex cursor-pointer items-center gap-2 text-[13px] text-ink-2" title="Teachers, sections and their weekly classes come from the uploaded document; this world's weekly history starts again">
          <input type="checkbox" checked={populate} onChange={(e) => setPopulate(e.target.checked)} />
          Use the uploaded document's people and sections for this world
        </label>
        {ov.source && <span className="text-[12.5px] text-ink-3">Loaded: {ov.source}</span>}
        {error && <span className="text-[13px] text-bad">{error}</span>}
      </div>
      {!!ov.warnings?.length && (
        <Collapse title={`${ov.warnings.length} things to check in the document`}>
          <ul className="space-y-1 text-[13px] text-ink-2">
            {ov.warnings.map((w, i) => (
              <li key={i} className="flex gap-2">
                <AlertTriangle size={14} className="mt-0.5 shrink-0 text-warn" />
                {w}
              </li>
            ))}
          </ul>
        </Collapse>
      )}

      <Card>
        <CardHeader
          title="Programmes and enrolment"
          subtitle="Students who take the same required courses. Sizes are estimates: correct them before building."
          action={
            Object.keys(sizes).length > 0 && (
              <Button
                size="sm"
                variant="primary"
                onClick={async () => {
                  await api.setSizes(sizes);
                  setSizes({});
                  toast("Enrolment saved");
                  await refresh();
                }}
              >
                Save sizes
              </Button>
            )
          }
        />
        <div className="max-h-[420px] overflow-y-auto">
          <table className="w-full text-[13px]">
            <thead className="sticky top-0 bg-panel">
              <tr className="border-b border-line text-left text-[12px] text-ink-3">
                <th className="px-5 py-2 font-medium">Programme</th>
                <th className="px-3 py-2 font-medium">Required courses</th>
                <th className="px-3 py-2 font-medium">Electives to choose</th>
                <th className="w-24 px-3 py-2 text-right font-medium">Students</th>
              </tr>
            </thead>
            <tbody>
              {ov.cohorts.map((c) => (
                <tr key={c.id} className="border-b border-line/60 last:border-0">
                  <td className="px-5 py-2">{c.name}</td>
                  <td className="px-3 py-2 font-mono text-[11.5px] text-ink-2">{c.courses.join(", ") || "—"}</td>
                  <td className="px-3 py-2 text-[12px] text-ink-3">{c.electives.join(", ") || "—"}</td>
                  <td className="px-3 py-2 text-right">
                    <input
                      type="number"
                      min={1}
                      value={sizes[c.id] ?? c.size}
                      onChange={(e) => setSizes((s) => ({ ...s, [c.id]: Number(e.target.value) }))}
                      className="h-7 w-20 rounded-md border border-line bg-panel px-2 text-right tabular-nums outline-none focus:border-brand/60"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card>
        <CardHeader
          title="Courses and what each needs"
          subtitle="Lecture pairs meet on a day and the day three later; tutorials meet in one hour across rooms; labs take the lab type of the course's area."
          action={
            <div className="relative w-56">
              <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-ink-3" />
              <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter" className="h-8 w-full rounded-md border border-line bg-panel pl-8 pr-3 text-[13px] outline-none focus:border-brand/60" />
            </div>
          }
        />
        <div className="max-h-[560px] overflow-y-auto">
          <table className="w-full min-w-[860px] text-[13px]">
            <thead className="sticky top-0 bg-panel">
              <tr className="border-b border-line text-left text-[12px] text-ink-3">
                <th className="px-5 py-2 font-medium">Course</th>
                <th className="px-3 py-2 font-medium">L-T-P-C</th>
                <th className="px-3 py-2 font-medium">Needs</th>
                <th className="px-3 py-2 font-medium">Faculty</th>
              </tr>
            </thead>
            <tbody>
              {courses.map((c: SemCourse) => (
                <tr key={c.code} className={clsx("border-b border-line/60 align-top last:border-0", !c.scheduled && "text-ink-3")}>
                  <td className="px-5 py-2">
                    <span className="font-mono text-[12px]">{c.code}</span> {c.name}
                    {c.half !== "full" && <Badge className="ml-1.5">{c.half}</Badge>}
                    {c.cap && <span className="ml-1.5 text-[11.5px] text-ink-3">cap {c.cap}</span>}
                  </td>
                  <td className="px-3 py-2 font-mono text-[12px]">
                    {c.L}-{c.T}-{c.P}-{c.C}
                  </td>
                  <td className="px-3 py-2 text-[12.5px]">
                    {c.scheduled ? (
                      <ul className="space-y-0.5">
                        {c.needs.map((n, i) => (
                          <li key={i} className={clsx(n.kind === "lab" && "text-info")}>
                            {needText(n)}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <span>Not timetabled: {c.note}</span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-[12.5px] text-ink-2">{c.faculty.join(", ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
      {unscheduled.length > 0 && <p className="text-[12.5px] text-ink-3">{unscheduled.length} courses are not timetabled (projects, theses, sports run by the PE Centre).</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------

/** Who and what the loaded offering document names: examples never hard-code a real person. */
function named(ov: SemesterOverview) {
  const teachers = [...new Set((ov.courses ?? []).flatMap((c) => c.faculty))];
  const group = (i: number) => {
    const name = ov.cohorts[i]?.name ?? ov.cohorts[0]?.name ?? "UG2 CSE";
    return name.match(/\bsection\s+\w+/i)?.[0] ?? name; // "Section 2" rather than the whole programme heading
  };
  const lecture = ov.rooms.find((r) => r.type === "lecture")?.id ?? "R-101";
  const lab = ov.rooms.find((r) => r.type !== "lecture")?.id ?? lecture;
  return { t: (i: number) => teachers[i] ?? teachers[0] ?? "Dr. Menon", group, code: ov.courses?.[0]?.code ?? "CS1.101", lecture, lab };
}

const prefExamples = (ov: SemesterOverview) => {
  const n = named(ov);
  return [
    `${n.t(0)} prefers not to teach before 10 am`,
    `${n.group(0)} students would like no classes after 4 pm on Friday`,
    `${n.t(1)} is unavailable on Fridays`,
    `${n.code} should be in the morning`,
    `${n.group(1)} wants Tuesday afternoons free`,
  ];
};

/** Who has sent semester preferences, and which teachers in the offerings the office has not heard from
 * yet: names are compared without "Dr."/"Prof." and case. */
function HeardFrom({ ov }: { ov: SemesterOverview }) {
  const key = (n: string) => n.replace(/^(dr|prof)\.?\s+/i, "").trim().toLowerCase();
  const teachers = [...new Set((ov.catalog ?? []).flatMap((c) => c.faculty))].sort();
  const sent = new Set((ov.preferences ?? []).map((p) => key(p.source)));
  const heard = teachers.filter((t) => sent.has(key(t)));
  const waiting = teachers.filter((t) => !sent.has(key(t)));
  const others = [...new Set((ov.preferences ?? []).map((p) => p.source))].filter((s) => s && !teachers.some((t) => key(t) === key(s)));
  if (!teachers.length) return null;
  return (
    <Card className="p-5">
      <div className="flex items-baseline justify-between gap-3">
        <Label>Heard from</Label>
        <span className="text-[12px] text-ink-3">
          {heard.length} of {teachers.length} teachers
        </span>
      </div>
      <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-panel-2">
        <div className="h-full rounded-full bg-ok" style={{ width: `${(100 * heard.length) / teachers.length}%` }} />
      </div>
      {(heard.length > 0 || others.length > 0) && (
        <p className="mt-3 flex flex-wrap gap-1.5">
          {[...heard, ...others].map((t) => (
            <Badge key={t} tone="ok">
              <CheckCircle2 size={12} /> {t}
            </Badge>
          ))}
        </p>
      )}
      {waiting.length > 0 && (
        <>
          <p className="mt-3 text-[12px] text-ink-3">Not heard from yet: add theirs here if they tell you in person</p>
          <p className="mt-1.5 flex flex-wrap gap-1.5">
            {waiting.map((t) => (
              <Badge key={t} tone="muted">
                {t}
              </Badge>
            ))}
          </p>
        </>
      )}
    </Card>
  );
}

function Preferences({ ov, refresh }: { ov: SemesterOverview; refresh: () => Promise<void> }) {
  const [text, setText] = useState("");
  const [preview, setPreview] = useState<ParsedInput | null>(null);
  const [busy, setBusy] = useState(false);
  if (!ov.loaded) return <EmptyState icon={FileUp} title="Load the course offerings first" />;
  const check = async (t: string) => {
    setText(t);
    setPreview(t.trim() ? await api.addPreference(t, true) : null);
  };
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
      <Card className="h-fit p-5">
        <Label>Add a preference, as the faculty member or students said it</Label>
        <textarea
          value={text}
          onChange={(e) => void check(e.target.value)}
          rows={3}
          placeholder={`e.g. ${prefExamples(ov)[0]}`}
          className="w-full resize-none rounded-md border border-line bg-panel p-3 text-[14px] outline-none focus:border-brand/60"
        />
        {preview && (
          <p className={clsx("mt-2 text-[13px]", preview.ok ? "text-ink-2" : "text-warn")}>
            {preview.ok ? `Understood: ${preview.summary?.join("; ")}` : preview.error}
          </p>
        )}
        <div className="mt-3 flex gap-2">
          <Button
            variant="primary"
            disabled={!preview?.ok}
            loading={busy}
            onClick={async () => {
              setBusy(true);
              await api.addPreference(text);
              setText("");
              setPreview(null);
              setBusy(false);
              await refresh();
            }}
          >
            Add
          </Button>
        </div>
        <Label className="mt-6">Examples</Label>
        <ul className="space-y-1">
          {prefExamples(ov).map((e) => (
            <li key={e}>
              <button onClick={() => void check(e)} className="text-left text-[13px] text-ink-2 hover:text-brand">
                “{e}”
              </button>
            </li>
          ))}
        </ul>
        <p className="mt-5 text-[12px] text-ink-3">
          “can't”, “unavailable” or “away” make it a hard rule; anything else is weighed against the other preferences. Faculty and class representatives can also
          send theirs from Semester preferences.
        </p>
      </Card>
      <div className="min-w-0 space-y-6">
      <HeardFrom ov={ov} />
      <Card>
        <CardHeader title={`Preferences (${ov.preferences?.length ?? 0})`} subtitle="Used by the next build or change, grouped by who sent them" />
        <ul className="divide-y divide-line">
          {[...(ov.preferences ?? [])].sort((a, b) => a.source.localeCompare(b.source)).flatMap((p, i, all) => [
            ...(i === 0 || all[i - 1].source !== p.source
              ? [
                  <li key={`h-${p.source}`} className="bg-panel-2/50 px-5 py-1.5 text-[11.5px] font-medium text-ink-3">
                    {p.source || "Unknown"} · {all.filter((x) => x.source === p.source).length}
                  </li>,
                ]
              : []),
            <li key={p.id} className="flex items-start gap-3 px-5 py-3">
              <div className="min-w-0 flex-1">
                <p className="text-[13.5px]">{p.text}</p>
                <p className="mt-0.5 text-[12px] text-ink-3">
                  {p.target} {p.who.length > 40 ? `${p.who.slice(0, 40)}…` : p.who} · {p.mode}
                  {p.days ? ` · ${p.days.join(", ")}` : ""}
                  {p.start || p.end ? ` · ${p.start ?? "start"}–${p.end ?? "end"}` : ""}
                </p>
              </div>
              <Badge tone={p.hard ? "bad" : "muted"}>{p.hard ? "required" : "preference"}</Badge>
              <button
                onClick={async () => {
                  await api.removePreference(p.id);
                  await refresh();
                }}
                className="grid size-7 place-items-center rounded-md text-ink-3 hover:bg-panel-2 hover:text-bad"
                title="Remove"
              >
                <Trash2 size={14} />
              </button>
            </li>,
          ])}
          {!ov.preferences?.length && <li className="px-5 py-8 text-center text-[13px] text-ink-3">No preferences yet.</li>}
        </ul>
        {!!ov.closures?.length && (
          <div className="border-t border-line px-5 py-3">
            <Label>Room closures</Label>
            {ov.closures.map((c, i) => (
              <p key={i} className="text-[13px] text-ink-2">
                {c.room}: {c.window.days.join(", ")} {c.window.start}–{c.window.end}
              </p>
            ))}
          </div>
        )}
      </Card>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

function Report({ tt }: { tt: SemTimetable }) {
  const r = tt.report;
  const met = r.preferences.filter((p) => p.met).length;
  const items: [string, string, boolean][] = [
    ["Sessions placed", `${r.components} (${r.meetings} meetings)`, true],
    ["Hard rules broken", String(r.hard_violations.length), r.hard_violations.length === 0],
    ["Preferences met", `${met} of ${r.preferences.length}`, met === r.preferences.length],
    ["Electives of one pool clashing", String(r.pool_clashes), r.pool_clashes === 0],
    ["Evening (slot 6) meetings", String(r.evening_sessions), true],
    [tt.kind === "change" ? "Sessions moved" : "Solver", tt.kind === "change" ? String(r.moved) : `${r.status} · ${r.seconds} s`, true],
  ];
  return (
    <Card className="grid grid-cols-2 gap-4 p-5 md:grid-cols-3 xl:grid-cols-6">
      {items.map(([k, v, ok]) => (
        <div key={k}>
          <p className="text-[12px] text-ink-3">{k}</p>
          <p className={clsx("mt-0.5 text-[15px] font-medium", !ok && "text-bad")}>{v}</p>
        </div>
      ))}
    </Card>
  );
}

type By = "cohort" | "faculty" | "room" | "course";

/** The solver matches teachers by a key without title or case (``faculty_key`` in semester/solver.py);
 * people are shown by the name the offering document gives, "Dr. Menon" rather than "menon". */
const facultyKey = (name: string) =>
  name.replace(/\(.*?\)|^(dr|prof)\.?\s+/gi, "").replace(/\s+/g, " ").trim().toLowerCase();

function facultyNames(ov: SemesterOverview): (key: string) => string {
  const names = new Map<string, string>();
  for (const c of ov.courses ?? []) for (const f of c.faculty) names.set(facultyKey(f), f);
  return (key) => names.get(key) ?? key;
}

function Viewer({ ov, tt, changed }: { ov: SemesterOverview; tt: SemTimetable; changed?: Set<string> }) {
  const [by, setBy] = useState<By>("cohort");
  const [who, setWho] = useState<string>("");
  const [picked, setPicked] = useState<SemMeeting | null>(null);
  const display = useMemo(() => facultyNames(ov), [ov]);
  const options = useMemo(() => {
    if (by === "cohort") return ov.cohorts.map((c) => ({ id: c.id, label: c.name }));
    if (by === "room") return ov.rooms.map((r) => ({ id: r.id, label: `${r.id} (${r.type}, ${r.capacity})` }));
    if (by === "faculty")
      return [...new Set(tt.meetings.flatMap((m) => m.faculty))].map((f) => ({ id: f, label: display(f) })).sort((a, b) => a.label.localeCompare(b.label));
    return [...new Map(tt.meetings.map((m) => [m.course, `${m.course} ${m.name}`])).entries()].sort().map(([id, label]) => ({ id, label }));
  }, [by, ov, tt, display]);
  useEffect(() => {
    if (!options.some((o) => o.id === who)) setWho(options[0]?.id ?? "");
  }, [options, who]);
  const fixed = tt.meetings.filter((m) =>
    by === "cohort" ? m.cohorts.includes(who) : by === "room" ? m.rooms.includes(who) : by === "faculty" ? m.faculty.includes(who) : m.course === who,
  );

  // A programme's elective slots, merged by name ("Bouquet Core" twice -> one chip, x2)
  const slots = useMemo(() => {
    if (by !== "cohort") return [];
    const out = new Map<string, { slot: string; count: number; pools: { id: string; name: string; courses: string[] }[] }>();
    for (const e of ov.cohorts.find((c) => c.id === who)?.elective_pools ?? []) {
      const s = out.get(e.slot);
      if (s) s.count += 1;
      else out.set(e.slot, { slot: e.slot, count: 1, pools: e.pools });
    }
    return [...out.values()];
  }, [by, who, ov]);
  const [on, setOn] = useState<Set<string>>(new Set());
  useEffect(() => {
    // a programme with only electives would otherwise show an empty week
    setOn(new Set(fixed.length === 0 ? slots.filter((s) => s.pools.length).map((s) => s.slot) : []));
  }, [who, by, slots]);
  const optionCourses = new Set(slots.filter((s) => on.has(s.slot)).flatMap((s) => s.pools.flatMap((p) => p.courses)));
  const fixedCourses = new Set(fixed.map((m) => m.course));
  const electives = by === "cohort" ? tt.meetings.filter((m) => m.kind === "lecture" && optionCourses.has(m.course) && !fixedCourses.has(m.course) && !m.cohorts.includes(who)) : [];
  const optional = new Set(electives.map((m) => m.component));
  const shown = [...fixed, ...electives];

  return (
    <Card>
      <div className="flex flex-wrap items-center gap-3 border-b border-line p-4">
        <Tabs
          tabs={[
            { id: "cohort", label: "Programme" },
            { id: "faculty", label: "Faculty" },
            { id: "room", label: "Room" },
            { id: "course", label: "Course" },
          ]}
          value={by}
          onChange={setBy}
        />
        <select value={who} onChange={(e) => setWho(e.target.value)} className="h-8 max-w-md rounded-md border border-line bg-panel px-2.5 text-[13px] outline-none">
          {options.map((o) => (
            <option key={o.id} value={o.id}>
              {o.label}
            </option>
          ))}
        </select>
        <span className="ml-auto text-[12.5px] text-ink-3">
          {fixed.length} meetings a week{electives.length > 0 && ` + ${electives.length} elective lectures`}
        </span>
      </div>
      {slots.length > 0 && (
        <div className="flex flex-wrap items-center gap-2 border-b border-line px-4 py-3">
          <span className="mr-1 text-[12.5px] text-ink-3">{fixed.length === 0 ? "No fixed courses; students choose:" : "Elective slots:"}</span>
          {slots.map((s) => {
            const active = on.has(s.slot);
            const n = s.pools.reduce((k, p) => k + p.courses.length, 0);
            return (
              <button
                key={s.slot}
                disabled={!s.pools.length}
                onClick={() => setOn((cur) => {
                  const next = new Set(cur);
                  if (next.has(s.slot)) next.delete(s.slot);
                  else next.add(s.slot);
                  return next;
                })}
                title={s.pools.length ? s.pools.map((p) => p.name).join(" · ") : "No class meetings (project work)"}
                className={clsx(
                  "rounded-md border px-2.5 py-1 text-[12.5px]",
                  !s.pools.length ? "cursor-default border-line text-ink-3" : active ? "border-brand/60 bg-brand/10 text-ink" : "border-dashed border-line text-ink-2 hover:border-brand/40",
                )}
              >
                {s.slot}
                {s.count > 1 && ` ×${s.count}`}
                <span className="ml-1.5 text-ink-3">{s.pools.length ? `${n} courses` : "no classes"}</span>
              </button>
            );
          })}
        </div>
      )}
      <div className="p-4">
        {fixed.length === 0 && by === "cohort" && !slots.some((s) => s.pools.length) && (
          <p className="mb-3 text-[13px] text-ink-3">The offering document lists no timetabled courses for this programme.</p>
        )}
        <SemesterGrid calendar={ov.calendar} meetings={shown} optional={optional} changed={changed} onPick={setPicked} />
        {electives.length > 0 && (
          <p className="mt-3 text-[12px] text-ink-3">
            Dashed boxes are elective lectures (their tutorials and labs are under Course): a student attends only the ones they register for. Lectures of one pool are kept in different slots where
            possible; courses from different pools may share one.
          </p>
        )}
      </div>
      {picked && (
        <div className="border-t border-line px-5 py-3 text-[13px]">
          <span className="font-medium">
            {picked.course} {picked.name}
          </span>{" "}
          · {picked.label} · {picked.day} {picked.start}–{picked.end}
          {picked.half !== "full" && ` · ${picked.half === "H1" ? "first" : "second"} half of the semester`} · {picked.rooms.join(", ")}
          {picked.faculty.length > 0 && ` · ${picked.faculty.map(display).join(", ")}`}
        </div>
      )}
    </Card>
  );
}

function TimetableTab({
  ov,
  refresh,
  running,
  start,
  toast,
}: {
  ov: SemesterOverview;
  refresh: () => Promise<void>;
  running: boolean;
  start: () => Promise<void>;
  toast: (s: string) => void;
}) {
  const versions = ov.versions ?? [];
  const [version, setVersion] = useState<number | undefined>(undefined);
  const current = version ?? versions.at(-1)?.version;
  const { data: tt } = useApi(() => (current ? api.semesterTimetable(current) : Promise.resolve(null)), [current, versions.length]);
  if (!ov.loaded) return <EmptyState icon={FileUp} title="Load the course offerings first" />;
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="primary"
          icon={Hammer}
          loading={running}
          onClick={async () => {
            await api.build(`build with ${ov.preferences?.length ?? 0} preferences`);
            await start();
          }}
        >
          {versions.length ? "Build again" : "Build the timetable"}
        </Button>
        {versions.length > 0 && (
          <select
            value={current}
            onChange={(e) => setVersion(Number(e.target.value))}
            className="h-9 rounded-md border border-line bg-panel px-2.5 text-[13px] outline-none"
          >
            {[...versions].reverse().map((v) => (
              <option key={v.version} value={v.version}>
                Version {v.version} · {v.kind === "change" ? `change: ${v.note.slice(0, 40)}` : v.note} {v.published ? "· published" : ""}
              </option>
            ))}
          </select>
        )}
        {tt && !tt.published && tt.report.hard_violations.length === 0 && (
          <Button
            icon={CheckCircle2}
            onClick={async () => {
              await api.publishSemester(tt.version);
              toast(`Version ${tt.version} published`);
              await refresh();
            }}
          >
            Publish version {tt.version}
          </Button>
        )}
        {tt?.published && <Badge tone="ok">published</Badge>}
      </div>
      {!versions.length && !running && <EmptyState icon={Hammer} title="No timetable yet" text="Build one from the offerings and preferences. It takes about a minute." />}
      {tt && (
        <>
          <Report tt={tt} />
          {tt.report.hard_violations.length > 0 && (
            <Card className="p-5">
              {tt.report.hard_violations.slice(0, 10).map((v, i) => (
                <p key={i} className="text-[13px] text-bad">
                  {v}
                </p>
              ))}
            </Card>
          )}
          {tt.report.preferences.some((p) => !p.met) && (
            <p className="text-[13px] text-ink-2">
              Not met: {tt.report.preferences.filter((p) => !p.met).map((p) => `“${p.text}”`).join("; ")}
            </p>
          )}
          <Viewer ov={ov} tt={tt} />
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

const changeExamples = (ov: SemesterOverview) => {
  const n = named(ov);
  return [
    `${n.t(2)} is unavailable on Tuesdays`,
    `Room ${n.lecture} is closed on Mondays`,
    `${n.t(0)} can't teach after 4 pm`,
    `Room ${n.lab} is closed on Thursday afternoons`,
  ];
};

function Changes({
  ov,
  refresh,
  running,
  start,
  toast,
}: {
  ov: SemesterOverview;
  refresh: () => Promise<void>;
  running: boolean;
  start: () => Promise<void>;
  toast: (s: string) => void;
}) {
  const [text, setText] = useState("");
  const [preview, setPreview] = useState<ParsedInput | null>(null);
  const [error, setError] = useState<string | null>(null);
  const last = (ov.versions ?? []).at(-1);
  const proposal = last && last.kind === "change" && !last.published ? last : null;
  const { data: tt } = useApi(() => (proposal ? api.semesterTimetable(proposal.version) : Promise.resolve(null)), [proposal?.version]);
  if (!ov.published) return <EmptyState icon={Hammer} title="Publish a timetable first" text="Changes during the semester re-solve around the published timetable." />;
  const check = async (t: string) => {
    setText(t);
    setError(null);
    setPreview(t.trim() ? await api.proposeChange(t, true) : null);
  };
  return (
    <div className="space-y-6">
      <Card className="p-5">
        <Label>What changed?</Label>
        <textarea
          value={text}
          onChange={(e) => void check(e.target.value)}
          rows={2}
          placeholder={`e.g. ${changeExamples(ov)[0]}`}
          className="w-full resize-none rounded-md border border-line bg-panel p-3 text-[14px] outline-none focus:border-brand/60"
        />
        {preview && <p className={clsx("mt-2 text-[13px]", preview.ok ? "text-ink-2" : "text-warn")}>{preview.ok ? `Understood: ${preview.summary?.join("; ")}` : preview.error}</p>}
        {error && <p className="mt-2 text-[13px] text-bad">{error}</p>}
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <Button
            variant="primary"
            disabled={!preview?.ok}
            loading={running}
            onClick={async () => {
              try {
                await api.proposeChange(text);
                setText("");
                setPreview(null);
                await start();
              } catch (e) {
                setError((e as Error).message);
              }
            }}
          >
            Re-solve around it
          </Button>
          <span className="flex flex-wrap gap-x-3 text-[12.5px] text-ink-3">
            {changeExamples(ov).map((e) => (
              <button key={e} onClick={() => void check(e)} className="hover:text-brand">
                “{e}”
              </button>
            ))}
          </span>
        </div>
        <p className="mt-3 text-[12px] text-ink-3">Every session keeps its time unless it has to move; moving one costs more than any preference.</p>
      </Card>
      <FromRequests />
      {proposal && tt && (
        <>
          <div className="flex flex-wrap items-center gap-3">
            <p className="text-[14px]">
              Version {tt.version} proposes <span className="font-medium">{tt.diff.length}</span> change{tt.diff.length === 1 ? "" : "s"} to version {proposal.based_on}.
            </p>
            {tt.report.hard_violations.length === 0 && (
              <Button
                variant="primary"
                icon={CheckCircle2}
                onClick={async () => {
                  await api.publishSemester(tt.version);
                  toast(`Version ${tt.version} published`);
                  await refresh();
                }}
              >
                Publish
              </Button>
            )}
          </div>
          <Card>
            <CardHeader title="What moves" subtitle={proposal.note} />
            <table className="w-full text-[13px]">
              <thead>
                <tr className="border-b border-line text-left text-[12px] text-ink-3">
                  <th className="px-5 py-2 font-medium">Session</th>
                  <th className="px-3 py-2 font-medium">Before</th>
                  <th className="px-3 py-2 font-medium">After</th>
                </tr>
              </thead>
              <tbody>
                {tt.diff.map((d) => (
                  <tr key={d.component} className="border-b border-line/60 last:border-0">
                    <td className="px-5 py-2">
                      <span className="font-mono text-[12px]">{d.course}</span> {d.name} <span className="text-ink-3">· {d.label}</span>
                    </td>
                    <td className="px-3 py-2 text-ink-3 line-through decoration-bad/50">{d.before}</td>
                    <td className="px-3 py-2 font-medium text-ok">{d.after}</td>
                  </tr>
                ))}
                {!tt.diff.length && (
                  <tr>
                    <td colSpan={3} className="px-5 py-6 text-center text-ink-3">
                      Nothing has to move.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </Card>
          <Report tt={tt} />
          <Viewer ov={ov} tt={tt} changed={new Set(tt.diff.map((d) => d.component))} />
        </>
      )}
    </div>
  );
}

/** What requests changed in the weekly timetable, for reference beside the semester plan. */
function FromRequests() {
  const { data } = useApi(() => api.weeklyChanges(), [], 10000);
  const where = (d: { day: string; time: string; room: string } | null) => (d ? `${d.day} ${d.time} · ${d.room}` : "");
  return (
    <Card>
      <CardHeader
        title="Changes made through requests"
        subtitle="Published in the weekly timetable (This week). They do not change this semester plan, which runs on its own 85-minute grid; re-solve above if one should."
      />
      {!data ? (
        <Skeleton className="m-5 h-16" />
      ) : !data.length ? (
        <p className="px-5 pb-5 text-[13px] text-ink-3">No request has changed the weekly timetable yet.</p>
      ) : (
        <ul className="divide-y divide-line/60">
          {data.map((c) => (
            <li key={c.case} className="px-5 py-3">
              <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                <Link to={`/requests/${c.case}`} className="font-mono text-[12px] text-brand hover:underline">
                  {c.case}
                </Link>
                <span className="text-[13px] font-medium">{c.sender_name}</span>
                <Badge tone={c.week ? "info" : "muted"}>{c.week ? `week ${c.week}` : "every week"}</Badge>
                {c.version ? <span className="text-[11.5px] text-ink-3">v{c.version}{c.decided_by ? ` · approved by ${c.decided_by}` : ""}</span> : <span className="text-[11.5px] text-ink-3">recorded · nothing had to move</span>}
              </div>
              <p className="mt-1 text-[13px] text-ink-2">“{c.text}”</p>
              {(c.moved.length > 0 || c.cancelled.length > 0) && (
                <ul className="mt-1.5 space-y-0.5 text-[12px]">
                  {c.moved.map((d) => (
                    <li key={d.session} className="text-ink-3">
                      {d.session_name.replace(/^the /, "")}: <span className="line-through decoration-bad/50">{where(d.before)}</span> → <span className="text-ok">{where(d.after)}</span>
                    </li>
                  ))}
                  {c.cancelled.map((d) => (
                    <li key={d.session} className="text-bad">
                      {d.session_name.replace(/^the /, "")}: cancelled ({where(d.before)}), make-up owed
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
