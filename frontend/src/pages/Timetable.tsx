import clsx from "clsx";
import { Clock, FlaskConical, GitBranch, History, MapPin, MessageSquare, RotateCcw, Users } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { TimetableGrid } from "../components/TimetableGrid";
import { WeekStepper } from "../components/WeekStepper";
import { Badge, Button, Card, CardHeader, SidePanel, Skeleton, Tabs, Toast } from "../components/ui";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useApi } from "../lib/hooks";
import { ago, workspacePath } from "../lib/meta";
import type { TimetableEntry } from "../lib/types";

type By = "all" | "faculty" | "group" | "room";

export default function Timetable() {
  const { user, can, sees } = useAuth();
  const navigate = useNavigate();
  const office = sees("history"); // versions, proposals and rollback; everyone else sees the latest published
  const { data: inst } = useApi(() => api.instance(), []);
  const { data: versions, refresh: refreshVersions } = useApi(() => (office ? api.versions() : Promise.resolve([])), [office], office ? 5000 : undefined);
  const [version, setVersion] = useState<number | null>(null);
  const { data: cal } = useApi(() => api.calendar(), []);
  const [week, setWeek] = useState<number | null>(null); // null: the semester timetable, every week without its own changes
  const [weekSet, setWeekSet] = useState(false);
  useEffect(() => {
    if (cal && !weekSet) {
      setWeek(cal.current_week); // open on this week
      setWeekSet(true);
    }
  }, [cal, weekSet]);
  const { data: tt, refresh } = useApi(() => api.timetable(version ? { version } : { week }), [version, week], version ? undefined : 5000);
  const isTeacher = !!inst?.faculty.some((f) => f.id === user?.id);
  // what each person browses: the office (and HoD, Dean) everything; a teacher their own classes,
  // the sections they teach and the rooms; a class rep their section and its teachers
  const scope: "all" | "teacher" | "student" | "other" =
    office || user?.role === "hod" || user?.role === "dean" ? "all" : isTeacher ? "teacher" : user?.role === "student" ? "student" : "other";
  const myGroup = user?.role === "student" ? user.id.replace("ST-", "") : null;
  const tabs = useMemo((): { id: By; label: string }[] => {
    if (scope === "teacher") return [{ id: "faculty", label: "My classes" }, { id: "group", label: "Sections I teach" }, { id: "room", label: "Rooms" }];
    if (scope === "student") return [{ id: "group", label: "My section" }, { id: "faculty", label: "My teachers" }];
    if (scope === "other") return [{ id: "all", label: "Everything" }, { id: "room", label: "Rooms" }];
    return [{ id: "all", label: "Everything" }, { id: "faculty", label: "Teacher" }, { id: "group", label: "Section" }, { id: "room", label: "Room" }];
  }, [scope]);
  const [by, setBy] = useState<By>("all");
  const [who, setWho] = useState<string>("");
  const [picked, setPicked] = useState<TimetableEntry | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    if (!inst) return;
    if (scope === "teacher") {
      setBy("faculty");
      setWho(user!.id);
    } else if (scope === "student") {
      setBy("group");
      setWho(myGroup!);
    } else setBy("all");
  }, [inst, scope, user, myGroup]);

  const options = useMemo(() => {
    if (!inst) return [];
    const all = tt?.entries ?? [];
    if (by === "faculty") {
      if (scope === "teacher") return inst.faculty.filter((f) => f.id === user?.id).map((f) => ({ id: f.id, label: f.name }));
      const mine = scope === "student" ? new Set(all.filter((e) => e.groups.includes(myGroup!)).map((e) => e.faculty)) : null;
      return inst.faculty.filter((f) => !mine || mine.has(f.id)).map((f) => ({ id: f.id, label: f.name }));
    }
    if (by === "group") {
      if (scope === "student") return inst.groups.filter((g) => g.id === myGroup).map((g) => ({ id: g.id, label: g.name }));
      const taught = scope === "teacher" ? new Set(all.filter((e) => e.faculty === user?.id).flatMap((e) => e.groups)) : null;
      return inst.groups.filter((g) => !taught || taught.has(g.id)).map((g) => ({ id: g.id, label: g.name }));
    }
    if (by === "room") return inst.rooms.map((r) => ({ id: r.id, label: `${r.name}${r.equipment.length ? ` · ${r.equipment.join(", ")}` : ""}` }));
    return [];
  }, [inst, tt, by, scope, user, myGroup]);

  useEffect(() => {
    if (by !== "all" && options.length && !options.some((o) => o.id === who)) setWho(options[0].id);
  }, [by, options, who]);

  const entries = useMemo(() => {
    const all = tt?.entries ?? [];
    if (by === "faculty") return all.filter((e) => e.faculty === who);
    if (by === "group") return all.filter((e) => e.groups.includes(who));
    if (by === "room") return all.filter((e) => e.room === who);
    return all;
  }, [tt, by, who]);
  const changed = useMemo(() => new Set(tt?.changed ?? []), [tt]);
  // the week's changes on the grid itself, for what is shown: classes not held in their usual place (why), moved
  // classes tagged (where they usually are), and a placeholder where a moved class usually sits
  const shows = (x: { faculty?: string | null; groups: string[] }, rooms: (string | null | undefined)[]) =>
    by === "all" || (by === "faculty" && x.faculty === who) || (by === "group" && x.groups.includes(who)) || (by === "room" && rooms.includes(who));
  const notHeld = (tt?.cancelled ?? [])
    .filter((c) => c.day && c.slot != null && shows(c, [c.room]))
    .map((c) => ({
      session: c.session, day: c.day!, slot: c.slot!, title: c.title ?? c.session_name, course: c.course, kind: c.kind,
      note: `Not held${c.faculty === user?.id ? ` · ${{ away: "you are away", asked: "you cancelled it", closed: "its room is closed" }[c.reason ?? "away"]}` : c.why ? ` · ${c.why}` : ""}`,
      tone: "cancelled" as const,
    }));
  const movedShown = (tt?.moved ?? []).filter((m) => shows(m, [m.from?.room, m.to?.room]));
  const marks = new Map(
    movedShown.map((m) => {
      const sameTime = !!m.from && !!m.to && m.from.day === m.to.day && m.from.slot === m.to.slot;
      return [m.session, { tag: sameTime ? "room" : "moved", usually: m.from ? `${m.from.day} ${m.from.time} · ${m.from.room_name}` : "elsewhere" }] as const;
    }),
  );
  const movedFrom = movedShown
    .filter((m) => m.from && m.to && (m.from.day !== m.to.day || m.from.slot !== m.to.slot) && (by !== "room" || m.from.room === who))
    .map((m) => ({ session: `${m.session}:from`, day: m.from!.day, slot: m.from!.slot, title: m.title, note: `→ ${m.to!.day} ${m.to!.time}`, tone: "from" as const, course: m.course ?? undefined, kind: m.kind }));
  const ghosts = version ? [] : [...notHeld, ...movedFrom];
  // the week strip's dots for what is shown: the teacher, section or room picked (the department's for "Everything")
  const of = by === "faculty" ? { teacher: who } : by === "group" ? { group: who } : by === "room" ? { room: who } : undefined;
  const { data: marksCal } = useApi(() => (of && who ? api.calendar(of) : Promise.resolve(null)), [by, who]);

  return (
    <div className="flex flex-col gap-3 lg:h-full">
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-x-4 gap-y-1">
        <div className="min-w-0">
          <h1 className="flex flex-wrap items-center gap-3 font-serif text-[22px] leading-tight font-semibold">
            {week && !version ? `Timetable · week ${week}` : "Timetable"}
            {tt && office && (
              <Badge tone="muted" className="text-[12.5px]">
                v{tt.version}
                {tt.week ? ` · week ${tt.week}` : " · semester"}
              </Badge>
            )}
          </h1>
          <p className="truncate text-[12.5px] text-ink-3" title={office ? "Every version is kept. The semester plan is built separately (Semester plan)." : undefined}>
            The live timetable: requests and negotiations change it. Classes that moved in the latest change are outlined.
          </p>
        </div>
        {office && (
            <select
              value={version ?? ""}
              onChange={(e) => setVersion(e.target.value ? +e.target.value : null)}
              className="h-9 rounded-md border border-line bg-panel px-2.5 text-[13px] outline-none"
            >
              <option value="">Latest for the week shown</option>
              {versions?.map((v) => (
                <option key={v.version} value={v.version}>
                  v{v.version}
                  {v.week ? ` (week ${v.week})` : ""} {v.published ? "" : "· proposed"}
                </option>
              ))}
            </select>
          )}
      </div>
      {/* fills the rest of the window: the week takes the height; the office's version history sits beside it on wide
          screens, and under it (a third of the height, scrolling) on narrower ones */}
      <div
        className={clsx(
          "grid min-h-0 gap-4 lg:flex-1",
          office ? "lg:grid-rows-[minmax(0,2fr)_minmax(0,1fr)] 2xl:grid-cols-[minmax(0,1fr)_320px] 2xl:grid-rows-[minmax(0,1fr)]" : "lg:grid-rows-[minmax(0,1fr)]",
        )}
      >
        <Card className="flex h-[640px] min-h-0 flex-col lg:h-auto">
          <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-2 border-b border-line px-3 py-2.5">
            <Tabs
              tabs={tabs}
              value={by}
              onChange={setBy}
            />
            {!version && cal && <WeekStepper cal={marksCal ?? cal} week={week} onChange={setWeek} mine={by !== "all"} />}
            {by !== "all" && options.length > 1 && (
              <select value={who} onChange={(e) => setWho(e.target.value)} className="h-8 rounded-md border border-line bg-panel px-2.5 text-[13px] outline-none">
                {options.map((o) => (
                  <option key={o.id} value={o.id}>
                    {o.label}
                  </option>
                ))}
              </select>
            )}
            {scope === "teacher" && by === "group" && (
              <span className="text-[12.5px] text-ink-3">The whole week of this section, every teacher: when its students are busy or free.</span>
            )}
            {scope === "student" && by === "faculty" && <span className="text-[12.5px] text-ink-3">Everything this teacher teaches, not only your section.</span>}
            <span className="ml-auto text-[12.5px] text-ink-3">
              {entries.length} sessions · {tt?.changed.length ?? 0} changed {office ? "in this version" : "in the latest change"}
            </span>
          </div>
          {/* the rows share the card's height, so the week fits the window; "Everything" (several classes to a
              cell) keeps its row height and scrolls inside the card instead */}
          <div className="min-h-0 flex-1 p-3">
            {!inst || !tt ? (
              <Skeleton className="h-full" />
            ) : by === "all" ? (
              <div className="h-full overflow-auto">
                <TimetableGrid instance={inst} entries={entries} changed={changed} onPick={setPicked} ghosts={ghosts} marks={version ? undefined : marks} />
              </div>
            ) : (
              <TimetableGrid instance={inst} entries={entries} changed={changed} onPick={setPicked} fill ghosts={ghosts} marks={version ? undefined : marks} />
            )}
          </div>
        </Card>

        {office && (
          <Card className="flex h-fit min-h-0 flex-col lg:h-auto">
            <CardHeader icon={History} title="Version history" subtitle="Each week shows its own latest change, else the semester timetable, so the version shown need not be the newest. Rollback publishes a copy of an older version." />
            <div className="max-h-[640px] min-h-0 space-y-0 overflow-y-auto p-3 lg:max-h-none lg:flex-1">
              {versions?.map((v, i) => (
                <div key={v.version} data-index={i} className={clsx("relative rounded-md px-3 py-2", tt?.version === v.version && "bg-panel-2")}>
                  <div className="flex items-center gap-2">
                    <GitBranch size={14} className={v.published ? "text-ok" : "text-ink-3"} />
                    <button onClick={() => setVersion(v.version)} className="font-medium hover:text-brand">
                      v{v.version}
                    </button>
                    {v.week ? <Badge tone="info">week {v.week}</Badge> : <Badge tone="muted">semester</Badge>}
                    <Badge tone={v.published ? "ok" : "muted"}>{v.published ? "published" : "proposed"}</Badge>
                    {tt?.version === v.version && (
                      <span title={version ? "The version picked above" : `What week ${week ?? ""} shows: its own latest change, or the semester timetable`}>
                        <Badge tone="brand">shown{!version && week ? ` · week ${week}` : ""}</Badge>
                      </span>
                    )}
                    {can("approve") && v.published && tt?.version !== v.version && (
                      <button
                        title="Roll back to this version"
                        onClick={async () => {
                          const r = await api.rollback(v.version);
                          setToast(`Rolled back: published v${r.version} as a copy of v${v.version}.`);
                          setVersion(null);
                          await Promise.all([refresh(), refreshVersions()]);
                        }}
                        className="ml-auto grid size-7 place-items-center rounded-lg text-ink-3 hover:bg-panel-2 hover:text-brand"
                      >
                        <RotateCcw size={14} />
                      </button>
                    )}
                  </div>
                  <p className="mt-1 pl-6 text-[11.5px] text-ink-3">
                    {v.approved_by_name ? `approved by ${v.approved_by_name}` : "awaiting approval"} · {v.case ?? ""} · {ago(v.created_at)}
                  </p>
                  {v.carried.length > 0 && (
                    <p className="mt-0.5 pl-6 text-[11.5px] text-ink-3">
                      also updates{" "}
                      {v.carried.map((c, j) => (
                        <span key={c.version}>
                          {j > 0 && ", "}
                          <button onClick={() => setVersion(c.version)} className="underline decoration-dotted hover:text-brand">
                            week {c.week}
                          </button>{" "}
                          as v{c.version}
                        </span>
                      ))}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </Card>
        )}
      </div>

      <SidePanel open={!!picked} onClose={() => setPicked(null)} width={400} title={picked?.title ?? ""} subtitle={picked?.session}>
        {picked && (
          <>
            <Badge tone={picked.kind === "practical" ? "warn" : "info"}>
              {picked.kind === "practical" && <FlaskConical size={11} />} {picked.kind}
            </Badge>
            {picked.extra && (
              <Badge tone="brand" className="ml-1.5">
                extra class, week {tt?.week}
              </Badge>
            )}
            <div className="mt-5 space-y-3 text-[14px]">
              <p className="flex items-center gap-3">
                <Clock size={16} className="text-ink-3" /> {inst?.day_names[picked.day]} · {inst?.slots[picked.slot]?.label} · {picked.duration}h
              </p>
              <p className="flex items-center gap-3">
                <MapPin size={16} className="text-ink-3" /> {picked.room_name}
              </p>
              <p className="flex items-center gap-3">
                <Users size={16} className="text-ink-3" /> {picked.faculty_name} · {picked.group_names.join(", ")}
              </p>
            </div>
            {changed.has(picked.session) && <p className="mt-6 rounded-md border border-brand/40 p-3 text-[13px]">This session moved in version {tt?.version}.</p>}
            {sees("new") && (
              <Button
                variant="primary"
                icon={MessageSquare}
                className="mt-6 w-full"
                title="Open the workspace with this class named in the chat"
                onClick={() =>
                  navigate(user ? workspacePath(user.role) : "/", {
                    state: {
                      mention: `${picked.faculty === user?.id ? "my" : `${picked.faculty_name}'s`} ${picked.title} ${picked.kind} on ${inst?.day_names[picked.day] ?? picked.day} at ${inst?.slots[picked.slot]?.label ?? ""}${tt?.week ? ` in week ${tt.week}` : ""}`,
                    },
                  })
                }
              >
                Ask or request about this class
              </Button>
            )}
            <Button className="mt-2 w-full" onClick={() => setPicked(null)}>
              Close
            </Button>
          </>
        )}
      </SidePanel>
      <Toast message={toast} onDone={() => setToast(null)} />
    </div>
  );
}
