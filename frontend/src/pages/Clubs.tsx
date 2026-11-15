import clsx from "clsx";
import { Check, Music, X } from "lucide-react";
import { useMemo, useState } from "react";
import { Badge, Button, Card, CardHeader, Collapse, EmptyState, Label, PageHeader, Toast } from "../components/ui";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useApi } from "../lib/hooks";
import type { ClubDecision, ClubRequest } from "../lib/types";

const STATUS: Record<ClubDecision["status"], { label: string; tone: "ok" | "warn" | "bad" }> = {
  booked: { label: "Booked", tone: "ok" },
  needs_approval: { label: "Waiting for the office", tone: "warn" },
  rejected: { label: "Not possible as asked", tone: "bad" },
};

function inDays(n: number): string {
  const d = new Date();
  d.setDate(d.getDate() + n);
  return d.toISOString().slice(0, 10);
}

const input = "h-9 w-full rounded-md border border-line bg-panel px-2.5 text-[13.5px] outline-none focus:border-brand/60";

export default function Clubs() {
  const { can } = useAuth();
  const office = can("approve");
  const { data: pub } = useApi(() => api.semesterPublic(), []);
  const { data: bookings, refresh } = useApi(() => api.clubs(), [], 5000);
  const [req, setReq] = useState<ClubRequest>({ club: "", activity: "", date: inDays(3), start: "19:00", end: "21:00", attendees: 30, cohorts: [], room: null });
  const [result, setResult] = useState<ClubDecision | null>(null);
  const [busy, setBusy] = useState<"check" | "book" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const rooms = useMemo(() => (pub?.rooms ?? []).filter((r) => r.type === "lecture").sort((a, b) => a.capacity - b.capacity), [pub]);
  const set = (patch: Partial<ClubRequest>) => {
    setReq((r) => ({ ...r, ...patch }));
    setResult(null);
  };

  const run = async (submit: boolean) => {
    setBusy(submit ? "book" : "check");
    setError(null);
    try {
      const d = await api.club(req, submit);
      setResult(d);
      if (submit) {
        setToast(d.status === "booked" ? `Booked ${d.room}` : d.status === "needs_approval" ? "Sent to the timetable office" : "Not booked: see the checks");
        await refresh();
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  if (pub && !pub.published)
    return (
      <>
        <PageHeader title="Club activities" />
        <EmptyState icon={Music} title="No semester timetable is published yet" text="Club bookings are checked against it, so they open once the office publishes one." />
      </>
    );

  const today = new Date().toISOString().slice(0, 10);
  const all = bookings ?? [];
  const waiting = all.filter((b) => b.status === "needs_approval");
  const upcoming = all.filter((b) => b.status === "booked" && b.request.date >= today).sort((x, y) => `${x.request.date} ${x.request.start}`.localeCompare(`${y.request.date} ${y.request.start}`));
  const past = all.filter((b) => !waiting.includes(b) && !upcoming.includes(b));
  const decide = async (id: string, ok: boolean) => {
    await api.decideClub(id, ok);
    setToast(ok ? "Booked; the club is told" : "Declined; the club is told");
    await refresh();
  };

  if (office)
    // the timetable office reviews the clubs' bookings: what waits on its decision, then what is booked;
    // booking a room itself (for a club that asked in person) is there, folded away
    return (
      <>
        <PageHeader title="Club activities" subtitle="The clubs' room bookings. Most are booked automatically after the checks; events over 150 people and the auditorium wait for your decision." />
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
          <div className="min-w-0 space-y-6">
            <Card>
              <CardHeader title={`Waiting for your decision (${waiting.length})`} />
              <ul className="divide-y divide-line">
                {waiting.map((b) => (
                  <li key={b.id} className="px-5 py-4">
                    <div className="flex flex-wrap items-start gap-3">
                      <div className="min-w-0 flex-1">
                        <p className="text-[14px] font-medium">
                          {b.request.club} · {b.request.activity}
                        </p>
                        <p className="mt-0.5 text-[12.5px] text-ink-2">
                          {b.day} {b.request.date}, {b.request.start}–{b.request.end} · {b.room ?? b.request.room ?? "no room yet"} · {b.request.attendees} people
                        </p>
                      </div>
                      <span className="flex gap-1.5">
                        <Button size="sm" variant="primary" onClick={() => void decide(b.id, true)}>
                          Approve
                        </Button>
                        <Button size="sm" onClick={() => void decide(b.id, false)}>
                          Decline
                        </Button>
                      </span>
                    </div>
                    {b.checks.filter((c) => !c.ok).map((c) => (
                      <p key={c.name} className="mt-1.5 flex gap-2 text-[12.5px] text-ink-2">
                        <X size={14} className="mt-0.5 shrink-0 text-warn" /> <span><span className="font-medium">{c.name}</span> · {c.detail}</span>
                      </p>
                    ))}
                  </li>
                ))}
                {!waiting.length && <li className="px-5 py-8 text-center text-[13px] text-ink-3">Nothing waiting on you.</li>}
              </ul>
            </Card>
            <Card>
              <CardHeader title={`Coming up (${upcoming.length})`} subtitle="Booked rooms from today on" />
              <ul className="divide-y divide-line">
                {upcoming.map((b) => (
                  <li key={b.id} className="flex items-center gap-4 px-5 py-2.5 text-[13px]">
                    <span className="w-28 shrink-0 tabular-nums text-ink-2">
                      {b.day} {b.request.date.slice(5)} · {b.request.start}
                    </span>
                    <span className="min-w-0 flex-1 truncate">
                      {b.request.club} · {b.request.activity}
                    </span>
                    <span className="shrink-0 text-ink-3">
                      {b.room} · {b.request.attendees}
                    </span>
                  </li>
                ))}
                {!upcoming.length && <li className="px-5 py-8 text-center text-[13px] text-ink-3">No bookings coming up.</li>}
              </ul>
            </Card>
            {past.length > 0 && (
              <Collapse title={`Past and declined (${past.length})`}>
                <ul className="divide-y divide-line">
                  {past.map((b) => (
                    <li key={b.id} className="flex items-center gap-3 py-2 text-[13px]">
                      <span className="min-w-0 flex-1 truncate">
                        {b.day} {b.request.date} · {b.request.club} · {b.request.activity}
                      </span>
                      <Badge tone={STATUS[b.status].tone}>{STATUS[b.status].label}</Badge>
                    </li>
                  ))}
                </ul>
              </Collapse>
            )}
          </div>
          <div className="space-y-6">
            <Collapse title="Book a room for a club">
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <Label>Club</Label>
              <input value={req.club} onChange={(e) => set({ club: e.target.value })} placeholder="Robotics Club" className={input} />
            </div>
            <div>
              <Label>Activity</Label>
              <input value={req.activity} onChange={(e) => set({ activity: e.target.value })} placeholder="Weekly build night" className={input} />
            </div>
            <div>
              <Label>Date</Label>
              <input type="date" value={req.date} onChange={(e) => set({ date: e.target.value })} className={input} />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <Label>From</Label>
                <input type="time" value={req.start} onChange={(e) => set({ start: e.target.value })} className={input} />
              </div>
              <div>
                <Label>To</Label>
                <input type="time" value={req.end} onChange={(e) => set({ end: e.target.value })} className={input} />
              </div>
            </div>
            <div>
              <Label>People expected</Label>
              <input type="number" min={1} value={req.attendees} onChange={(e) => set({ attendees: Number(e.target.value) })} className={input} />
            </div>
            <div>
              <Label>Room (optional)</Label>
              <select value={req.room ?? ""} onChange={(e) => set({ room: e.target.value || null })} className={input}>
                <option value="">Pick one for me</option>
                {rooms.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.id} · seats {r.capacity}
                  </option>
                ))}
                <option value="Auditorium">Auditorium (office approval)</option>
              </select>
            </div>
          </div>
          <Label className="mt-4">Members come from (their classes are checked)</Label>
          <div className="max-h-40 overflow-y-auto rounded-md border border-line p-2">
            {(pub?.cohorts ?? []).map((c) => (
              <label key={c.id} className="flex items-center gap-2 py-0.5 text-[13px]">
                <input
                  type="checkbox"
                  checked={req.cohorts.includes(c.id)}
                  onChange={(e) => set({ cohorts: e.target.checked ? [...req.cohorts, c.id] : req.cohorts.filter((x) => x !== c.id) })}
                />
                {c.name}
              </label>
            ))}
          </div>
          <div className="mt-4 flex gap-2">
            <Button loading={busy === "check"} disabled={!req.club || !req.activity} onClick={() => run(false)}>
              Check
            </Button>
            <Button variant="primary" loading={busy === "book"} disabled={!req.club || !req.activity} onClick={() => run(true)}>
              Request booking
            </Button>
          </div>
          {error && <p className="mt-3 text-[13px] text-bad">{error}</p>}
            </Collapse>
          {result ? (
            <Card>
              <CardHeader
                title={
                  <span className="flex items-center gap-2">
                    {result.request.club}: {result.day} {result.request.date}, {result.request.start}–{result.request.end}
                  </span>
                }
                subtitle={result.week ? `Week ${result.week} of the semester` : "Outside the semester weeks"}
                action={<Badge tone={STATUS[result.status].tone}>{STATUS[result.status].label}</Badge>}
              />
              <ul className="divide-y divide-line">
                {result.checks.map((c) => (
                  <li key={c.name} className="flex gap-3 px-5 py-2.5 text-[13.5px]">
                    {c.ok ? <Check size={16} className="mt-0.5 shrink-0 text-ok" /> : <X size={16} className="mt-0.5 shrink-0 text-bad" />}
                    <span>
                      <span className="font-medium">{c.name}</span> <span className="text-ink-2">· {c.detail}</span>
                    </span>
                  </li>
                ))}
              </ul>
              {result.alternatives.length > 0 && (
                <div className="border-t border-line px-5 py-3">
                  <Label>These would work</Label>
                  <div className="flex flex-wrap gap-2">
                    {result.alternatives.map((a, i) => (
                      <button
                        key={i}
                        onClick={() => set({ date: a.date, start: a.start, end: a.end, room: a.room })}
                        className="rounded-md border border-line px-2.5 py-1 text-[12.5px] hover:border-brand/60"
                      >
                        {a.day} {a.date.slice(5)} · {a.start}–{a.end} · {a.room}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {result.rules.length > 0 && (
                <div className="border-t border-line px-5 py-3 text-[12.5px] text-ink-3">
                  Booking rules: {result.rules.map((r) => `${r.title} (${r.source})`).join("; ")}
                </div>
              )}
            </Card>
          ) : (
            <Card className="p-6 text-[13.5px] text-ink-3">
              Checks: club hours ({pub?.calendar.club_hours.map((w) => `${w.days.join(", ")} ${w.start}–${w.end}`).join("; ")}), at least two days' notice (five working
              days for the auditorium), the members' programmes have no class then, and the room is a free lecture room big enough. Events over 150 people go to the office.
            </Card>
          )}
          </div>
        </div>
        <Toast message={toast} onDone={() => setToast(null)} />
      </>
    );

  return (
    <>
      <PageHeader title="Club activities" subtitle="Evenings and the free Wednesday and Saturday afternoons. Every request is checked automatically against the timetable." />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card className="h-fit p-5">
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <Label>Club</Label>
              <input value={req.club} onChange={(e) => set({ club: e.target.value })} placeholder="Robotics Club" className={input} />
            </div>
            <div>
              <Label>Activity</Label>
              <input value={req.activity} onChange={(e) => set({ activity: e.target.value })} placeholder="Weekly build night" className={input} />
            </div>
            <div>
              <Label>Date</Label>
              <input type="date" value={req.date} onChange={(e) => set({ date: e.target.value })} className={input} />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <Label>From</Label>
                <input type="time" value={req.start} onChange={(e) => set({ start: e.target.value })} className={input} />
              </div>
              <div>
                <Label>To</Label>
                <input type="time" value={req.end} onChange={(e) => set({ end: e.target.value })} className={input} />
              </div>
            </div>
            <div>
              <Label>People expected</Label>
              <input type="number" min={1} value={req.attendees} onChange={(e) => set({ attendees: Number(e.target.value) })} className={input} />
            </div>
            <div>
              <Label>Room (optional)</Label>
              <select value={req.room ?? ""} onChange={(e) => set({ room: e.target.value || null })} className={input}>
                <option value="">Pick one for me</option>
                {rooms.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.id} · seats {r.capacity}
                  </option>
                ))}
                <option value="Auditorium">Auditorium (office approval)</option>
              </select>
            </div>
          </div>
          <Label className="mt-4">Members come from (their classes are checked)</Label>
          <div className="max-h-40 overflow-y-auto rounded-md border border-line p-2">
            {(pub?.cohorts ?? []).map((c) => (
              <label key={c.id} className="flex items-center gap-2 py-0.5 text-[13px]">
                <input
                  type="checkbox"
                  checked={req.cohorts.includes(c.id)}
                  onChange={(e) => set({ cohorts: e.target.checked ? [...req.cohorts, c.id] : req.cohorts.filter((x) => x !== c.id) })}
                />
                {c.name}
              </label>
            ))}
          </div>
          <div className="mt-4 flex gap-2">
            <Button loading={busy === "check"} disabled={!req.club || !req.activity} onClick={() => run(false)}>
              Check
            </Button>
            <Button variant="primary" loading={busy === "book"} disabled={!req.club || !req.activity} onClick={() => run(true)}>
              Request booking
            </Button>
          </div>
          {error && <p className="mt-3 text-[13px] text-bad">{error}</p>}
        </Card>

        <div className="space-y-6">
          {result ? (
            <Card>
              <CardHeader
                title={
                  <span className="flex items-center gap-2">
                    {result.request.club}: {result.day} {result.request.date}, {result.request.start}–{result.request.end}
                  </span>
                }
                subtitle={result.week ? `Week ${result.week} of the semester` : "Outside the semester weeks"}
                action={<Badge tone={STATUS[result.status].tone}>{STATUS[result.status].label}</Badge>}
              />
              <ul className="divide-y divide-line">
                {result.checks.map((c) => (
                  <li key={c.name} className="flex gap-3 px-5 py-2.5 text-[13.5px]">
                    {c.ok ? <Check size={16} className="mt-0.5 shrink-0 text-ok" /> : <X size={16} className="mt-0.5 shrink-0 text-bad" />}
                    <span>
                      <span className="font-medium">{c.name}</span> <span className="text-ink-2">· {c.detail}</span>
                    </span>
                  </li>
                ))}
              </ul>
              {result.alternatives.length > 0 && (
                <div className="border-t border-line px-5 py-3">
                  <Label>These would work</Label>
                  <div className="flex flex-wrap gap-2">
                    {result.alternatives.map((a, i) => (
                      <button
                        key={i}
                        onClick={() => set({ date: a.date, start: a.start, end: a.end, room: a.room })}
                        className="rounded-md border border-line px-2.5 py-1 text-[12.5px] hover:border-brand/60"
                      >
                        {a.day} {a.date.slice(5)} · {a.start}–{a.end} · {a.room}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {result.rules.length > 0 && (
                <div className="border-t border-line px-5 py-3 text-[12.5px] text-ink-3">
                  Booking rules: {result.rules.map((r) => `${r.title} (${r.source})`).join("; ")}
                </div>
              )}
            </Card>
          ) : (
            <Card className="p-6 text-[13.5px] text-ink-3">
              Checks: club hours ({pub?.calendar.club_hours.map((w) => `${w.days.join(", ")} ${w.start}–${w.end}`).join("; ")}), at least two days' notice (five working
              days for the auditorium), the members' programmes have no class then, and the room is a free lecture room big enough. Events over 150 people go to the office.
            </Card>
          )}
          <Card>
            <CardHeader title={office ? "All requests" : "Your requests"} />
            <ul className="divide-y divide-line">
              {(bookings ?? []).map((b) => (
                <li key={b.id} className="flex flex-wrap items-center gap-3 px-5 py-3">
                  <div className="min-w-0 flex-1">
                    <p className="text-[13.5px]">
                      {b.request.club} · {b.request.activity}
                    </p>
                    <p className="text-[12px] text-ink-3">
                      {b.day} {b.request.date} {b.request.start}–{b.request.end} · {b.room ?? "no room"} · {b.request.attendees} people
                    </p>
                  </div>
                  <Badge tone={STATUS[b.status].tone}>{STATUS[b.status].label}</Badge>
                  {office && b.status === "needs_approval" && (
                    <span className="flex gap-1.5">
                      <Button size="sm" variant="primary" onClick={async () => { await api.decideClub(b.id, true); await refresh(); }}>
                        Approve
                      </Button>
                      <Button size="sm" onClick={async () => { await api.decideClub(b.id, false); await refresh(); }}>
                        Decline
                      </Button>
                    </span>
                  )}
                </li>
              ))}
              {!bookings?.length && <li className={clsx("px-5 py-8 text-center text-[13px] text-ink-3")}>No requests yet.</li>}
            </ul>
          </Card>
        </div>
      </div>
      <Toast message={toast} onDone={() => setToast(null)} />
    </>
  );
}
