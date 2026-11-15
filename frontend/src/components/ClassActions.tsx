import { ArrowLeftRight, CalendarClock, PencilLine, Send, X } from "lucide-react";
import type { ReactNode } from "react";
import { ready, type ActionKind, type Draft } from "../lib/draft";
import type { InstanceView, TimetableEntry } from "../lib/types";
import { segment, WhenField } from "./draft/WhenField";
import { Button, Popover } from "./ui";

/** Requests made by clicking instead of typing: cancel a class, move it, or swap it with a colleague's.
 * The card sends the request as data (``intent``, read by no parser) together with the same request in
 * words (``sentence``), which is what people see in the conversation. */

/** What can be done with a class, at the place it was clicked. */
export function ClassMenu({ at, cls, own, onPick, onClose }: {
  at: { x: number; y: number };
  cls: TimetableEntry;
  /** The person teaches it: they may cancel, move or swap it. */
  own: boolean;
  onPick: (kind: ActionKind | "write") => void;
  onClose: () => void;
}) {
  const item = (kind: ActionKind | "write", icon: ReactNode, label: string) => (
    <button onClick={() => onPick(kind)} className="flex w-full items-center gap-2.5 rounded px-2.5 py-1.5 text-left text-[13px] hover:bg-panel-2">
      {icon}
      {label}
    </button>
  );
  return (
    <Popover at={at} onClose={onClose} width={240} height={own ? 200 : 100}>
      <div className="px-2.5 pb-1.5 pt-1">
        <p className="truncate text-[13px] font-semibold">
          {cls.title} <span className="font-normal text-ink-3">{cls.kind}</span>
        </p>
        <p className="truncate text-[11.5px] text-ink-3">
          {cls.room_name} · {cls.group_names.join(", ")}
        </p>
      </div>
      {own && (
        <div className="border-t border-line pt-1">
          {item("cancel", <X size={14} className="text-bad" />, "Request cancellation")}
          {item("move", <CalendarClock size={14} className="text-brand" />, "Request rescheduling")}
          {item("swap", <ArrowLeftRight size={14} className="text-info" />, "Swap with a colleague")}
        </div>
      )}
      <div className="mt-1 border-t border-line pt-1">{item("write", <PencilLine size={14} className="text-ink-3" />, "Write about this class")}</div>
    </Popover>
  );
}

/** The card in the panel while a click-made request is being put together. */
export function ActionCard({ d, set, inst, week, weeks, current, sending, error, onCancel, onSend }: {
  d: Draft;
  set: (d: Draft) => void;
  inst: InstanceView;
  week: number;
  weeks: number;
  /** The teaching week now: earlier weeks have passed and cannot be asked for. */
  current: number;
  sending: boolean;
  error: string | null;
  onCancel: () => void;
  onSend: () => void;
}) {
  const day = (x: string) => inst.day_names[x] ?? x;
  const time = (s: number) => inst.slots.find((x) => x.slot === s)?.label ?? `slot ${s}`;
  const title = { cancel: "Cancel a class", move: "Reschedule a class", swap: "Swap with a colleague" }[d.kind];
  const t = d.to;

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        <div>
          <p className="text-[14px] font-semibold">{title}</p>
          <p className="mt-0.5 text-[12.5px] text-ink-3">
            {d.cls.title} {d.cls.kind} · {day(d.cls.day)} {time(d.cls.slot)} · {d.cls.group_names.join(", ")}
          </p>
        </div>

        {d.kind === "move" && (
          <Field label="To">
            {t?.slot != null && t.day ? (
              <p className="text-[13px]">
                {day(t.day)} {time(t.slot)}{" "}
                <button onClick={() => set({ ...d, to: null })} className="ml-1 text-[12px] text-brand hover:underline">
                  change
                </button>
              </p>
            ) : (
              <div className="space-y-2">
                <p className="text-[12.5px] text-ink-2">Click a green slot on the timetable, or:</p>
                <div className="flex flex-wrap gap-1.5">
                  {(["morning", "afternoon"] as const).map((p) => (
                    <button key={p} onClick={() => set({ ...d, to: { ...t, part: t?.part === p ? undefined : p, slot: undefined } })} className={segment(t?.part === p)}>
                      Any {p}
                    </button>
                  ))}
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {inst.calendar.days.map((x) => (
                    <button key={x} onClick={() => set({ ...d, to: { ...t, day: t?.day === x ? undefined : x, slot: undefined } })} className={segment(t?.day === x)}>
                      {x}
                    </button>
                  ))}
                </div>
                <p className="text-[11.5px] text-ink-3">A part of the day, a day, or both.</p>
              </div>
            )}
          </Field>
        )}

        {d.kind === "swap" && (
          <Field label="You take">
            {d.theirs ? (
              <p className="text-[13px]">
                {d.theirs.faculty_name}'s {d.theirs.title} {d.theirs.kind} · {day(d.theirs.day)} {time(d.theirs.slot)}{" "}
                <button onClick={() => set({ ...d, theirs: null })} className="ml-1 text-[12px] text-brand hover:underline">
                  change
                </button>
              </p>
            ) : (
              <p className="text-[12.5px] text-ink-2">Click a colleague's class on the timetable. Classes that cannot be swapped are faded.</p>
            )}
          </Field>
        )}

        <Field label="When">
          <WhenField kind={d.kind} value={d.when} onChange={(when) => set({ ...d, when })} week={week} weeks={weeks} current={current} />
        </Field>

        {d.kind === "move" && (
          <Field label="How">
            <div className="flex flex-wrap gap-1.5">
              <button onClick={() => set({ ...d, hard: true })} className={segment(d.hard)}>
                Must be then
              </button>
              <button onClick={() => set({ ...d, hard: false })} className={segment(!d.hard)}>
                If possible
              </button>
            </div>
            {!d.hard && <p className="mt-1.5 text-[11.5px] text-ink-3">Only if no other class has to move for it, so it may not happen.</p>}
          </Field>
        )}

        <Field label="Reason (optional)">
          <input
            value={d.reason}
            onChange={(e) => set({ ...d, reason: e.target.value })}
            placeholder={d.kind === "cancel" ? "e.g. conference, leave" : "e.g. exam duty"}
            className="h-8 w-full rounded-md border border-line bg-panel px-2.5 text-[13px] outline-none focus:border-brand/60"
          />
          <p className="mt-1 text-[11.5px] text-ink-3">Kept private: only the fact that a reason was given is used.</p>
        </Field>

        {d.kind === "cancel" && <Note>A cancelled class needs a make-up class within two weeks.</Note>}
        {d.kind === "swap" && <Note>Your colleague will be asked to agree, and the timetable office is told before the swap takes effect.</Note>}
        {d.kind === "move" && <Note>Green slots are free for you and your section in the published timetable. The final check is the solver's.</Note>}
      </div>

      <div className="border-t border-line p-3">
        {error && <p className="mb-2 text-[12.5px] text-bad">{error}</p>}
        <div className="flex justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant="primary" size="sm" icon={Send} loading={sending} disabled={!ready(d)} onClick={onSend}>
            Send request
          </Button>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-ink-3">{label}</p>
      {children}
    </div>
  );
}

function Note({ children }: { children: ReactNode }) {
  return <p className="rounded-md bg-panel-2/70 px-3 py-2 text-[12px] text-ink-2">{children}</p>;
}
