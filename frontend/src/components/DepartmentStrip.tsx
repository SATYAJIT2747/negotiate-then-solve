import { ArrowRight, CheckCircle2, Scale, TriangleAlert } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth";
import { useApi } from "../lib/hooks";
import { STATUS } from "../lib/meta";
import type { CaseSummary } from "../lib/types";
import { Card } from "./ui";

/** The head of department's home is a teacher's home with their department on top: the escalations that
 * wait on their decision first (a teaching-need clash among their teachers is sent to them by name or as
 * "hod"), then how much of the department's business is still open, with the fairness ledger a click away. */
export function DepartmentStrip() {
  const { user } = useAuth();
  const { data: cases } = useApi(() => api.cases("all"), [], 5000);
  const { data: ov } = useApi(() => api.overview(), [], 10_000);
  const toMe = (c: CaseSummary) => {
    const to = c.escalated_to ?? "";
    return to === "hod" || to.split(", ").includes(user?.id ?? "");
  };
  const decide = (cases ?? []).filter((c) => c.status === "escalated" && toMe(c));
  const open = (cases ?? []).filter((c) => c.sender !== user?.id && (c.running || !STATUS[c.status]?.terminal)).length;

  return (
    <Card className="px-4 py-2.5">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 text-[13px]">
        <span className="text-[11px] font-medium uppercase tracking-wide text-ink-3">Your department</span>
        {!cases ? (
          <span className="text-ink-3">Loading…</span>
        ) : decide.length ? (
          <Link to="/inbox?view=needs" className="group flex items-center gap-1.5 font-medium text-warn">
            <TriangleAlert size={14} />
            {decide.length} decision{decide.length > 1 ? "s" : ""} waiting on you
            <ArrowRight size={13} className="transition-transform group-hover:translate-x-0.5" />
          </Link>
        ) : (
          <span className="flex items-center gap-1.5 text-ink-2">
            <CheckCircle2 size={14} className="text-ok" /> No decisions waiting on you
          </span>
        )}
        {cases && (
          <Link to="/requests" className="text-ink-2 hover:text-ink hover:underline">
            {open} request{open === 1 ? "" : "s"} from your teachers in progress
          </Link>
        )}
        <Link to="/fairness" className="flex items-center gap-1.5 text-ink-2 hover:text-ink hover:underline" title="0 is perfectly even, 1 is one person giving way every time">
          <Scale size={14} /> Giving way{ov ? `: ${ov.gini.toFixed(2)}` : ""}
        </Link>
      </div>
      {decide.length > 0 && (
        <ul className="mt-2 space-y-1 border-t border-line pt-2">
          {decide.slice(0, 3).map((c) => (
            <li key={c.id}>
              <Link to={`/requests/${c.id}?details=1`} className="flex items-center gap-2 text-[13px] hover:underline">
                <span className="shrink-0 text-ink-3">{c.sender_name}:</span>
                <span className="min-w-0 truncate">“{c.text}”</span>
                <span className="ml-auto shrink-0 font-medium text-brand">Decide</span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
