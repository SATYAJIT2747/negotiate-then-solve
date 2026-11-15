import { Link } from "react-router-dom";
import { api } from "../../lib/api";
import { useApi } from "../../lib/hooks";
import { SidePanel, Skeleton } from "../ui";

/** A rule a chat answer cites, opened beside the chat: its words and where they come from, without leaving
 * the conversation. ``id``: the rule's id or its number ("P-MAXCONSEC", "2.3"). */
export function RulePanel({ id, onClose }: { id: string | null; onClose: () => void }) {
  const { data: rules } = useApi(() => (id ? api.handbook() : Promise.resolve(null)), [!!id]);
  const rule = id ? rules?.find((r) => r.id === id || r.number === id) : undefined;
  return (
    <SidePanel open={!!id} onClose={onClose} width={560} title={rule ? `§${rule.number} ${rule.title}` : id ?? ""} subtitle={rule ? `${rule.source}${rule.pages.length ? `, page ${rule.pages.join(", ")}` : ""}` : "From the handbook"}>
      {!rules ? (
        <Skeleton className="h-40" />
      ) : rule ? (
        <div className="space-y-4">
          <p className="font-mono text-[12px] text-ink-3">{rule.id}</p>
          <p className="whitespace-pre-wrap text-[14px] leading-relaxed">{rule.text}</p>
          <Link to="/policy" className="text-[13px] text-brand hover:underline">
            Open the handbook
          </Link>
        </div>
      ) : (
        <p className="text-[13.5px] text-ink-3">{id} is not in the handbook as it stands now.</p>
      )}
    </SidePanel>
  );
}
