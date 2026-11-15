import clsx from "clsx";
import { Info, Plus, Search } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ConversationPane, useConversation } from "../components/ConversationPane";
import { ConversationList, useConversations, type Conversation } from "../components/Conversations";
import { DetailsDrawer } from "../components/DetailsDrawer";
import { Button, Card } from "../components/ui";
import { OUTCOME_DOT, OUTCOME_TEXT } from "../lib/meta";

/** Faculty and class representatives: their requests and what colleagues' requests asked of them, each
 * read as a chat; the list on the left, the open conversation on the right. /requests/R-00025 opens one.
 * How the system handled a request (the pipeline view) opens from "Details" in a panel over the chat. */
/** A change to this person's timetable made by someone else's request: nothing of theirs, nothing to answer. */
const isChange = (c: Conversation) => !c.own && !!c.changeOnly && !c.needs;

export default function RequestChat() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { convs, refresh } = useConversations();
  const s = useConversation(convs, refresh, { autoOpen: !id });
  const [q, setQ] = useState("");
  const [details, setDetails] = useState<string | null>(null); // the request whose details are open
  // two lists: the person's own requests (and anything waiting on them), and changes to their timetable
  const [tab, setTab] = useState<"requests" | "changes">("requests");
  const selectedConv = convs.find((c) => c.key === s.selected);
  useEffect(() => {
    if (selectedConv) setTab(isChange(selectedConv) ? "changes" : "requests"); // the list shows what is open
  }, [selectedConv?.key]); // eslint-disable-line react-hooks/exhaustive-deps
  const shown = convs.filter((c) => (tab === "changes") === isChange(c));
  const unread = (t: "requests" | "changes") => convs.filter((c) => c.unread && !c.partOf && (t === "changes") === isChange(c)).length;

  // /requests/R-00025: open that request, or the colleague's request it is about
  useEffect(() => {
    if (!id || !convs.length) return;
    const key = convs.find((c) => c.key === id)?.key ?? convs.find((c) => c.key === `other:${id}`)?.key;
    if (key && key !== s.selected) s.open(key);
  }, [id, convs.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const open = (key: string | null) => {
    s.open(key);
    const c = convs.find((x) => x.key === key);
    navigate(c?.case ? `/requests/${c.case}` : "/requests", { replace: true });
  };

  return (
    <div className="grid gap-4 lg:h-full lg:grid-cols-[minmax(260px,320px)_minmax(0,1fr)]">
      <Card className="flex max-h-[420px] min-h-0 flex-col lg:max-h-none lg:h-full">
        <div className="flex items-center justify-between gap-2 border-b border-line px-4 py-3">
          <h1 className="text-[15px] font-semibold">Requests</h1>
          <Button size="sm" variant="primary" icon={Plus} onClick={() => open(null)}>
            New
          </Button>
        </div>
        <div className="flex gap-1 border-b border-line px-3 py-2" role="tablist">
          {(["requests", "changes"] as const).map((t) => (
            <button
              key={t}
              role="tab"
              aria-selected={tab === t}
              onClick={() => setTab(t)}
              className={clsx(
                "flex flex-1 items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-[12.5px] font-medium transition-colors",
                tab === t ? "bg-panel-2 text-ink" : "text-ink-3 hover:text-ink-2",
              )}
            >
              {t === "requests" ? "Your requests" : "Timetable changes"}
              {unread(t) > 0 && <span className="rounded-full bg-brand/15 px-1.5 text-[11px] text-brand">{unread(t)}</span>}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2 border-b border-line px-3">
          <Search size={14} className="text-ink-3" />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder={tab === "changes" ? "Find a change" : "Find a request or message"} className="h-9 flex-1 bg-transparent text-[13px] outline-none placeholder:text-ink-3" />
        </div>
        <div className="flex-1 overflow-y-auto py-1">
          {shown.length ? (
            <ConversationList items={shown} selected={s.selected} onSelect={open} q={q} />
          ) : (
            <p className="px-4 py-8 text-center text-[13px] text-ink-3">
              {tab === "changes" ? "No changes to your timetable yet." : "No requests yet. Write one with New."}
            </p>
          )}
        </div>
      </Card>

      <Card className="flex h-[640px] min-h-0 flex-col lg:h-full">
        <div className="flex items-center gap-3 border-b border-line px-4 py-3">
          <div className="min-w-0 flex-1">
            <p className="truncate text-[14px] font-semibold">{s.conv ? s.conv.title : "New request"}</p>
            {s.conv &&
              (s.conv.own && s.conv.status ? (
                <p className={clsx("flex items-center gap-1.5 truncate text-[12px] font-medium", OUTCOME_TEXT[s.conv.outcome ?? "progress"])}>
                  <span className={clsx("size-2 shrink-0 rounded-full", OUTCOME_DOT[s.conv.outcome ?? "progress"])} />
                  {s.conv.latest}
                </p>
              ) : (
                <p className="truncate text-[12px] text-ink-3">{s.conv.own ? s.conv.latest : s.conv.changeOnly ? `Change to your timetable · ${s.conv.latest}` : `A colleague's request · ${s.conv.latest}`}</p>
              ))}
          </div>
          {s.caseId && (
            <Button size="sm" icon={Info} onClick={() => setDetails(s.caseId)} title="How the system handled this request">
              Details
            </Button>
          )}
        </div>
        <ConversationPane s={s} convs={convs} wide />
      </Card>
      <DetailsDrawer id={details} onClose={() => setDetails(null)} />
    </div>
  );
}
