import { Plus, Search } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ConversationPane, useConversation } from "../components/ConversationPane";
import { ConversationList, useConversations } from "../components/Conversations";
import { Button, Card } from "../components/ui";

/** Faculty and class representatives: their requests and what colleagues' requests asked of them, each
 * read as a chat; the list on the left, the open conversation on the right. /requests/R-00025 opens one.
 * The pipeline view of a request is behind "Technical details" in the thread. */
export default function RequestChat() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { convs, refresh } = useConversations();
  const s = useConversation(convs, refresh, { autoOpen: !id });
  const [q, setQ] = useState("");

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
        <div className="flex items-center gap-2 border-b border-line px-3">
          <Search size={14} className="text-ink-3" />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Find a request or message" className="h-9 flex-1 bg-transparent text-[13px] outline-none placeholder:text-ink-3" />
        </div>
        <div className="flex-1 overflow-y-auto py-1">
          <ConversationList items={convs} selected={s.selected} onSelect={open} q={q} />
        </div>
      </Card>

      <Card className="flex h-[640px] min-h-0 flex-col lg:h-full">
        <div className="border-b border-line px-4 py-3">
          <p className="truncate text-[14px] font-semibold">{s.conv ? s.conv.title : "New request"}</p>
          {s.conv && <p className="truncate text-[12px] text-ink-3">{s.conv.own ? s.conv.latest : `A colleague's request · ${s.conv.latest}`}</p>}
        </div>
        <ConversationPane s={s} convs={convs} wide />
      </Card>
    </div>
  );
}
