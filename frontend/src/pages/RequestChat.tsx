import { ArrowLeft, Info } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ConversationPane, useConversation } from "../components/ConversationPane";
import { useConversations } from "../components/Conversations";
import { DetailsDrawer } from "../components/DetailsDrawer";
import { Button, Skeleton, StatusBadge } from "../components/ui";

/** One request of a faculty member or class representative, read as a conversation (/requests/R-00025): what
 * they asked, what happened, and the answer to a question back or the options, given here. The list is the
 * Requests table (pages/RequestsTable); new requests are made in the chat. How the system handled it (the
 * pipeline view) opens from "Details" in a panel. */
export default function RequestChat() {
  const { id } = useParams();
  const { convs, refresh } = useConversations();
  const s = useConversation(convs, refresh, { autoOpen: false });
  const [details, setDetails] = useState<string | null>(null);

  // open the request in the address, or the colleague's request it is about
  useEffect(() => {
    if (!id || !convs.length) return;
    const key = convs.find((c) => c.key === id)?.key ?? convs.find((c) => c.key === `other:${id}`)?.key;
    if (key && key !== s.selected) s.open(key);
  }, [id, convs.length]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="mx-auto flex w-full flex-col gap-3 lg:h-full lg:w-[88%]">
      <Link to="/requests" className="flex items-center gap-1 text-[12.5px] text-ink-3 hover:text-brand">
        <ArrowLeft size={13} /> Requests
      </Link>
      <div className="flex h-[640px] min-h-0 min-w-0 flex-col lg:h-auto lg:flex-1">
        <div className="flex items-center gap-3 border-b border-line pb-3">
          <div className="min-w-0 flex-1">
            {s.conv ? (
              <>
                <p className="truncate text-[14px] font-semibold">{s.conv.title}</p>
                <p className="flex items-center gap-2 truncate text-[12px] text-ink-3">
                  <span className="font-mono">{id}</span>
                  {s.shown && <StatusBadge status={s.shown.status} live={s.shown.running} />}
                  {!s.conv.own && <span>A colleague's request</span>}
                </p>
              </>
            ) : (
              <Skeleton className="h-9 w-72" />
            )}
          </div>
          {s.caseId && (
            <Button size="sm" icon={Info} onClick={() => setDetails(s.caseId)} title="How the system handled this request">
              Details
            </Button>
          )}
        </div>
        {convs.length > 0 && !s.conv ? (
          <p className="px-4 py-16 text-center text-[13.5px] text-ink-3">
            {id} is not among your requests. <Link to="/requests" className="text-brand hover:underline">Back to your requests</Link>
          </p>
        ) : (
          <ConversationPane s={s} convs={convs} wide answerOnly />
        )}
      </div>
      <DetailsDrawer id={details} onClose={() => setDetails(null)} />
    </div>
  );
}
