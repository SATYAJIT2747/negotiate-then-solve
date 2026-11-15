import CaseDetail from "../pages/CaseDetail";
import { SidePanel } from "./ui";

/** How the system handled a request (lifecycle, parse, rules, solver, negotiation, fairness), in a panel
 * that slides over the conversation instead of a page of its own. */
export function DetailsDrawer({ id, onClose }: { id: string | null; onClose: () => void }) {
  return (
    <SidePanel
      open={!!id}
      onClose={onClose}
      title="Details"
      subtitle={`How the system handled ${id ?? ""}: the rules it checked, how it was solved and negotiated`}
    >
      {id && <CaseDetail id={id} embedded />}
    </SidePanel>
  );
}
