import { Lock } from "lucide-react";
import { lazy, Suspense, type ReactNode } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation, useSearchParams } from "react-router-dom";
import { Shell } from "./components/Shell";
import { EmptyState, Mark, Skeleton } from "./components/ui";
import { AuthProvider, useAuth } from "./lib/auth";
import { REQUESTS_AS_CHATS, WORKSPACE_HOME } from "./lib/meta";
import { ThemeProvider } from "./lib/theme";
import type { Role, View } from "./lib/types";

const Login = lazy(() => import("./pages/Login"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Home = lazy(() => import("./pages/Home"));
const Requests = lazy(() => import("./pages/Requests"));
const RequestChat = lazy(() => import("./pages/RequestChat"));
const RequestsTable = lazy(() => import("./pages/RequestsTable"));
const CaseDetail = lazy(() => import("./pages/CaseDetail"));
const Inbox = lazy(() => import("./pages/Inbox"));
const Timetable = lazy(() => import("./pages/Timetable"));
const Fairness = lazy(() => import("./pages/Fairness"));
const Graph = lazy(() => import("./pages/Graph"));
const Policy = lazy(() => import("./pages/Policy"));
const SettingsPage = lazy(() => import("./pages/Settings"));
const SemesterBuild = lazy(() => import("./pages/SemesterBuild"));
const SemesterPreferences = lazy(() => import("./pages/SemesterPreferences"));
const Clubs = lazy(() => import("./pages/Clubs"));
const Transparency = lazy(() => import("./pages/Transparency"));
const Observability = lazy(() => import("./pages/Observability"));
const Experiments = lazy(() => import("./pages/Experiments"));
const Study = lazy(() => import("./pages/Study"));
const HumanStudy = lazy(() => import("./pages/HumanStudy"));

function Splash() {
  return (
    <div className="grid h-full place-items-center">
      <Mark size={36} className="animate-pulse" />
    </div>
  );
}

function PageFallback() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-9 w-64" />
      <Skeleton className="h-64" />
    </div>
  );
}

function Guard({ view, children }: { view: View; children: ReactNode }) {
  const { sees } = useAuth();
  if (!sees(view)) return <EmptyState icon={Lock} title="Not available for your role" text="Your role cannot open this page." />;
  return <>{children}</>;
}

/** The home page: the request workspace for the roles that live in the timetable (WORKSPACE_HOME: teachers, the
 * HoD, class representatives, the lab in-charge and the exam cell); the dashboard, a queue of what waits on them,
 * for the timetable office and the Dean (their workspace is New request, /new). */
function HomePage() {
  const { user } = useAuth();
  return WORKSPACE_HOME.includes(user?.role as Role) ? <Home /> : <Dashboard />;
}

/** For faculty and class representatives, requests are conversations (their own screen, read as a chat);
 * the pipeline view of one (parse, policy, solver, fairness) is behind "Technical details" (?details=1).
 * Everyone else gets the request list and the detailed page. */
function RequestsPage({ one = false }: { one?: boolean }) {
  const { user } = useAuth();
  const [params] = useSearchParams();
  // the list as a table (quick actions); one request as a conversation
  if (REQUESTS_AS_CHATS.includes(user?.role as Role) && !params.has("details")) return one ? <RequestChat /> : <RequestsTable />;
  return one ? <CaseDetail /> : <Requests />;
}

/** Teachers and class representatives send their own semester preferences here; the timetable office
 * sees everyone's, in the semester plan's Preferences tab. */
function PreferencesPage() {
  const { user } = useAuth();
  return user?.role === "coordinator" ? <Navigate to="/semester?tab=preferences" replace /> : <SemesterPreferences />;
}

function Routed() {
  const { user, ready } = useAuth();
  const { pathname } = useLocation();
  if (!ready) return <Splash />;
  if (pathname.startsWith("/study"))
    // participants enter by code; no portal sign-in needed for the labelling tasks
    return (
      <Suspense fallback={<Splash />}>
        <Study />
      </Suspense>
    );
  if (!user)
    return (
      <Suspense fallback={<Splash />}>
        <Login />
      </Suspense>
    );
  return (
    <Shell>
      <Suspense fallback={<PageFallback />}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/new" element={<Guard view="new"><Home /></Guard>} /> {/* the timetable and the chat beside it, for every role */}
          <Route path="/requests" element={<RequestsPage />} />
          <Route path="/requests/:id" element={<RequestsPage one />} />
          <Route path="/inbox" element={<Guard view="inbox"><Inbox /></Guard>} />
          <Route path="/approvals" element={<Navigate to="/inbox?view=needs" replace />} /> {/* approvals are in the inbox now */}
          <Route path="/timetable" element={<Timetable />} />
          <Route path="/fairness" element={<Guard view="fairness"><Fairness /></Guard>} />
          <Route path="/policy" element={<Policy />} />
          <Route path="/documents" element={<Navigate to="/policy?tab=documents" replace />} /> {/* documents are a tab of the handbook */}
          <Route path="/settings" element={<SettingsPage />} /> {/* each person's own choices; the coordinator's section inside */}
          <Route path="/semester" element={<Guard view="semester"><SemesterBuild /></Guard>} />
          <Route path="/preferences" element={<Guard view="prefs"><PreferencesPage /></Guard>} />
          <Route path="/clubs" element={<Guard view="clubs"><Clubs /></Guard>} />
          <Route path="/transparency" element={<Transparency />} />
          <Route path="/graph" element={<Guard view="graph"><Graph /></Guard>} />
          <Route path="/observability" element={<Guard view="health"><Observability /></Guard>} />
          <Route path="/experiments" element={<Guard view="experiments"><Experiments /></Guard>} />
          <Route path="/human-study" element={<Guard view="study"><HumanStudy /></Guard>} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </Shell>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <Routed />
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  );
}
