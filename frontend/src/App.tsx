import { Lock } from "lucide-react";
import { lazy, Suspense, type ReactNode } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { Shell } from "./components/Shell";
import { EmptyState, Mark, Skeleton } from "./components/ui";
import { AuthProvider, useAuth } from "./lib/auth";
import { TIMETABLE_HOME } from "./lib/meta";
import { ThemeProvider } from "./lib/theme";
import type { Role, View } from "./lib/types";

const Login = lazy(() => import("./pages/Login"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Home = lazy(() => import("./pages/Home"));
const NewRequest = lazy(() => import("./pages/NewRequest"));
const Requests = lazy(() => import("./pages/Requests"));
const CaseDetail = lazy(() => import("./pages/CaseDetail"));
const Inbox = lazy(() => import("./pages/Inbox"));
const Approvals = lazy(() => import("./pages/Approvals"));
const Timetable = lazy(() => import("./pages/Timetable"));
const Fairness = lazy(() => import("./pages/Fairness"));
const Graph = lazy(() => import("./pages/Graph"));
const Policy = lazy(() => import("./pages/Policy"));
const PolicyDocuments = lazy(() => import("./pages/PolicyDocuments"));
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

function HomePage() {
  const { user } = useAuth();
  return TIMETABLE_HOME.includes(user?.role as Role) ? <Home /> : <Dashboard />;
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
          <Route path="/new" element={<Guard view="new"><NewRequest /></Guard>} />
          <Route path="/requests" element={<Requests />} />
          <Route path="/requests/:id" element={<CaseDetail />} />
          <Route path="/inbox" element={<Guard view="inbox"><Inbox /></Guard>} />
          <Route path="/approvals" element={<Guard view="approvals"><Approvals /></Guard>} />
          <Route path="/timetable" element={<Timetable />} />
          <Route path="/fairness" element={<Guard view="fairness"><Fairness /></Guard>} />
          <Route path="/policy" element={<Policy />} />
          <Route path="/documents" element={<Guard view="documents"><PolicyDocuments /></Guard>} />
          <Route path="/semester" element={<Guard view="semester"><SemesterBuild /></Guard>} />
          <Route path="/preferences" element={<Guard view="prefs"><SemesterPreferences /></Guard>} />
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
