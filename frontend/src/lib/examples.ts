import type { InstanceView, Persona } from "./types";

/** The chat's examples, in three kinds (CHAT-PLAN): questions about the timetable, questions about the rules, and
 * requests. Written from the person's own classes and colleagues where it helps; a click puts one in the box. */
export function chatExamples(user: Persona | null, inst: InstanceView | null | undefined): { title: string; items: string[] }[] {
  if (!user || !inst) return [];
  const rep = user.role === "student";
  const colleague = inst.faculty.find((f) => f.id !== user.id && inst.sessions.some((s) => s.faculty === f.id))?.name ?? "Dr. Iyer";
  const lab = inst.rooms.find((r) => r.type === "lab")?.name ?? "Lab 2";
  const timetable = rep
    ? ["What are our free slots this week?", `When is ${colleague} free on Friday?`, `Is ${lab} free on Thursday at 3 pm?`, "What changed this week?"]
    : ["What are my free slots this week?", `When are ${colleague} and I both free?`, `Is ${lab} free on Thursday at 3 pm?`, "When is my next class?"];
  const rules = ["How many online classes can a course have?", "Is it allowed to teach more than three hours in a row?",
                 "Within how many weeks must a missed class be made up?"];
  const requests = requestExamples(user, inst).filter((x) => !rules.includes(x) && !/^is it allowed/i.test(x)).slice(0, 3);
  return [
    { title: "Timetable", items: timetable },
    { title: "Rules", items: rules },
    ...(requests.length ? [{ title: "Requests", items: requests }] : []),
  ];
}

/** Requests a person of this role might send, written from their own classes: a click puts one in the box. */
export function requestExamples(user: Persona | null, inst: InstanceView | null | undefined): string[] {
  if (!user || !inst) return [];
  const practical = inst.sessions.find((s) => s.faculty === user.id && s.kind === "practical");
  if (user.role === "student") return ["Machine Learning and Algorithms clash on Tuesday at 10 am for 12 of us. Can this be looked at?", "Please cancel Dr. Khan's Friday lecture, we have a quiz."];
  if (user.role === "lab_incharge") return ["Lab 1 is closed for maintenance in weeks 11-12.", "Lab 2 will be unavailable in week 6 while the wiring is replaced."];
  if (user.role === "exam_cell") return ["Hall A is closed for mid-semester exams in week 8.", "Hall B is needed for exams in week 15."];
  if (user.role === "coordinator") return ["Who has to approve moving a class outside regular hours?"];
  return [
    "I'm at a conference in week 10, Wednesday and Thursday, so I can't take my classes then.",
    "I'd prefer no classes before 11 am on Friday, if possible.",
    practical ? `My ${practical.title} practical needs the routers; it has to be on Wednesday afternoon.` : "Please keep my Friday afternoons free.",
    "Could you schedule my lecture at 1 pm on Thursday? It's the only time that works.",
    "Is it allowed to teach more than three hours in a row?",
  ];
}
