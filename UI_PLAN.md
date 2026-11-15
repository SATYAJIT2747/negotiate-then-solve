# UI plan: one chat screen

Plan for the `f-frontend` branch. Nothing here is built yet.

## Why

The web app has grown to 18 pages (Home, Inbox, Requests, New request,
Approvals, This week, Handbook, Fairness, ...). A faculty member who wants to
move a class or ask about a rule has to know which page to open, and the
answer to their request then turns up on yet another page (Inbox for a
negotiation, Requests for the status). The aim is one place to talk to the
system, one place to approve, and the rest out of the way.

## The idea

A single **chat** screen is the home page for everyone. People type anything:
a request, a policy question, or an answer to something the system asked. The
system replies in the same thread. Anything that needs the person's action
(pick one of several times, accept an option in a negotiation) appears as a
**card inside the chat**.

**Approvals are separate**: a section of its own, shown only to people who
have the permission to decide something.

```
┌──────────────────────────────────────────────────────────────┐
│  Dr. Sharma                                   [My week ▸]     │
├──────────────────────────────────────────────────────────────┤
│  You: Can I teach four hours in a row?                        │
│  ⓘ  No. At most 3 consecutive hours (§2.3 P-MAXCONSEC).       │
│                                                               │
│  You: On Thursday in week 12 move my ML class before lunch    │
│  ⏳ Understood: ML lecture (C-021-L3), Thu morning, week 12   │
│  ┌ Choose a time ──────────────────────────────────────┐      │
│  │ (A) Thu 11 am R-102  (B) Thu 12 pm R-103  (C) Thu 9 am   │
│  └─────────────────────────────────────────── [Choose] ┘      │
│  ✔ Waiting for the timetable office's approval                │
├──────────────────────────────────────────────────────────────┤
│  [ Type a request or a question…                     ] [Send]│
└──────────────────────────────────────────────────────────────┘

Sidebar:  Chat (home) · My week · Approvals (badge)* · More ▾
          * only for people who can approve or decide
```

## 1. Telling a question from a request

The classifier already exists. Every message goes through System One and the
parser (`Orchestrator.submit`), which produce an action; the chat sends every
message to the existing `POST /api/requests` and shows the result by action:

| Action | Shown in the chat as |
|---|---|
| `answer` | The policy answer, directly in the thread, with the cited rule |
| `compile` | A request with a live status: understood as … → checking → negotiating → waiting for approval → published |
| `clarify` | The question in the thread; the person's next message is the answer (`POST /api/cases/{id}/clarify`) |
| `refuse`, `out_of_scope`, `investigate` | A short reply saying why, or where the message went |

To fix or add:

- **Questions without "?"** are read as requests ("Tell me the rule about
  swapping lectures"). People type casually in a chat, so this matters more
  here. See issues #17–#20 in `RAG_ISSUES.md` (branch `f-rag`).
- **A reply or a new message?** When a card is waiting for the person, their
  next message goes to it by default (`POST /api/inbox/{id}/reply-text`
  already reads a reply in the person's own words and asks them to confirm
  the reading). A "replying to …" chip, as in messaging apps, lets them change
  that.
- **Small talk** ("hi", "thanks", "what can you do?") gets a short fixed reply
  and starts no request.

## 2. Negotiation in the chat

Inbox items are already structured messages with options, so they become
cards in the thread:

| Today | In the chat |
|---|---|
| Inbox → "A clash with your timetable" | Card: what clashes, options A/B/C, **Accept / Suggest another time / Decline**, or a typed reply |
| Inbox → "Choose a time for your request" | Card with the time options and **Choose** |
| Requests → the request's status | Status lines under the request ("Negotiating with Dr. Iyer…", "Approved, version 19") |
| Requests → the request's page | Still there for the full audit (parse, rules, conflict, events); opened by clicking a request in the chat |

**Backend:** one new endpoint, `GET /api/chat`, returning the person's thread
in time order: their messages, the system's replies and status changes, and
their inbox items. The existing `GET /api/feed` (`_feed` in `src/web/app.py`)
already gathers all of this per person and is the starting point. Updates
poll, as the app does today.

**One thread per person**, with each request's updates grouped under it and
collapsible.

## 3. Approvals section

A separate section, shown to anyone with permission for at least one kind of
decision (not only when their queue is non-empty):

| Role | In their queue | Existing endpoints |
|---|---|---|
| Timetable office / coordinator | Changes waiting to be published (approve / reject); messages forwarded to the office (reply) | `/api/approvals`, `/api/approvals/{id}/approve`, `/reject`, `/api/cases/{id}/handle` |
| Head of department / Dean | Escalations they have the authority to decide (grant / refuse) | `/api/cases/{id}/decide` |
| Faculty, students | No section | — |

- Each item opens the full view: the diff table, who is affected, the change
  in fairness, and the conflict if there was one.
- The requester sees only the outcome, in their chat: "Waiting for the
  timetable office's approval" → "Approved, version 19" or "Rejected: …".
- Someone with both roles (a head of department who also teaches) has their
  own requests and negotiations in their chat, and the decisions they owe
  others in Approvals.
- The badge uses the counts `/api/overview` already returns
  (`pending_approvals`, `my_decisions`, `my_forwarded`).

## 4. Drag and drop

A **"My week" panel**, sliding out from the chat, shows the person's own
classes on the week grid (the existing `TimetableGrid`).

1. **Pick up a class.** While it is held, every cell is coloured by a new
   `GET /api/feasibility?session=…&week=…`, built on the existing
   `free_placements()` helper used for extra classes:
   - green: free, nothing else moves;
   - amber: possible, but other classes move (how many);
   - red: blocked by a rule (lunch hour, out of hours) or the room or section
     is busy, with the reason on hover.
2. **Drop it.** A small popup asks: this week only, or every week? Required,
   or a preference?
3. **Send.** It posts a structured request to a new
   `POST /api/requests/structured` (session, day, slot, weeks, hard or soft).
   This skips the parser, so nothing can be misread; authority, the policy
   check, the solver, negotiation and approval run exactly as for a typed
   request.
4. The request also appears in the chat ("Move the ML lecture (C-021-L3) to
   Thu 10 am, week 12"), so everything stays in one thread.

The colours are a hint; the real check is the full solve after the drop.

## 5. The other pages

| Kept, reached from the "More" menu or the "My week" panel | Folded into the chat | Coordinators and research only (under "More") |
|---|---|---|
| This week (timetable), Handbook, Policy documents, Semester plan, Semester preferences, Club activities | Inbox, New request, Requests list | Fairness, Knowledge graph, System health, Experiments, Human study, How it works |

The request detail page stays, reached from the chat.

## Phases

| Phase | Scope | Builds on |
|---|---|---|
| 1. Chat MVP | Chat as home; questions answered in place; requests with live status; clarifying questions answered in the chat | `POST /api/requests`, `/clarify`; new `GET /api/chat` |
| 2. Cards in the chat | Negotiation and choice-of-time cards; "replying to …" chip | `/api/inbox/{id}/reply`, `/reply-text` |
| 3. Approvals section | One queue per person (publish, escalation, forwarded), shown by permission; outcomes flow back to the requester's chat | `/api/approvals`, `/decide`, `/handle` |
| 4. Classifier fixes | Questions without "?", small talk, a reply vs a new request | System One and the rule parser (`RAG_ISSUES.md` #17–#20) |
| 5. Drag and drop | "My week" panel, feasibility colours, structured requests | `free_placements()`; new `/api/feasibility`, `/api/requests/structured` |
| 6. Tidy up | Smaller sidebar; Inbox, New request and Requests folded into the chat; research pages under "More" | — |

Phases 1–3 give the single-screen experience. Phase 5 is the most
independent and can be built in parallel.
