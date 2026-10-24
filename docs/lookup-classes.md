# Classes behind the timetable questions (`f-lookup`)

Three diagrams: **the data** the questions are answered from, **the lookup classes** that read it, and **how one
question flows** through them. At the end, the data no tool reads yet: candidates for new tools.

The diagrams are [Mermaid](https://mermaid.js.org/syntax/classDiagram.html): GitHub shows them as pictures; in VS
Code, open the Markdown preview (with a Mermaid extension). Types are simplified: `str?` means "a string or
nothing", `list~Room~` a list of rooms.

## 1. The data

Everything the questions are about. The **department** (`Instance`) comes from the department file and does not
change during the semester; the **timetable and the requests** live in the `Store` (SQLite) and change as
requests are published.

```mermaid
classDiagram
    direction LR

    class Instance {
        +str name
        +Calendar calendar
        +list~Room~ rooms
        +list~Faculty~ faculty
        +list~Group~ groups
        +list~Session~ sessions
        +Policy policy
        +dict course_titles
        +course_title(course) str
        +session_size(session) int
    }
    class Calendar {
        +list~str~ days
        +int slots_per_day
        +int? lunch_slot
        +int weeks
    }
    class Policy {
        +int? max_consecutive
        +bool lunch_break
    }
    class Faculty {
        +str id
        +str name
        +str department
        +Role role
        +str? reports_to
    }
    class Group {
        <<a section>>
        +str id
        +str name
        +int size
    }
    class Room {
        +str id
        +str name
        +int capacity
        +RoomType type
        +list~str~ equipment
    }
    class Session {
        <<one weekly class>>
        +str id
        +str course
        +SessionKind kind
        +str faculty
        +list~str~ groups
        +int duration
        +RoomType room_type
        +list~str~ equipment
        +int? size
    }

    Instance *-- Calendar
    Instance *-- Policy
    Instance *-- "many" Faculty
    Instance *-- "many" Group
    Instance *-- "many" Room
    Instance *-- "many" Session
    Session --> Faculty : taught by
    Session --> "1..*" Group : attended by
    Faculty --> Faculty : reports_to

    class Store {
        <<SQLite>>
        +current_version(week?) TimetableVersion
        +versions() list
        +extras(week?) list~ExtraClass~
        +constraints() list~Constraint~
        +requests(status?) list~Request~
        +ledger() list~ConcessionEntry~
        +events(case_id?) list
    }
    class TimetableVersion {
        +int version
        +dict assignment
        +int? parent
        +str? approved_by
        +int? week
        +list~str~ cancelled
        +diff(other) dict
    }
    class Placement {
        <<where a class is>>
        +str day
        +int slot
        +str room
    }
    class ExtraClass {
        +Session session
        +int week
        +Placement placement
        +str? request
    }
    class Request {
        +str id
        +Channel channel
        +str sender_id
        +Role role
        +str raw_text
        +datetime received_at
        +RequestStatus status
    }
    class Constraint {
        <<a rule from a request or the policy>>
        +str id
        +ConstraintType type
        +bool hard
        +Tier tier
        +str? owner
        +Scope scope
        +When when
        +Justification justification
    }
    class ConcessionEntry {
        <<the fairness ledger>>
        +str stakeholder
        +str constraint_id
        +str? conceded_to
        +int? week
        +float credit
    }

    Store o-- "many" TimetableVersion
    Store o-- "many" ExtraClass
    Store o-- "many" Request
    Store o-- "many" Constraint
    Store o-- "many" ConcessionEntry
    TimetableVersion *-- "many" Placement : session id to placement
    TimetableVersion --> TimetableVersion : parent
    ExtraClass *-- Session
    ExtraClass *-- Placement
    Placement --> Room
    Constraint --> Request : made from
```

**Reading it:** a `Session` is one weekly class (e.g. the Data Structures lecture for Section 1); where it is held
is a `Placement` inside a `TimetableVersion`. Version `week = None` is the semester timetable; a version with a
week number is that week's own timetable (after an absence, a closed room), and its `cancelled` lists the
classes not held that week. Extra classes are separate (`ExtraClass`), for one week.

## 2. The lookup classes

What `f-lookup` adds. Left: **understanding** a question (regex or a model) into a `ToolCall`. Right: **facts**
(`WeekTimetable`, `Department`) and **the reply** (`Answerer`). The two sides only meet in the `ToolCall`, so a
misunderstanding can never change a fact.

```mermaid
classDiagram
    direction LR

    class ToolCall {
        <<what to look up>>
        +str tool
        +dict args
        +int week
        +bool confident
        +str note
    }

    class QuestionReader {
        <<regex>>
        +Instance instance
        +str sender
        +int current_week
        +date today
        +datetime? now
        +date? semester_start
        +read(text) ToolCall?
        -_when(text)
        -_teachers(text)
        -_names(text)
    }
    class ToolAgent {
        <<a model picks the lookup>>
        +Instance instance
        +Client client
        +str sender
        +int current_week
        +date today
        +datetime? now
        +date? semester_start
        +list trace
        +read(text) ToolCall?
        -_prompt(text) str
        -_check(choice) ToolCall?
    }
    class Choice {
        <<the model's answer, names not IDs>>
        +str tool
        +list teachers / sections / rooms
        +str? course
        +list days / dates / hours
        +str? part
        +int? length
        +str? kind
        +list equipment
        +int? min_capacity
        +bool next / count
        +str? of / what / reason
        +int week
        +list unknown / ambiguous
    }
    class Client {
        <<interface>>
        +generate(system, prompt, schema) Choice
    }
    class GeminiClient
    class LocalClient {
        <<llama-server + nts-lookup-v1>>
    }
    class Directory {
        +find(text) list
    }

    class WeekTimetable {
        <<facts: one week>>
        +Instance instance
        +int week
        +list~ScheduledClass~ classes
        +list~ExtraClass~ extras
        +classes_of(teacher, section, room, course, day)
        +next_class(after, teacher, section)
        +free_hours(teachers, sections, rooms, days, length)
        +free_with_room(teachers, sections, kind, equipment, min_capacity, days, length)
        +free_rooms(kind, equipment, min_capacity, days)
        +who_is_free(days, slots, course, sections)
        +changes(teacher, section) list~Change~
    }
    class ScheduledClass {
        +Session session
        +Placement placement
        +int week
        +hours() list
    }
    class Change {
        +str kind
        +Session session
        +Placement? before
        +Placement? after
    }
    class Department {
        <<facts: the whole semester>>
        +rooms(kind)
        +sections()
        +teachers()
        +courses_of(teacher)
        +teachers_of(course)
        +sections_of(course)
        +sessions_of(course)
        +weekly_hours(teacher)
    }
    class Answerer {
        <<the reply, fixed sentences>>
        +Instance instance
        +str sender
        +answer(call, week) str
    }

    QuestionReader ..> ToolCall : returns
    QuestionReader ..> Directory : names
    ToolAgent ..> Client : asks
    Client ..> Choice : returns
    Client <|.. GeminiClient
    Client <|.. LocalClient
    ToolAgent ..> Choice : checks
    ToolAgent ..> Directory : names to IDs
    ToolAgent ..> ToolCall : returns
    Answerer ..> ToolCall : runs
    Answerer ..> WeekTimetable : week facts
    Answerer ..> Department : department facts
    WeekTimetable *-- "many" ScheduledClass
    WeekTimetable ..> Change : changes()
```

**The six tools** are methods of `WeekTimetable` and `Department`; `ToolCall.tool` names one:

| `ToolCall.tool` | Method(s) |
|---|---|
| `free_hours` | `WeekTimetable.free_hours`, or `free_with_room` when a room is needed too |
| `classes_of` | `WeekTimetable.classes_of`, or `next_class` for "my next class" |
| `free_rooms` | `WeekTimetable.free_rooms` |
| `who_is_free` | `WeekTimetable.who_is_free` |
| `describe` | `Department` |
| `changes` | `WeekTimetable.changes` |
| `unclear` | none: the `note` is the reply ("I don't know a teacher called Smith.") |

## 3. One question, end to end

```mermaid
sequenceDiagram
    actor U as Teacher
    participant API as POST /api/ask
    participant QR as QuestionReader (regex)
    participant TA as ToolAgent (model)
    participant WT as WeekTimetable
    participant AN as Answerer

    U->>API: "when are Dr. Iyer and I both free on Friday?"
    API->>QR: read(text)
    alt the regex understands it
        QR-->>API: ToolCall(free_hours, teachers=[me, F-103], days=[Fri], week 13)
    else the regex can't read it (not wired in yet)
        API->>TA: read(text)
        TA->>TA: model returns a Choice, checked, names to IDs
        TA-->>API: ToolCall(...)
    else a request or a rule question
        QR-->>API: None, so the message goes to the request flow
    end
    API->>WT: WeekTimetable(instance, store, 13)
    API->>AN: answer(call, week)
    AN->>WT: free_hours(teachers=[F-102, F-103], days=[Fri])
    WT-->>AN: [(Fri, 0), (Fri, 1), (Fri, 2), (Fri, 3)]
    AN-->>API: "You and Dr. Iyer are both free on Friday 9 am–1 pm (week 13)."
    API-->>U: the answer (and the free hours, for the grid)
```

## 4. Data no tool reads yet: candidates for new tools

| Data | Class / method | Questions it could answer | Notes |
|---|---|---|---|
| who leads whom | `Faculty.role`, `Faculty.reports_to` | "who is the head of department?", "who do I report to?" | today read as "not a timetable question" (the test set even expects that) |
| preferences and unavailability | `Store.constraints()` | "what preferences have I given?", "when is Dr. Iyer unavailable?" | times only, never reasons (P-PRIVACY) |
| requests and their status | `Store.requests()`, `Store.events()` | "what happened to my swap request?" | also on the Requests tab |
| the fairness ledger | `Store.ledger()` | "how many times have I given way this semester?" | |
| timetable history | `Store.versions()`, `TimetableVersion.parent / approved_by` | "when did the timetable last change, and who approved it?" | |
| policy limits | `Instance.policy` | "how many hours in a row can I teach?" | answered by rule search today; the number is also data |
| what a class needs | `Session.equipment`, `room_type`, `Instance.session_size()` | "what does my DS lab need?", "how many students are in my ML class?" | `describe` covers part of it |
| club bookings | `semester.clubs` (not in the diagrams) | "is Room 103 booked on Friday evening?" | evenings, outside the teaching timetable |

The first three look the most useful: a `describe` for people (`role`, `reports_to`), a `my_requests` tool, and a
`constraints` tool that never shows a reason.
