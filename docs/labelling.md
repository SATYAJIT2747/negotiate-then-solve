# Labelling timetable requests

How to label a message for `data/human/*.jsonl` (format: `src/training/human.py`). The generators and
the checks in the code follow the same rules, so a labelled message teaches the model what the rest of
the data does. When this guide and a generator disagree, the guide is right and the generator is a bug.

The quickest way: run `python -m training.human prefill` on a batch of messages, then correct each
`gold` and set `"reviewed": true`. Don't accept a prefilled reading without reading it.

## 1. What the message asks: `action`

| action | when | example |
|---|---|---|
| `compile` | a timetable request clear enough to write down | "no classes before 10 on Mondays" |
| `clarify` | a timetable request missing what you would have to guess | "I'll be away a few days soon" |
| `investigate` | students report two of their classes at the same time | "DS and ML clash on Tue at 2" |
| `answer` | a question about the rules | "how many hours in a row are allowed?" |
| `withdraw` | takes back an earlier request | "ignore my last message" |
| `out_of_scope` | not about the teaching timetable | "the wifi in H101 is down" |

**Never guess.** If you can't tell which days, weeks, times or class without inventing something, it's
`clarify` with `missing` (days, weeks, slots, session) and a short `question`.

## 2. What kind of request: the intent

| kind | means | not to be confused with |
|---|---|---|
| `unavailable` | they **cannot** teach then (leave, duty, appointment) | `avoid`: they'd rather not |
| `avoid` | times they don't want | |
| `only` | the only times they want to teach | `avoid` the rest |
| `move` | one class to other times | `only`: all their classes |
| `swap` | their class with a colleague's | |
| `extra_class` / `cancel` | a class on top of the weekly ones / one not held | |
| `load_limit` | at most N hours in a row | |
| `needs_room` / `room_closed` | equipment a class needs / a room out of use | |
| `club_booking` | a room for a club activity | |
| `clash_report` | with `investigate` | |

A message may ask several things: one intent each, in the order they're asked.

## 3. How firm: `hard`

Based on the tiers in the project brief: a preferred time is a preference (Tier 5).

- **`hard: true` only when the message says it must be so**: *must, cannot, can't, have to, need,
  no other time works, under no circumstances, not negotiable*. For a **move**, also when the message names
  a conflict that rules out the time the class is at now ("it clashes with a committee meeting").
- **`hard: false`** (leave it out) for wishes (*if possible, ideally, I'd prefer, would be nice, no worries
  if not*) **and for plain requests** ("please move my lecture to Friday", "could my class go to Tuesday?").
  Politeness is not firmness.
- **"It's hard for me", "difficult for me", "I find it tough"** are about difficulty: a wish, not a must.
- `unavailable` has no `hard`: it always is.
- If one part of a message is a must and another a wish, label each intent by its own words.

## 4. Which class, and where it should go

- `which` describes the class **as it is now**: `course`, `kind`, `day` and `time` *where it is now*,
  only what the message says. "My Thursday 10 am lecture" → `day: Thu, time: 10:00, kind: lecture`.
- `to` is **where it should go**. "I need my ML lecture on Friday at 2 pm" → `which: {course: ML}`,
  `to: {days: [Fri], at: [14:00]}`. Don't put the destination in `which`.
- `owner` only for someone else's class; `who` only when it isn't the sender.

## 5. Times and weeks

- Times are 24-hour `HH:MM`, on the hour. Use **one** of: `at` (start times), `before`/`after`
  (between = both), or `part` (`morning`, `afternoon`, `lunch`). "After lunch" is `part: afternoon`.
- No times at all means the whole day: leave them out (never `"at": ["all"]`).
- Weeks: `weeks` (named weeks), `from_week`/`to_week`, or counted from now: `this_week` / `next_week`
  with `count` ("the next three weeks" → `next_week, count 3`; "this week and next" → `this_week, count 2`),
  or `rest_of_semester`. Leave the period out when it's every week.

## 6. Club bookings

`club`, `activity`, `day` or `date` (ISO, only when the message gives a date), `start`, `end`,
`attendees`, `cohorts`. A named room ("H101", "the auditorium") is `room`; a kind of room ("a seminar
room", "a lecture hall") is `room_kind`.

## 7. Before you mark a row reviewed

- Defaults left out: no `hard: false`, no `justification: none`, no empty lists.
- `justification: stated` when a reason is given; never copy the reason itself (it may be private).
- `split`: keep `human` (evaluation only) unless the team has decided to train on this batch.
- `training.dataset build` runs every labelled row through the same gates as generated data; a row that
  fails is listed with the reason. Fix the label, don't remove the gate.
