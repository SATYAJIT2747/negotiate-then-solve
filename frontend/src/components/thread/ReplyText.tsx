import clsx from "clsx";

/** A reply as the system wrote it, laid out to be read at a glance. The answers list things in one sentence
 * ("You teach Communication Systems in week 5: Communication Systems lecture, Tuesday 2–3 pm (Room 104); …"),
 * so a paragraph of the form "Lead: item; item; item." is drawn as the lead and a list, and a list whose items
 * all read "what, when (where)" as rows: the time first, then the class, then the room as a tag, with what
 * every item repeats ("Communication Systems") said once, in the lead. A sentence after the list ("Withdraw
 * the one you mean…") stays a sentence. Anything else is shown as written. */
export function ReplyText({ text, className }: { text: string; className?: string }) {
  const paras = text.trim().split(/\n{2,}/);
  return (
    <div className={clsx("space-y-2", className)}>
      {paras.map((p, i) => (
        <Para key={i} text={p} />
      ))}
    </div>
  );
}

// "Lead: rest" with the lead short and free of list punctuation
const LISTED = /^([^:;\n]{3,180}?):\s+([\s\S]+)$/;
// a sentence after the list: ". Capital" not after a title ("Dr. Rao")
const AFTER = /^([\s\S]*?[)\w"”])\.\s+(?<!\b(?:Dr|Mr|Mrs|Ms|Prof|St)\.\s+)([A-Z][\s\S]*)$/;
// "what, when (where)"
const ROW = /^(.+?), (.+?) \(([^()]+)\)$/;

function Para({ text }: { text: string }) {
  // "Lead:\n- item\n- item" (a message split into requests, one line each): the lead and a list, each item
  // laid out as a paragraph of its own (its question's options listed under it)
  const lines = text.trim().split("\n");
  if (lines.length > 1 && lines.slice(1).every((l) => l.startsWith("- "))) {
    return (
      <div className="space-y-1.5">
        <p>{lines[0]}</p>
        <ul className="ml-4 list-disc space-y-1.5 marker:text-ink-3">
          {lines.slice(1).map((l, k) => (
            <li key={k}>
              <Para text={l.slice(2)} />
            </li>
          ))}
        </ul>
      </div>
    );
  }
  const m = LISTED.exec(text.trim());
  if (!m) return <p className="whitespace-pre-wrap">{text}</p>;
  const items = m[2].split(/;\s+/);
  if (items.length < 2) return <p className="whitespace-pre-wrap">{text}</p>;
  let after = "";
  const last = AFTER.exec(items[items.length - 1]);
  if (last) {
    items[items.length - 1] = last[1];
    after = last[2];
  } else {
    items[items.length - 1] = items[items.length - 1].replace(/[.?]\s*$/, ""); // the lead asks the question
  }
  const rows = items.map((it) => ROW.exec(it));
  const lead = <p>{m[1]}:</p>;
  const tail = after && <p className="whitespace-pre-wrap">{after}</p>;

  if (rows.every(Boolean)) {
    const whats = rows.map((r) => r![1]);
    const common = sharedWords(whats);
    return (
      <div className="space-y-1.5">
        {lead}
        <ul className="divide-y divide-line/60 overflow-hidden rounded-md border border-line/70 bg-panel">
          {rows.map((r, k) => (
            <li key={k} className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 px-2.5 py-1.5">
              <span className="min-w-[9.5rem] font-medium tabular-nums">{r![2]}</span>
              <span className="flex-1 text-ink-2">{cap(r![1].slice(common.length))}</span>
              <span className="rounded border border-line bg-panel-2 px-1.5 py-0.5 text-[11.5px] text-ink-2">{r![3]}</span>
            </li>
          ))}
        </ul>
        {tail}
      </div>
    );
  }
  return (
    <div className="space-y-1.5">
      {lead}
      <ul className="ml-4 list-disc space-y-0.5 marker:text-ink-3">
        {items.map((it, k) => (
          <li key={k}>{it}</li>
        ))}
      </ul>
      {tail}
    </div>
  );
}

/** The words every one of ``xs`` starts with, kept only when each still has something left after them. */
function sharedWords(xs: string[]): string {
  const words = xs.map((x) => x.split(" "));
  let n = 0;
  while (words.every((w) => w.length > n + 1 && w[n] === words[0][n])) n++;
  return n ? words[0].slice(0, n).join(" ") + " " : "";
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);
