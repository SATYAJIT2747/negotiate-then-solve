import clsx from "clsx";

export function Json({ value, className }: { value: unknown; className?: string }) {
  const text = JSON.stringify(value, null, 2) ?? "null";
  const html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?)/g, (m, str, colon, lit, n) => {
      if (str) return colon ? `<span class="text-brand">${str}</span>${colon}` : `<span class="text-ok">${str}</span>`;
      if (lit) return `<span class="text-warn">${lit}</span>`;
      if (n) return `<span class="text-info">${n}</span>`;
      return m;
    });
  return (
    <pre
      className={clsx("overflow-auto rounded-md border border-line bg-panel-2 p-4 font-mono text-[12px] leading-relaxed text-ink-2", className)}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
