import clsx from "clsx";
import { BookOpen, ChevronRight, FileText, FileUp, Globe, Image as ImageIcon, Loader2, Trash2, Upload, type LucideIcon } from "lucide-react";
import { useRef, useState } from "react";
import { Badge, Button, Card, CardHeader, PageHeader, Skeleton, Toast } from "../components/ui";
import { api } from "../lib/api";
import { useApi } from "../lib/hooks";
import { how } from "../lib/meta";
import type { PolicyDocument, Rule } from "../lib/types";

const KIND_ICON: Record<PolicyDocument["kind"], LucideIcon> = { pdf: FileText, html: Globe, image: ImageIcon, markdown: BookOpen };

function readFile(f: File): Promise<string> {
  return new Promise((res, rej) => {
    const reader = new FileReader();
    reader.onload = () => res(String(reader.result).split(",")[1] ?? "");
    reader.onerror = () => rej(reader.error);
    reader.readAsDataURL(f);
  });
}

/** ``embedded``: inside the Handbook page (its header and tabs above), so the add button sits in a row here. */
export default function PolicyDocuments({ embedded = false }: { embedded?: boolean }) {
  const { data: docs, refresh: refreshDocs } = useApi(() => api.documents(), []);
  const { data: rules, refresh: refreshRules } = useApi(() => api.handbook(), []);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);
  const file = useRef<HTMLInputElement>(null);

  const refresh = () => Promise.all([refreshDocs(), refreshRules()]);

  const upload = async (f: File) => {
    setBusy(f.name);
    setError(null);
    try {
      const r = await api.uploadDocument(f.name, await readFile(f));
      setToast(`Read ${r.name}: ${r.rules.length} rule${r.rules.length === 1 ? "" : "s"} added`);
      setOpen(r.name);
      await refresh();
    } catch (e) {
      setError(`${f.name}: ${(e as Error).message}`);
    } finally {
      setBusy(null);
      if (file.current) file.current.value = "";
    }
  };

  const bySource = new Map<string, Rule[]>();
  for (const r of rules ?? []) bySource.set(r.source, [...(bySource.get(r.source) ?? []), r]);

  return (
    <>
      {embedded ? (
        <div className="mb-4 flex justify-end">
          <Button variant="primary" icon={Upload} loading={!!busy} onClick={() => file.current?.click()}>
            Add document
          </Button>
        </div>
      ) : (
        <PageHeader
          title="Policy documents"
          subtitle="Add the institute's regulations, circulars and notices. The policy agent retrieves and cites their rules."
          actions={
            <Button variant="primary" icon={Upload} loading={!!busy} onClick={() => file.current?.click()}>
              Add document
            </Button>
          }
        />
      )}
      <input ref={file} type="file" className="hidden" accept={docs?.accepts.join(",")} onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />

      <button
        type="button"
        onClick={() => file.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          const f = e.dataTransfer.files?.[0];
          if (f) void upload(f);
        }}
        disabled={!!busy}
        className={clsx(
          "mb-6 flex w-full flex-col items-center rounded-lg border border-dashed px-6 py-8 text-center transition-colors",
          drag ? "border-brand bg-brand/[0.05]" : "border-line bg-panel hover:border-ink-3/60",
        )}
      >
        {busy ? (
          <>
            <Loader2 size={22} className="animate-spin text-brand" />
            <p className="mt-3 text-[14px] font-medium">Reading {busy}…</p>
            <p className="mt-1 text-[12.5px] text-ink-3">Scans and photos go through OCR and take a few seconds a page.</p>
          </>
        ) : (
          <>
            <FileUp size={22} className="text-ink-3" />
            <p className="mt-3 text-[14px] font-medium">Drop a file here, or click to choose one</p>
            <p className="mt-1 text-[12.5px] text-ink-3">PDF (with text or scanned) · photo or scan (PNG, JPG, TIFF) · web page (HTML) · Markdown or text</p>
          </>
        )}
      </button>
      {error && <p className="-mt-3 mb-5 text-[13px] text-bad">{error}</p>}

      <Card>
        <CardHeader title="Documents" subtitle={docs ? `Scans and photos are read by ${docs.ocr.model}. Open a document to check the rules taken from it.` : undefined} />
        {!docs ? (
          <Skeleton className="m-5 h-40" />
        ) : (
          <div className="divide-y divide-line">
            {docs.documents.map((d) => {
              const Icon = KIND_ICON[d.kind] ?? FileText;
              const isOpen = open === d.name;
              const list = bySource.get(d.name) ?? [];
              return (
                <div key={d.name}>
                  <div className="flex items-center gap-3 px-5 py-3">
                    <button onClick={() => setOpen(isOpen ? null : d.name)} className="flex min-w-0 flex-1 items-center gap-3 text-left">
                      <ChevronRight size={15} className={clsx("shrink-0 text-ink-3 transition-transform", isOpen && "rotate-90")} />
                      <Icon size={16} className="shrink-0 text-ink-3" />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-[14px]">{d.name}</span>
                        <span className="mt-0.5 flex flex-wrap items-center gap-1.5 text-[12px] text-ink-3">
                          {d.error ? (
                            <span className="text-bad">{d.error}</span>
                          ) : (
                            <>
                              {Object.keys(d.methods).map((m) => (
                                <Badge key={m} tone={m.startsWith("ocr") ? "info" : "muted"}>
                                  {how(m)}
                                </Badge>
                              ))}
                              {d.ocr_confidence !== null && <span>confidence {(d.ocr_confidence * 100).toFixed(0)}%</span>}
                              <span>
                                · {d.pages} page{d.pages === 1 ? "" : "s"} · {d.rules.length} rule{d.rules.length === 1 ? "" : "s"}
                              </span>
                            </>
                          )}
                        </span>
                      </span>
                    </button>
                    {d.name !== "handbook.md" && (
                      <button
                        title={`Remove ${d.name}`}
                        onClick={async () => {
                          await api.removeDocument(d.name);
                          setToast(`Removed ${d.name} and its rules`);
                          await refresh();
                        }}
                        className="grid size-8 shrink-0 place-items-center rounded-md text-ink-3 hover:bg-panel-2 hover:text-bad"
                      >
                        <Trash2 size={15} />
                      </button>
                    )}
                  </div>
                  {isOpen && (
                    <div className="border-t border-line bg-panel-2/40 px-5 py-3 pl-[68px]">
                      {list.map((r) => (
                        <div key={r.id} className="py-2">
                          <p className="flex flex-wrap items-baseline gap-x-2 text-[13.5px]">
                            <span className="font-medium">
                              {r.number && `${r.number} `}
                              {r.title}
                            </span>
                            <span className="font-mono text-[11.5px] text-ink-3">{r.id}</span>
                            <span className="text-[12px] text-ink-3">p.{r.pages.join(", ")}</span>
                          </p>
                          <p className="mt-0.5 text-[13px] leading-relaxed text-ink-2">{r.text}</p>
                        </div>
                      ))}
                      {!list.length && <p className="py-2 text-[13px] text-ink-3">No rules were found in this document.</p>}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </Card>
      <p className="mt-4 text-[12.5px] text-ink-3">
        For harder scans, run the server with <code className="font-mono">NTS_OCR=unlimited</code> (Unlimited-OCR on a GPU), or add the{" "}
        <code className="font-mono">.ocr.json</code> files from the Kaggle notebook next to the documents.
      </p>
      <Toast message={toast} onDone={() => setToast(null)} />
    </>
  );
}
