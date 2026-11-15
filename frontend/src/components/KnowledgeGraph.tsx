import { forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY, type SimulationLinkDatum, type SimulationNodeDatum } from "d3-force";
import { Maximize2, Minus, Plus } from "lucide-react";
import clsx from "clsx";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { GraphNode } from "../lib/types";

/* The knowledge graph drawn as a force layout. Shapes say what a node is (person, constraint, class, room,
 * section, request), colour says how firm a constraint is (its tier). Input is described above
 * ``KnowledgeGraph``. Positions are kept by id, so switching views does not throw the layout around. */

export type Rel =
  | "owns" | "derived_from" | "references" | "reports_to" | "teaches" | "oversees" | "manages"
  | "attended_by" | "member_of" | "conflicts_with" | "affects";

export const EDGE_STYLE: Record<string, { color: string; dash?: string; label: string; width?: number }> = {
  owns: { color: "var(--brand)", label: "owns" },
  derived_from: { color: "var(--ink-3)", dash: "3 3", label: "comes from" },
  references: { color: "var(--warn)", label: "touches" },
  reports_to: { color: "var(--violet)", dash: "5 3", label: "reports to", width: 1.6 },
  teaches: { color: "var(--teal)", label: "teaches" },
  oversees: { color: "var(--ink-3)", dash: "1 3", label: "oversees (head of department)" },
  manages: { color: "var(--ok)", dash: "2 2", label: "manages" },
  attended_by: { color: "var(--info)", label: "attended by" },
  member_of: { color: "var(--info)", dash: "2 2", label: "represents" },
  conflicts_with: { color: "var(--bad)", dash: "6 3", label: "clashes with", width: 2.4 },
  affects: { color: "var(--teal)", label: "affects" },
};

export const KIND: Record<string, { label: string; color: string; r: number }> = {
  stakeholder: { label: "Person", color: "var(--brand)", r: 11 },
  constraint: { label: "Constraint", color: "var(--warn)", r: 8 },
  session: { label: "Class", color: "var(--teal)", r: 5.5 },
  room: { label: "Room", color: "var(--ok)", r: 8 },
  group: { label: "Section", color: "var(--info)", r: 9 },
  request: { label: "Request", color: "var(--ink-3)", r: 6 },
  rule: { label: "Rule", color: "var(--ink-3)", r: 6 },
  change: { label: "Change", color: "var(--teal)", r: 6 },
};

/** A constraint's colour by tier: the firmer, the hotter. */
export function tierColor(tier?: number): string {
  if (tier == null) return "var(--warn)";
  return tier <= 1 ? "var(--bad)" : tier === 2 ? "var(--warn)" : tier === 3 ? "var(--violet)" : tier === 4 ? "var(--brand)" : "var(--ink-3)";
}

type N = SimulationNodeDatum & GraphNode & { deg: number };
type L = SimulationLinkDatum<N> & { rel: string };

function Shape({ n, r, fill, ring }: { n: N; r: number; fill: string; ring?: string }) {
  const stroke = ring ?? "var(--panel)";
  if (n.kind === "constraint") {
    const s = r * 1.25;
    return n.hard === false ? (
      <path d={`M0 ${-s} L${s} 0 L0 ${s} L${-s} 0 Z`} fill="var(--panel)" stroke={fill} strokeWidth={2.2} />
    ) : (
      <path d={`M0 ${-s} L${s} 0 L0 ${s} L${-s} 0 Z`} fill={fill} stroke={stroke} strokeWidth={1.5} />
    );
  }
  if (n.kind === "room") return <rect x={-r} y={-r} width={r * 2} height={r * 2} rx={2} fill={fill} stroke={stroke} strokeWidth={1.5} />;
  if (n.kind === "group") return <rect x={-r} y={-r * 0.75} width={r * 2} height={r * 1.5} rx={r * 0.6} fill={fill} stroke={stroke} strokeWidth={1.5} />;
  if (n.kind === "request" || n.kind === "rule")
    return <path d={`M0 ${-r * 1.1} L${r} ${r * 0.8} L${-r} ${r * 0.8} Z`} fill={fill} stroke={stroke} strokeWidth={1.5} />;
  return <circle r={r} fill={fill} stroke={stroke} strokeWidth={n.kind === "stakeholder" ? 2.5 : 1.5} />;
}

/* How input is read (one pointer at a time, a press that only becomes something once it means it):
 *
 *   press ──(moves < DRAG_PX)──▶ release = click      (node: select / unselect it; background: clear)
 *         └─(moves ≥ DRAG_PX)──▶ drag                 (node: move it, pinned while held; background: pan)
 *
 * Zoom: Ctrl/⌘ + wheel or a trackpad pinch, smoothed by the wheel's distance; a plain wheel scrolls the
 * page (a hint says how to zoom); double-click zooms in on the point; the buttons and + / − / 0 keys too.
 * Hover lights a node's neighbours only after it rests (HOVER_MS) and never during a drag. The view moves
 * to fit or to a picked node by easing, and an automatic fit never overrides a view the user has moved. */
const DRAG_PX = 5;
const HOVER_MS = 140;
const ZOOM_MIN = 0.2;
const ZOOM_MAX = 4;

type View = { x: number; y: number; k: number };

export function KnowledgeGraph({
  nodes: inNodes,
  edges,
  selected,
  onSelect,
  labelAll,
}: {
  nodes: GraphNode[];
  edges: { source: string; target: string; rel: string }[];
  selected: string | null;
  onSelect: (id: string | null) => void;
  labelAll?: boolean;
}) {
  const box = useRef<HTMLDivElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState({ w: 900, h: 620 });
  const [, setTick] = useState(0);
  const [hover, setHover] = useState<string | null>(null);
  const [view, setView] = useState<View>({ x: 0, y: 0, k: 1 });
  const [hint, setHint] = useState(false);
  const [mode, setMode] = useState<"idle" | "pan" | "drag">("idle");
  const sizeRef = useRef(size);
  sizeRef.current = size;
  const viewRef = useRef(view);
  viewRef.current = view;
  const press = useRef<{ id: number; sx: number; sy: number; node: N | null; vx: number; vy: number; active: boolean } | null>(null);
  const touched = useRef(false); // the user has moved the view since the last automatic fit
  const hoverTimer = useRef<number | undefined>(undefined);
  const hintTimer = useRef<number | undefined>(undefined);
  const anim = useRef<number | undefined>(undefined);
  const pos = useRef(new Map<string, { x: number; y: number }>());
  const sim = useRef<ReturnType<typeof forceSimulation<N>> | null>(null);

  // the graph's structure: which nodes and edges, not the objects. A refresh of the same data, or a new
  // selection, must not restart the layout or refit the view; only a different set of nodes or edges does.
  const sig = useMemo(
    () => `${inNodes.map((n) => n.id).sort().join(",")}|${edges.map((e) => `${e.source}>${e.target}:${e.rel}`).sort().join(",")}`,
    [inNodes, edges],
  );
  const latest = useRef({ inNodes, edges });
  latest.current = { inNodes, edges };
  // details (titles, labels) may change without the structure: read them from the latest props when drawing
  const info = useMemo(() => new Map(inNodes.map((n) => [n.id, n])), [inNodes]);

  const { nodes, links } = useMemo(() => {
    const { inNodes, edges } = latest.current;
    const deg = new Map<string, number>();
    for (const e of edges) {
      deg.set(e.source, (deg.get(e.source) ?? 0) + 1);
      deg.set(e.target, (deg.get(e.target) ?? 0) + 1);
    }
    const nodes: N[] = inNodes.map((n) => ({ ...n, deg: deg.get(n.id) ?? 0, ...(pos.current.get(n.id) ?? {}) }));
    const ids = new Set(nodes.map((n) => n.id));
    const links: L[] = edges.filter((e) => ids.has(e.source) && ids.has(e.target)).map((e) => ({ source: e.source, target: e.target, rel: e.rel }));
    return { nodes, links };
  }, [sig]);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    // a resize (a scrollbar appearing, the side panel changing) keeps the picture centred; it never restarts
    // the layout
    const ro = new ResizeObserver(([e]) =>
      setSize((old) => {
        const w = e.contentRect.width;
        const h = e.contentRect.height;
        if (old.w !== w || old.h !== h) setView((v) => ({ ...v, x: v.x + (w - old.w) / 2, y: v.y + (h - old.h) / 2 }));
        return { w, h };
      }),
    );
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const radius = useCallback((n: N) => (KIND[n.kind]?.r ?? 6) + Math.min(6, Math.sqrt(n.deg)), []);

  useEffect(() => {
    const fresh = nodes.some((n) => n.x == null);
    const s = forceSimulation<N>(nodes)
      .force(
        "link",
        forceLink<N, L>(links)
          .id((d) => d.id)
          .distance((l) => (l.rel === "conflicts_with" ? 70 : l.rel === "attended_by" || l.rel === "teaches" ? 40 : 55))
          .strength(0.6),
      )
      .force("charge", forceManyBody<N>().strength((d) => (d.kind === "stakeholder" ? -260 : -110)))
      .force("x", forceX(sizeRef.current.w / 2).strength(0.04))
      .force("y", forceY(sizeRef.current.h / 2).strength(0.05))
      .force("collide", forceCollide<N>().radius((d) => radius(d) + 6))
      .alpha(fresh ? 1 : 0.3)
      .alphaDecay(0.04)
      .velocityDecay(0.5) // heavier: nodes settle instead of drifting
      .on("tick", () => {
        for (const n of nodes) if (n.x != null && n.y != null) pos.current.set(n.id, { x: n.x, y: n.y });
        setTick((t) => t + 1);
      });
    sim.current = s;
    return () => {
      s.stop();
    };
  }, [nodes, links, radius]);

  // -- the view: eased moves, and zoom about a point --------------------------------------------------------

  const ease = useCallback((to: View, ms = 380) => {
    if (anim.current) cancelAnimationFrame(anim.current);
    const from = viewRef.current;
    const t0 = performance.now();
    const step = (t: number) => {
      const p = Math.min(1, (t - t0) / ms);
      const e = 1 - Math.pow(1 - p, 3);
      setView({ x: from.x + (to.x - from.x) * e, y: from.y + (to.y - from.y) * e, k: from.k + (to.k - from.k) * e });
      if (p < 1) anim.current = requestAnimationFrame(step);
    };
    anim.current = requestAnimationFrame(step);
  }, []);

  const zoomTo = useCallback((k: number, px: number, py: number, animate = false) => {
    const v = viewRef.current;
    const nk = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, k));
    const to = { k: nk, x: px - ((px - v.x) / v.k) * nk, y: py - ((py - v.y) / v.k) * nk };
    touched.current = true;
    if (animate) ease(to, 220);
    else setView(to);
  }, [ease]);

  const fit = useCallback(
    (animate = true) => {
      const ps = nodes.filter((n) => n.x != null && n.y != null);
      if (!ps.length) return;
      const xs = ps.map((n) => n.x!);
      const ys = ps.map((n) => n.y!);
      const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
      const k = Math.max(0.25, Math.min(1.8, 0.9 * Math.min(size.w / Math.max(1, x1 - x0 + 90), size.h / Math.max(1, y1 - y0 + 90))));
      const to = { k, x: size.w / 2 - k * ((x0 + x1) / 2), y: size.h / 2 - k * ((y0 + y1) / 2) };
      touched.current = false;
      if (animate) ease(to);
      else setView(to);
    },
    [nodes, size.w, size.h, ease],
  );

  // fit when the node set changes, once the layout has mostly settled, unless the user has moved the view
  useEffect(() => {
    touched.current = false;
    const id = window.setTimeout(() => !touched.current && fit(), 1200);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig]);

  // a node picked from outside the canvas (search, lists): ease it into view if it is off screen
  useEffect(() => {
    if (!selected) return;
    const n = nodes.find((x) => x.id === selected);
    if (n?.x == null || n.y == null) return;
    const v = viewRef.current;
    const sx = v.x + v.k * n.x;
    const sy = v.y + v.k * n.y;
    if (sx > 60 && sx < size.w - 60 && sy > 60 && sy < size.h - 60) return;
    ease({ ...v, x: size.w / 2 - v.k * n.x, y: size.h / 2 - v.k * n.y });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

  // wheel: zoom only with Ctrl/⌘ (a trackpad pinch sends ctrlKey); a plain wheel scrolls the page. Native
  // and non-passive, so zooming does not also scroll the page.
  useEffect(() => {
    const el = svg.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey)) {
        setHint(true);
        clearTimeout(hintTimer.current);
        hintTimer.current = window.setTimeout(() => setHint(false), 1400);
        return;
      }
      e.preventDefault();
      const r = el.getBoundingClientRect();
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY; // lines -> pixels
      const factor = Math.exp(-Math.max(-60, Math.min(60, dy)) * 0.004); // a notch ~ 25%, a trackpad tick much less
      zoomTo(viewRef.current.k * factor, e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoomTo]);

  // keys, while the pointer is over the canvas or it has focus
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") onSelect(null);
    else if (e.key === "+" || e.key === "=") zoomTo(viewRef.current.k * 1.25, size.w / 2, size.h / 2, true);
    else if (e.key === "-") zoomTo(viewRef.current.k / 1.25, size.w / 2, size.h / 2, true);
    else if (e.key === "0") fit();
    else return;
    e.preventDefault();
  };

  // -- pointer: press, then click or drag ------------------------------------------------------------------

  const local = (cx: number, cy: number) => {
    const r = svg.current!.getBoundingClientRect();
    return { x: cx - r.left, y: cy - r.top };
  };

  const down = (e: React.PointerEvent, node: N | null) => {
    if (e.button !== 0 || press.current) return;
    e.stopPropagation();
    svg.current?.setPointerCapture(e.pointerId);
    if (anim.current) cancelAnimationFrame(anim.current);
    press.current = { id: e.pointerId, sx: e.clientX, sy: e.clientY, node, vx: viewRef.current.x, vy: viewRef.current.y, active: false };
  };

  const move = (e: React.PointerEvent) => {
    const p = press.current;
    if (!p || p.id !== e.pointerId) return;
    const dx = e.clientX - p.sx;
    const dy = e.clientY - p.sy;
    if (!p.active) {
      if (Math.hypot(dx, dy) < DRAG_PX) return; // still a click
      p.active = true;
      setHover(null);
      clearTimeout(hoverTimer.current);
      setMode(p.node ? "drag" : "pan");
      if (p.node) sim.current?.alphaTarget(0.08).restart(); // gentle: only the neighbourhood follows
    }
    if (p.node) {
      const v = viewRef.current;
      const l = local(e.clientX, e.clientY);
      p.node.fx = (l.x - v.x) / v.k;
      p.node.fy = (l.y - v.y) / v.k;
    } else {
      touched.current = true;
      setView((v) => ({ ...v, x: p.vx + dx, y: p.vy + dy }));
    }
  };

  const up = (e: React.PointerEvent) => {
    const p = press.current;
    if (!p || p.id !== e.pointerId) return;
    press.current = null;
    setMode("idle");
    if (p.active && p.node) {
      p.node.fx = null;
      p.node.fy = null;
      sim.current?.alphaTarget(0);
    } else if (!p.active) {
      if (p.node) onSelect(p.node.id === selected ? null : p.node.id);
      else onSelect(null);
    }
  };

  const cancel = () => {
    const p = press.current;
    if (p?.node) {
      p.node.fx = null;
      p.node.fy = null;
      sim.current?.alphaTarget(0);
    }
    press.current = null;
    setMode("idle");
  };

  const enter = (id: string) => {
    if (press.current?.active) return;
    clearTimeout(hoverTimer.current);
    hoverTimer.current = window.setTimeout(() => setHover(id), HOVER_MS);
  };
  const leave = () => {
    clearTimeout(hoverTimer.current);
    hoverTimer.current = window.setTimeout(() => setHover(null), 60);
  };

  useEffect(() => () => {
    clearTimeout(hoverTimer.current);
    clearTimeout(hintTimer.current);
    if (anim.current) cancelAnimationFrame(anim.current);
  }, []);

  const focus = mode === "idle" ? hover ?? selected : selected;
  const lit = useMemo(() => {
    if (!focus) return null;
    const set = new Set([focus]);
    for (const l of links) {
      const a = (l.source as N).id ?? (l.source as unknown as string);
      const b = (l.target as N).id ?? (l.target as unknown as string);
      if (a === focus) set.add(b);
      if (b === focus) set.add(a);
    }
    return set;
  }, [focus, links]);

  const hoverNode = mode === "idle" && hover ? nodes.find((n) => n.id === hover) : null;
  const hoverInfo = hoverNode ? info.get(hoverNode.id) ?? hoverNode : null;

  return (
    <div ref={box} className="relative h-full w-full">
      <svg
        ref={svg}
        width={size.w}
        height={size.h}
        tabIndex={0}
        onKeyDown={onKey}
        onPointerEnter={() => svg.current?.focus({ preventScroll: true })}
        className={clsx("touch-none outline-none select-none", mode === "pan" ? "cursor-grabbing" : mode === "drag" ? "cursor-grabbing" : "cursor-grab")}
        onPointerDown={(e) => down(e, null)}
        onPointerMove={move}
        onPointerUp={up}
        onPointerCancel={cancel}
        onLostPointerCapture={cancel}
        onDoubleClick={(e) => {
          const l = local(e.clientX, e.clientY);
          zoomTo(viewRef.current.k * 1.6, l.x, l.y, true);
        }}
      >
        <defs>
          <marker id="kg-arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M0 0 L10 5 L0 10 z" fill="var(--violet)" />
          </marker>
        </defs>
        <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
          {links.map((l, i) => {
            const a = l.source as N;
            const b = l.target as N;
            if (a.x == null || b.x == null) return null;
            const st = EDGE_STYLE[l.rel] ?? { color: "var(--ink-3)" };
            const on = lit ? a.id === focus || b.id === focus : true;
            const conflict = l.rel === "conflicts_with";
            return (
              <line
                key={i}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke={st.color}
                strokeWidth={(st.width ?? 1.2) / Math.sqrt(view.k)}
                strokeDasharray={st.dash}
                markerEnd={l.rel === "reports_to" ? "url(#kg-arrow)" : undefined}
                opacity={lit ? (on ? 0.95 : 0.06) : conflict ? 0.9 : 0.4}
                className={clsx("pointer-events-none transition-opacity duration-200", conflict && "kg-flow")}
              />
            );
          })}
          {nodes.map((n) => {
            if (n.x == null || n.y == null) return null;
            const r = radius(n);
            const fill = n.kind === "constraint" ? tierColor(n.tier) : KIND[n.kind]?.color ?? "var(--ink-3)";
            const isSel = n.id === selected;
            const dim = lit && !lit.has(n.id);
            const label = info.get(n.id)?.label ?? n.label;
            const showLabel =
              labelAll || isSel || n.id === focus || (lit && lit.has(n.id)) || n.kind === "stakeholder" || n.kind === "group" || n.kind === "room" || view.k > 1.5;
            return (
              <g
                key={n.id}
                transform={`translate(${n.x} ${n.y})`}
                opacity={dim ? 0.14 : 1}
                className={clsx("transition-opacity duration-200", mode === "idle" && "cursor-pointer")}
                onPointerEnter={() => enter(n.id)}
                onPointerLeave={leave}
                onPointerDown={(e) => down(e, n)}
              >
                {/* a generous, invisible hit area: small nodes are easy to catch */}
                <circle r={Math.max(r + 8, 14) / Math.sqrt(Math.max(1, view.k))} fill="transparent" />
                {isSel && <circle r={r + 7} fill="none" stroke="var(--brand)" strokeWidth={2} className="kg-pulse pointer-events-none" />}
                <g className="pointer-events-none">
                  <Shape n={n} r={r} fill={fill} />
                </g>
                {showLabel && (
                  <text
                    y={r + 12}
                    textAnchor="middle"
                    className="pointer-events-none fill-ink text-[10.5px]"
                    style={{ paintOrder: "stroke", stroke: "var(--bg)", strokeWidth: 3.5, fontWeight: n.kind === "stakeholder" || isSel ? 600 : 400 }}
                  >
                    {label.length > 24 ? `${label.slice(0, 22)}…` : label}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>
      {hoverNode && (
        <div className="pointer-events-none absolute top-3 left-3 max-w-sm rounded-md border border-line bg-panel/95 px-3 py-2 text-[12px] shadow-lg backdrop-blur">
          <p className="text-[10.5px] tracking-wide text-ink-3 uppercase">{KIND[hoverNode.kind]?.label ?? hoverNode.kind}</p>
          <p className="font-medium">{hoverInfo?.title}</p>
          <p className="mt-0.5 text-[11px] text-ink-3">
            {hoverNode.deg} connection{hoverNode.deg === 1 ? "" : "s"} · click for details · drag to move
          </p>
        </div>
      )}
      <div
        className={clsx(
          "pointer-events-none absolute inset-x-0 top-1/2 mx-auto w-fit -translate-y-1/2 rounded-md bg-ink/80 px-3 py-1.5 text-[12.5px] text-bg transition-opacity duration-300",
          hint ? "opacity-100" : "opacity-0",
        )}
      >
        Hold Ctrl (⌘ on a Mac) and scroll to zoom, or pinch
      </div>
      <p className="pointer-events-none absolute bottom-3 left-3 text-[11px] text-ink-3">
        click to select · drag to move · Ctrl + scroll to zoom · double-click to zoom in · Esc to clear
      </p>
      <div className="absolute right-3 bottom-3 flex flex-col overflow-hidden rounded-md border border-line bg-panel shadow-sm">
        <button onClick={() => zoomTo(viewRef.current.k * 1.3, size.w / 2, size.h / 2, true)} className="grid size-8 place-items-center hover:bg-panel-2" title="Zoom in (+)">
          <Plus size={14} />
        </button>
        <button onClick={() => zoomTo(viewRef.current.k / 1.3, size.w / 2, size.h / 2, true)} className="grid size-8 place-items-center border-t border-line hover:bg-panel-2" title="Zoom out (−)">
          <Minus size={14} />
        </button>
        <button onClick={() => fit()} className="grid size-8 place-items-center border-t border-line hover:bg-panel-2" title="Fit to view (0)">
          <Maximize2 size={13} />
        </button>
      </div>
    </div>
  );
}
