import { colorFor, initials } from "../../lib/meta";

export function Avatar({ name, id, size = 32 }: { name: string; id?: string; size?: number }) {
  const c = colorFor(id ?? name);
  return (
    <div
      className="grid shrink-0 place-items-center rounded-full font-semibold"
      style={{ width: size, height: size, fontSize: size * 0.38, color: c, background: `color-mix(in oklab, ${c} 16%, var(--panel))` }}
    >
      {initials(name)}
    </div>
  );
}
