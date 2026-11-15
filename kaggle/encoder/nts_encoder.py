"""NTS encoder intent reader: train and evaluate (docs/encoder-intents.md). Runs as a Kaggle script kernel.

Labels are input tokens. One encoder (DeBERTa-v3-base by default), two passes over a message:

* pass 1 (what the message is): ``[P] action [L] compile [L] clarify ...`` (softmax over the ``[L]``),
  ``[P] choice [L] any_of [L] ranked [L] fallback`` (sigmoid each), ``[P] intent [E] move: ... [E] avoid: ...``:
  per kind, start and end pointers over the message's tokens, decoded like brackets (a choice's second item
  sits inside the first's words);
* pass 2 (one request): the request's words marked by a learned embedding added to their tokens;
  ``[P] <kind> [E] class day [E] target day ...`` scored against every span of up to ``MAX_W`` tokens (several
  per role: "Monday and Tuesday"); ``[P] <attribute> [L] value ...`` (softmax, "none" when absent);
  ``[P] <ref> same [L] no [L] yes`` (a class named as an earlier item's).

Training uses gold extents in pass 2; evaluation runs pass 2 on pass 1's predicted extents. A row is
correct when the action, the choices and every request (kind, role spans by text, attributes, refs) equal
gold. ``conf`` is the least sure decision (including the most likely thing left out), so a gate "conf >= tau"
gives the coverage / accuracy curve in metrics.json. Writes metrics.json, predictions/<split>.jsonl,
schema.json, train_log.jsonl and the weights (model.pt, kept on Kaggle).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ACTIONS = ["compile", "clarify", "investigate", "answer", "out_of_scope", "withdraw"]
COMPOSITES = ["any_of", "ranked", "fallback"]
GLOSS = {
    "move": "one of the sender's classes should go to another time",
    "unavailable": "the sender cannot teach at some time",
    "avoid": "the sender would rather not teach at some time",
    "only": "the sender wants to teach only at some times",
    "cancel": "a class should not happen",
    "extra_class": "an additional class should be scheduled",
    "swap": "exchange a class slot with a colleague",
    "relate": "a class placed relative to another class",
    "limit": "a cap on classes, hours or days, or a gap",
    "load_limit": "no more than some hours in a row",
    "needs_room": "a class needs a room with equipment",
    "room_closed": "a room cannot be used",
    "club_booking": "a student club wants a room",
    "clash_report": "two classes clash for students",
}
PREFIX = {"which": "class", "other": "other class", "mine": "my class", "theirs": "their class", "to": "target",
          "time": "time", "period": "period"}
SUFFIX = {"course": "course", "day": "day", "days": "day", "time": "time", "at": "time", "after": "after",
          "before": "before", "kind": "kind", "owner": "teacher", "group": "group", "weeks": "week",
          "from_week": "from week", "to_week": "to week", "except_weeks": "except week", "part": "part of day"}
TOP = {"courses": "course", "club": "club", "activity": "activity", "room": "room", "room_kind": "room type",
       "cohorts": "student group", "equipment": "equipment", "day": "day", "start": "start time",
       "end": "end time", "attendees": "attendees", "n": "number", "max_consecutive": "hours in a row",
       "date": "date", "who": "person", "days": "days apart"}
SPECIAL = ["[CTX]", "[MSG]", "[/MSG]", "[P]", "[L]", "[E]"]
MAX_W = 12  # longest role span, in tokens


def role_name(role: str) -> str:
    if "." in role:
        p, s = role.split(".", 1)
        return f"{PREFIX.get(p, p)} {SUFFIX.get(s, s.replace('_', ' '))}"
    return TOP.get(role, role.replace("_", " "))


# yes/no attributes said in words the label tokens can attend to ("[P] firmness [L] wish [L] must")
FIELD_TEXT = {"hard": "firmness", "negate": "negated", "any": "which of them", "justification": "reason given",
              "period.count": "number of weeks", "period.relative": "period", "period.parity": "odd or even weeks"}
BOOL_TEXT = {"hard": ("wish", "must"), "negate": ("as said", "not"), "any": ("all of them", "any one")}


def value_text(field: str, v: str) -> str:
    if v == "none":
        return "none"
    x = json.loads(v)
    if isinstance(x, bool) and field in BOOL_TEXT:
        return BOOL_TEXT[field][int(x)]
    return str(x).lower().replace("_", " ")


def norm(t: str) -> str:
    return " ".join(t.lower().strip(" .,;:!?()'\"").split())


# --- data ------------------------------------------------------------------------------------------------

def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def build_schema(rows: list[dict]) -> dict:
    kinds = sorted({it["kind"] for r in rows for it in r["intents"]})
    sch = {}
    for k in kinds:
        roles, classes, refs = set(), defaultdict(set), set()
        for r in rows:
            for it in r["intents"]:
                if it["kind"] != k:
                    continue
                roles |= {s["role"] for s in it["spans"]}
                for f, v in it["classes"].items():
                    classes[f].add(json.dumps(v))
                refs |= set(it["refs"])
        sch[k] = {"roles": sorted(roles), "classes": {f: ["none"] + sorted(v) for f, v in sorted(classes.items())},
                  "refs": sorted(refs | ({"which"} if k in ("move", "cancel", "relate", "extra_class") else set()))}
    roles_all = sorted({f"{k}/{r}" for k, s in sch.items() for r in s["roles"]})
    return {"kinds": kinds, "per_kind": sch, "roles_all": roles_all}


class Encoder:
    """Builds the token sequences: label prompts first (never cut), then the context and the message."""

    def __init__(self, tok, schema: dict, max_len: int) -> None:
        self.tok, self.schema, self.max_len = tok, schema, max_len
        self.ids = {t: tok.convert_tokens_to_ids(t) for t in SPECIAL}

    def _encode(self, prompt: str, row: dict) -> dict:
        ctx = f" [CTX] {row['context'].get('role') or 'faculty'} [MSG] "
        head = prompt + ctx
        full = head + row["text"] + " [/MSG]"
        enc = self.tok(full, return_offsets_mapping=True, truncation=True, max_length=self.max_len)
        ms, me = len(head), len(head) + len(row["text"])
        offs = [tuple(o) for o in enc["offset_mapping"]]
        msg = [s >= ms and e <= me and e > s for s, e in offs]
        return {"input_ids": enc["input_ids"], "offs": [(s - ms, e - ms) for s, e in offs], "msg": msg}

    def _positions(self, ids: list[int], token: str) -> list[int]:
        return [i for i, x in enumerate(ids) if x == self.ids[token]]

    @staticmethod
    def char_to_tokens(x: dict, a: int, b: int) -> tuple[int, int] | None:
        toks = [i for i, (s, e) in enumerate(x["offs"]) if x["msg"][i] and e > a and s < b]
        return (toks[0], toks[-1]) if toks else None

    def pass1(self, row: dict) -> dict:
        kinds = self.schema["kinds"]
        prompt = ("[P] action " + " ".join(f"[L] {a.replace('_', ' ')}" for a in ACTIONS)
                  + " [P] choice " + " ".join(f"[L] {c.replace('_', ' ')}" for c in COMPOSITES)
                  + " [P] intent " + " ".join(f"[E] {k.replace('_', ' ')}: {GLOSS.get(k, k)}" for k in kinds))
        x = self._encode(prompt, row)
        L, E = self._positions(x["input_ids"], "[L]"), self._positions(x["input_ids"], "[E]")
        assert len(L) == len(ACTIONS) + len(COMPOSITES) and len(E) == len(kinds), "prompt cut"
        x.update(kind="p1", L_action=L[:len(ACTIONS)], L_comp=L[len(ACTIONS):], E_kind=E,
                 action=ACTIONS.index(row["action"]) if row["action"] in ACTIONS else 0,
                 comp=[float(any(c["kind"] == k for c in row.get("composites", []))) for k in COMPOSITES])
        T = len(x["input_ids"])
        st, en = [[0.0] * len(kinds) for _ in range(T)], [[0.0] * len(kinds) for _ in range(T)]
        for it in row["intents"]:
            if it.get("extent") and it["kind"] in kinds:
                tt = self.char_to_tokens(x, *it["extent"])
                if tt:
                    k = kinds.index(it["kind"])
                    st[tt[0]][k], en[tt[1]][k] = 1.0, 1.0
        x.update(start=st, end=en)
        return x

    def pass2(self, row: dict, kind: str, extent: tuple[int, int], gold: dict | None) -> dict:
        sk = self.schema["per_kind"][kind]
        prompt = f"[P] {kind.replace('_', ' ')} " + " ".join(f"[E] {role_name(r)}" for r in sk["roles"])
        for f, vals in sk["classes"].items():
            prompt += f" [P] {FIELD_TEXT.get(f, f.replace('.', ' ').replace('_', ' '))} " + " ".join(
                f"[L] {value_text(f, v)}" for v in vals)
        for f in sk["refs"]:
            prompt += f" [P] {PREFIX.get(f, f)} same as before [L] no [L] yes"
        x = self._encode(prompt, row)
        L, E = self._positions(x["input_ids"], "[L]"), self._positions(x["input_ids"], "[E]")
        assert len(E) == len(sk["roles"]), "prompt cut"
        groups, i = [], 0
        for f, vals in sk["classes"].items():
            groups.append(("class", f, L[i:i + len(vals)], vals))
            i += len(vals)
        for f in sk["refs"]:
            groups.append(("ref", f, L[i:i + 2], ["no", "yes"]))
            i += 2
        tt = self.char_to_tokens(x, *extent)
        tgt = [0] * len(x["input_ids"])
        if tt:
            for j in range(tt[0], tt[1] + 1):
                tgt[j] = 1
        x.update(kind="p2", intent_kind=kind, E_role=E, groups=groups, target=tgt, extent=list(extent),
                 role_idx=[self.schema["roles_all"].index(f"{kind}/{r}") for r in sk["roles"]])
        if gold is not None:
            spans = []
            for s in gold["spans"]:
                if s["role"] in sk["roles"]:
                    t = self.char_to_tokens(x, s["start"], s["end"])
                    if t and t[1] - t[0] < MAX_W:
                        spans.append((t[0], t[1], sk["roles"].index(s["role"])))
            gl = []
            for typ, f, _, vals in groups:
                if typ == "class":
                    v = json.dumps(gold["classes"][f]) if f in gold["classes"] else "none"
                    gl.append(vals.index(v) if v in vals else 0)
                else:
                    gl.append(1 if f in gold["refs"] else 0)
            x.update(spans=spans, group_gold=gl)
        return x


def usable(row: dict) -> bool:
    return row["aligned"] and all(it.get("extent") for it in row["intents"])


# --- model -----------------------------------------------------------------------------------------------

class Reader(nn.Module):
    """The label tokens' states give the queries, plus a learned vector per kind and per role: 7k rows
    are too few for new ``[E]`` tokens alone to tell the kinds apart (a 40-row overfit put every kind's
    start on the same token)."""

    def __init__(self, name: str, vocab: int, n_kinds: int, n_roles: int, d: int = 256, encoder=None) -> None:
        super().__init__()
        self.kind_emb, self.role_emb = nn.Embedding(n_kinds, d), nn.Embedding(n_roles, d)
        # fp32 master weights: newer transformers load DeBERTa-v3 as fp16, and GradScaler cannot unscale fp16
        # grads (Kaggle run 1); autocast still computes in fp16
        self.enc = (encoder if encoder is not None else AutoModel.from_pretrained(name)).float()
        self.enc.resize_token_embeddings(vocab)
        H = self.enc.config.hidden_size
        self.target = nn.Embedding(2, H)
        nn.init.zeros_(self.target.weight)
        mlp = lambda o: nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Dropout(0.1), nn.Linear(H, o))
        self.label = mlp(1)          # [L] -> score
        self.q_kind, self.q_role = mlp(d), mlp(d)
        self.tok_start, self.tok_end = nn.Linear(H, d), nn.Linear(H, d)
        self.width = nn.Embedding(MAX_W, H // 4)
        self.span = nn.Sequential(nn.Linear(2 * H + H // 4, H), nn.GELU(), nn.Dropout(0.1), nn.Linear(H, d))
        self.d = d

    def forward(self, ids, mask, target):
        emb = self.enc.get_input_embeddings()(ids) + self.target(target)
        return self.enc(inputs_embeds=emb, attention_mask=mask).last_hidden_state

    def pointers(self, h, E):  # (T, K) start and end logits
        q = self.q_kind(h[E]) + self.kind_emb.weight[:len(E)]
        return self.tok_start(h) @ q.T / math.sqrt(self.d), self.tok_end(h) @ q.T / math.sqrt(self.d)

    def spans(self, h, msg_idx, E, role_idx):  # candidate spans (i, j) and their (S, R) logits
        cand = [(i, j) for a, i in enumerate(msg_idx) for j in msg_idx[a:a + MAX_W]]
        if not cand:
            return cand, h.new_zeros((0, len(E)))
        i = torch.tensor([c[0] for c in cand], device=h.device)
        j = torch.tensor([c[1] for c in cand], device=h.device)
        w = (j - i).clamp(max=MAX_W - 1)
        rep = self.span(torch.cat([h[i], h[j], self.width(w)], -1))
        q = self.q_role(h[E]) + self.role_emb(torch.tensor(role_idx, device=h.device))
        return cand, rep @ q.T / math.sqrt(self.d)


def batchify(xs: list[dict], pad: int, device) -> tuple:
    T = max(len(x["input_ids"]) for x in xs)
    ids = torch.full((len(xs), T), pad, dtype=torch.long)
    mask = torch.zeros((len(xs), T), dtype=torch.long)
    tgt = torch.zeros((len(xs), T), dtype=torch.long)
    for b, x in enumerate(xs):
        n = len(x["input_ids"])
        ids[b, :n] = torch.tensor(x["input_ids"])
        mask[b, :n] = 1
        if "target" in x:
            tgt[b, :n] = torch.tensor(x["target"])
    return ids.to(device), mask.to(device), tgt.to(device)


# Pointer and span losses are summed per kind / per role, not averaged over every (token or span) x label
# pair: averaged, a confidently wrong role among ~600 easy negatives moved the loss by 0.005 and was never
# fixed (a 12-row overfit at loss 0.007 still read "Friday" as the target day). Summed, no pos_weight is
# needed and probabilities stay calibrated (no shift when decoding).
SHIFT = 0.0


def loss_of(model: Reader, h, x: dict) -> torch.Tensor:
    n = len(x["input_ids"])
    h = h[:n].float()
    if x["kind"] == "p1":
        la = model.label(h[x["L_action"]]).squeeze(-1)
        loss = F.cross_entropy(la[None], torch.tensor([x["action"]], device=h.device))
        lc = model.label(h[x["L_comp"]]).squeeze(-1)
        loss = loss + F.binary_cross_entropy_with_logits(lc, torch.tensor(x["comp"], device=h.device))
        s, e = model.pointers(h, x["E_kind"])
        m = torch.tensor(x["msg"], device=h.device)
        K = len(x["E_kind"])
        for M, key in ((s, "start"), (e, "end")):
            y = torch.tensor(x[key], device=h.device)[m]
            loss = loss + F.binary_cross_entropy_with_logits(M[m], y, reduction="sum") / K
        return loss
    msg_idx = [i for i, ok in enumerate(x["msg"]) if ok]
    loss = h.new_zeros(())
    if x["E_role"]:
        cand, logits = model.spans(h, msg_idx, x["E_role"], x["role_idx"])
        y = torch.zeros_like(logits)
        index = {c: k for k, c in enumerate(cand)}
        for i, j, r in x["spans"]:
            if (i, j) in index:
                y[index[(i, j)], r] = 1.0
        loss = loss + F.binary_cross_entropy_with_logits(logits, y, reduction="sum") / len(x["E_role"])
    for (typ, f, L, vals), g in zip(x["groups"], x["group_gold"]):
        lg = model.label(h[L]).squeeze(-1)
        loss = loss + F.cross_entropy(lg[None], torch.tensor([g], device=h.device))
    return loss


# --- decoding --------------------------------------------------------------------------------------------

def decode_brackets(ps: torch.Tensor, pe: torch.Tensor, msg_idx: list[int], th: float = 0.5):
    """Extents of one kind from start and end probabilities, paired like brackets; and the most likely
    start or end left out (for conf)."""
    events = []
    for i in msg_idx:
        if ps[i] >= th:
            events.append((i, 0, float(ps[i])))
        if pe[i] >= th:
            events.append((i, 1, float(pe[i])))
    stack, out = [], []
    for i, typ, p in events:
        if typ == 0:
            stack.append((i, p))
        elif stack:
            s, p0 = stack.pop()
            out.append((s, i, min(p0, p)))
    rest = [float(ps[i]) for i in msg_idx if ps[i] < th] + [float(pe[i]) for i in msg_idx if pe[i] < th]
    return sorted(out), max(rest, default=0.0)


@torch.no_grad()
def predict(model: Reader, enc: Encoder, rows: list[dict], device, bs: int = 32) -> list[dict]:
    model.eval()
    pad = enc.tok.pad_token_id
    kinds = enc.schema["kinds"]
    out = []
    for b0 in range(0, len(rows), bs):
        chunk = rows[b0:b0 + bs]
        xs = [enc.pass1(r) for r in chunk]
        ids, mask, tgt = batchify(xs, pad, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            H = model(ids, mask, tgt)
        jobs = []
        for b, (r, x) in enumerate(zip(chunk, xs)):
            h = H[b, :len(x["input_ids"])].float()
            pa = model.label(h[x["L_action"]]).squeeze(-1).softmax(-1)
            pc = model.label(h[x["L_comp"]]).squeeze(-1).sigmoid()
            s, e = model.pointers(h, x["E_kind"])
            s, e = (s - SHIFT).sigmoid(), (e - SHIFT).sigmoid()  # calibrated: trained with pos_weight
            msg_idx = [i for i, ok in enumerate(x["msg"]) if ok]
            confs = [float(pa.max())] + [float(max(p, 1 - p)) for p in pc]
            found = []
            for k, kind in enumerate(kinds):
                ext, left = decode_brackets(s[:, k], e[:, k], msg_idx)
                confs.append(1 - left)
                for i, j, p in ext:
                    found.append((x["offs"][i][0], x["offs"][j][1], kind, p))
                    confs.append(p)
            found.sort()
            pred = {"id": r["id"], "action": ACTIONS[int(pa.argmax())],
                    "composites": sorted(COMPOSITES[k] for k, p in enumerate(pc) if p >= 0.5), "intents": []}
            out.append((pred, confs))
            for a, z, kind, _ in found:
                jobs.append((len(out) - 1, r, kind, (a, z)))
        # pass 2 for every found request
        for j0 in range(0, len(jobs), bs):
            part = jobs[j0:j0 + bs]
            xs = [enc.pass2(r, kind, ext, None) for _, r, kind, ext in part]
            ids, mask, tgt = batchify(xs, pad, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                H = model(ids, mask, tgt)
            for b, ((k, r, kind, ext), x) in enumerate(zip(part, xs)):
                h = H[b, :len(x["input_ids"])].float()
                pred, confs = out[k]
                sk = enc.schema["per_kind"][kind]
                spans = []
                msg_idx = [i for i, ok in enumerate(x["msg"]) if ok]
                if x["E_role"]:
                    cand, lg = model.spans(h, msg_idx, x["E_role"], x["role_idx"])
                    p = (lg - SHIFT).sigmoid()
                    for ri, role in enumerate(sk["roles"]):
                        order = sorted(range(len(cand)), key=lambda c: -float(p[c, ri]))
                        kept = []
                        for c in order:
                            pc_ = float(p[c, ri])
                            if pc_ < 0.5:
                                confs.append(1 - pc_)  # the likeliest span left out
                                break
                            i, j = cand[c]
                            if any(not (j < a or i > b_) for a, b_ in kept):
                                continue
                            kept.append((i, j))
                            confs.append(pc_)
                            a_, z_ = x["offs"][i][0], x["offs"][j][1]
                            spans.append({"role": role, "start": a_, "end": z_, "text": r["text"][a_:z_]})
                classes, refs = {}, {}
                for typ, f, L, vals in x["groups"]:
                    pr = model.label(h[L]).squeeze(-1).softmax(-1)
                    confs.append(float(pr.max()))
                    v = vals[int(pr.argmax())]
                    if typ == "class" and v != "none":
                        classes[f] = json.loads(v)
                    elif typ == "ref" and v == "yes":
                        refs[f] = True
                pred["intents"].append({"kind": kind, "extent": list(ext), "spans": spans, "classes": classes,
                                        "refs": refs})
    res = []
    for pred, confs in out:
        pred["conf"] = round(min(confs), 4)
        res.append(pred)
    return res


# --- scoring ---------------------------------------------------------------------------------------------

def shape(intent: dict) -> tuple:
    spans = tuple(sorted((s["role"], norm(s["text"])) for s in intent["spans"]))
    return (intent["kind"], spans, tuple(sorted((k, json.dumps(v)) for k, v in intent["classes"].items())),
            tuple(sorted(intent["refs"])))


def correct(gold: dict, pred: dict) -> bool:
    if gold["action"] != pred["action"]:
        return False
    if sorted(c["kind"] for c in gold.get("composites", [])) != pred["composites"]:
        return False
    g = sorted(gold["intents"], key=lambda it: (it["extent"] or [10 ** 6])[0])
    p = pred["intents"]
    return len(g) == len(p) and all(shape(a) == shape(b) for a, b in zip(g, p, strict=True))


def score(gold_rows: list[dict], preds: list[dict]) -> dict:
    ok = [correct(g, p) for g, p in zip(gold_rows, preds, strict=True)]
    conf = [p["conf"] for p in preds]
    n = len(ok)
    res = {"rows": n, "exact": round(sum(ok) / max(n, 1), 4),
           "action": round(sum(g["action"] == p["action"] for g, p in zip(gold_rows, preds)) / max(n, 1), 4),
           "unaligned_gold": sum(not g["aligned"] for g in gold_rows)}
    by = defaultdict(list)
    for g, o in zip(gold_rows, ok):
        by[g.get("tier", "?")].append(o)
    res["by_tier"] = {t: f"{sum(v)}/{len(v)}" for t, v in sorted(by.items())}
    gate = {}
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98):
        kept = [o for o, c in zip(ok, conf) if c >= tau]
        gate[str(tau)] = {"coverage": round(len(kept) / max(n, 1), 4),
                          "accuracy": round(sum(kept) / max(len(kept), 1), 4)}
    res["gate"] = gate
    return res


# --- training --------------------------------------------------------------------------------------------

def examples(enc: Encoder, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    p1, p2 = [], []
    for r in rows:
        if not usable(r):
            continue
        p1.append(enc.pass1(r))
        for it in r["intents"]:
            p2.append(enc.pass2(r, it["kind"], tuple(it["extent"]), it))
    return p1, p2


def load_gliner2(repo: str):
    """(tokenizer, encoder) of a GLiNER2 checkpoint (``pip install gliner2``)."""
    from gliner2 import GLiNER2

    g = GLiNER2.from_pretrained(repo)
    tok, encoder = g.processor.tokenizer, g.encoder
    if not getattr(tok, "is_fast", False):  # offsets need a fast tokenizer
        tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
    return tok, encoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="folder with train.jsonl, val.jsonl, test.jsonl, ...")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--model", default="microsoft/deberta-v3-base",
                    help="a HF encoder, or gliner2:<repo> for a GLiNER2 checkpoint's encoder and tokenizer")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: rows per split")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    try:  # DeBERTa-v3's fast tokenizer is converted from its sentencepiece model
        import sentencepiece  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "protobuf"], check=False)
    if args.data is None:  # the Kaggle dataset, wherever it is mounted
        hits = sorted(Path("/kaggle/input").rglob("train.jsonl"))
        hits = [h for h in hits if (h.parent / "composed.jsonl").exists()]
        args.data = str(hits[0].parent)
    data, out = Path(args.data), Path(args.out)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")

    def say(**kw):
        kw["t"] = round(time.time() - t0, 1)
        print(json.dumps(kw), flush=True)
        log.write(json.dumps(kw) + "\n")
        log.flush()

    t0 = time.time()
    lim = (lambda xs: xs[:args.limit]) if args.limit else (lambda xs: xs)
    train, val = lim(load(data / "train.jsonl")), lim(load(data / "val.jsonl"))
    schema = build_schema(train)
    (out / "schema.json").write_text(json.dumps(schema, indent=1), encoding="utf-8")
    encoder, before = None, None
    if args.model.startswith("gliner2:"):
        # GLiNER2's encoder (a DeBERTa-v3 trained on label-prompt extraction) and its tokenizer, whose [P], [L]
        # and [E] are already trained: only [CTX], [MSG], [/MSG] are new
        tok, encoder = load_gliner2(args.model.split(":", 1)[1])
        before = len(tok)
    else:
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    known = [t for t in SPECIAL if tok.convert_tokens_to_ids(t) not in (None, tok.unk_token_id)]
    if encoder is None:  # as in v1
        tok.add_special_tokens({"additional_special_tokens": SPECIAL})
    else:  # keep GLiNER2's own special tokens and add ours beside them
        tok.add_tokens([t for t in SPECIAL if t not in known], special_tokens=True)
    enc = Encoder(tok, schema, args.max_len)
    model = Reader(args.model, len(tok), len(schema["kinds"]), len(schema["roles_all"]), encoder=encoder).to(device)
    say(event="tokens", pretrained_labels=known, new=[t for t in SPECIAL if t not in known], vocab_before=before,
        vocab=len(tok))
    p1, p2 = examples(enc, train)
    say(event="data", device=str(device), model=args.model, train_rows=len(train),
        usable=sum(map(usable, train)), pass1=len(p1), pass2=len(p2), val=len(val),
        longest=max(len(x["input_ids"]) for x in p1 + p2), kinds=schema["kinds"])

    enc_params = list(model.enc.parameters())
    ids_enc = {id(p) for p in enc_params}
    head = [p for p in model.parameters() if id(p) not in ids_enc]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr}, {"params": head, "lr": args.head_lr}],
                            weight_decay=0.01)
    steps_per_epoch = math.ceil(len(p1) / args.bs) + math.ceil(len(p2) / args.bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps_per_epoch * args.epochs),
                                            steps_per_epoch * args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    pad = tok.pad_token_id
    best, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        batches = [p1[i:i + args.bs] for i in range(0, len(p1), args.bs)]
        random.shuffle(p2)
        batches += [p2[i:i + args.bs] for i in range(0, len(p2), args.bs)]
        random.shuffle(batches)
        tot, n = 0.0, 0
        for step, xs in enumerate(batches, 1):
            ids, mask, tgt = batchify(xs, pad, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                H = model(ids, mask, tgt)
            loss = sum(loss_of(model, H[b], x) for b, x in enumerate(xs)) / len(xs)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss)
            n += 1
            if step % 200 == 0:
                say(event="step", epoch=ep, step=step, of=len(batches), loss=round(tot / n, 4))
        preds = predict(model, enc, val, device)
        sc = score(val, preds)
        say(event="epoch", epoch=ep, train_loss=round(tot / max(n, 1), 4), val=sc)
        if sc["exact"] > best:
            best = sc["exact"]
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            best_epoch = ep
    model.load_state_dict(best_state)
    torch.save({"state": {k: v.half() if v.is_floating_point() else v for k, v in best_state.items()},
                "model": args.model, "special": SPECIAL, "epoch": best_epoch}, out / "model.pt")
    tok.save_pretrained(out / "tokenizer")

    metrics = {"best_epoch": best_epoch, "args": vars(args)}
    splits = ["val", "test", "composed"] + sorted(p.stem for p in data.glob("suite-*.jsonl"))
    for name in splits:
        path = data / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = lim(load(path))
        t1 = time.time()
        preds = predict(model, enc, rows, device)
        metrics[name] = score(rows, preds)
        metrics[name]["ms_per_row"] = round(1000 * (time.time() - t1) / max(len(rows), 1), 1)
        with (out / "predictions" / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for g, p in zip(rows, preds):
                p["ok"] = correct(g, p)
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        say(event="eval", split=name, exact=metrics[name]["exact"], tiers=metrics[name]["by_tier"])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    say(event="done")


if __name__ == "__main__" and not os.environ.get("NTS_NO_MAIN"):  # set by make_kernel's GLiNER2 kernel
    main()
