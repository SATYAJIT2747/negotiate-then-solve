"""System One decision model (proposal Section 8.2, step 1).

A small, fast, calibrated classifier over the request text: request type,
the next action (including refusal for missing authority), and a
confidence ``p``. Requests with ``p >= tau`` on a compilable type take the
fast path; everything else goes to the System Two LLM.

This is the proposal's fallback design (TF-IDF + logistic regression, the
"calibrated small classifier" in the risk table). SemIf, Kev or OpenJev
Verdict can replace it behind the same ``decide`` interface.

``model`` picks the classifier, all without torch (``evaluation.system_one``
compares them):

* ``logreg``: TF-IDF words and letter chunks, logistic regression;
* ``svm``: the same features, a linear SVM with sigmoid-calibrated
  probabilities (the default: the most accurate on wording it has not seen,
  and the best calibrated);
* ``embed``: bge-small sentence embeddings (fastembed, ONNX) and the sender's
  role and channel, logistic regression;
* ``hybrid``: TF-IDF and the embeddings side by side, logistic regression.

The request corpus has no swaps, moves, cancellations, extra classes, club
bookings or withdrawals, so a model trained on it alone calls them preferences
and sends most of them down the fast path, which cannot read them.
``intent_examples`` adds those kinds from the intent data
(``data/compiler-intents``) under their own types, none of which is a
fast-path type.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from core.schemas import Channel, Request, Role

from .corpus import CorpusExample, ExpectedAction, RequestType

FAST_PATH_TYPES = frozenset({RequestType.PREFERENCE, RequestType.UNAVAILABILITY, RequestType.ROOM_ISSUE})
MODELS = ("logreg", "svm", "embed", "hybrid")
DEFAULT_MODEL = "svm"
INTENT_DIR = Path(__file__).resolve().parents[2] / "data" / "compiler-intents"
# intent kinds the request corpus has no example of -> the System One type they are learnt under
INTENT_TYPES = {
    "swap": RequestType.SWAP,
    "move": RequestType.MOVE,
    "cancel": RequestType.CANCEL,
    "extra_class": RequestType.EXTRA_CLASS,
    "club_booking": RequestType.CLUB_BOOKING,
    "club_more": RequestType.CLUB_BOOKING,
    "withdraw": RequestType.WITHDRAW,
    "avoid_between": RequestType.AVOID_BETWEEN,
    "load_limit": RequestType.LOAD_LIMIT,
    "unavailable_relative": RequestType.UNAVAILABLE_RELATIVE,
}


@dataclass
class Labelled:
    """A message with what System One should say about it. ``request_type`` / ``expected_action`` are None for
    a kind System One has no label for. ``kind`` is the intent data's own label."""

    request: Request
    request_type: RequestType | None
    expected_action: ExpectedAction | None
    kind: str
    split: str | None = None


def read_intents(path: Path, split: str | None = None) -> list[Labelled]:
    """Intent-data rows (chat examples for the intent reader) as labelled messages: the message, the sender's
    role and the channel from the prompt; the type from the row's kind, the action from the gold answer."""
    corpus_types = {t.value for t in RequestType} - {t.value for t in INTENT_TYPES.values()}
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        prompt = row["messages"][1]["content"]
        text = re.search(r"<message>\n(.*)\n</message>", prompt, re.S)[1]
        role = re.search(r"^Sender: .*\(([^)]+)\)$", prompt, re.M)[1].replace(" ", "_")
        channel = re.search(r"^Message \((\w+)\):", prompt, re.M)[1]
        request = Request(id=row.get("id", f"T-{n}"), channel=Channel(channel), sender_id="T-0", role=Role(role),
                          raw_text=text, received_at=datetime(2026, 9, 1))
        kind = row["kind"]
        rtype = INTENT_TYPES.get(kind) or (RequestType(kind) if kind in corpus_types else None)
        action = json.loads(row["messages"][2]["content"]).get("action") if rtype else None
        out.append(Labelled(request, rtype, ExpectedAction(action) if action else None, kind, split))
    return out


def intent_examples(split: str, directory: Path = INTENT_DIR) -> list[Labelled]:
    """The ``split`` rows of the kinds in ``INTENT_TYPES``: what the request corpus is missing. The kinds it has
    are left out, since the intent data labels them differently (a rule-breaking request is ``compile`` there,
    ``deny`` in the corpus).

    They teach the type only (``expected_action`` is None): their own type already keeps them off the fast
    path, and their actions (922 of 994 ``compile``, students' club bookings among them) taught the action
    model to deny and refuse less (unseen wording: 95.4% -> 88.7%)."""
    path = directory / f"{split}.jsonl"
    if not path.exists():
        return []
    return [Labelled(e.request, e.request_type, None, e.kind, split)
            for e in read_intents(path, split) if e.kind in INTENT_TYPES]


@dataclass
class Decision:
    request_type: RequestType
    type_p: float
    action: ExpectedAction
    action_p: float
    latency_ms: float

    @property
    def confidence(self) -> float:
        return min(self.type_p, self.action_p)

    def fast_path(self, tau: float) -> bool:
        return (
            self.request_type in FAST_PATH_TYPES
            and self.action == ExpectedAction.COMPILE
            and self.confidence >= tau
        )


def _features(request: Request) -> str:
    # The role is part of the input: the same words mean different things
    # from a student than from the faculty member who owns the class.
    return f"__role_{request.role.value}__ __channel_{request.channel.value}__ {request.raw_text}"


_TAGS = re.compile(r"^__role_(\w+?)__ __channel_(\w+?)__ ")
_ROLES = [r.value for r in Role]
_CHANNELS = [c.value for c in Channel]


class Embeddings(BaseEstimator, TransformerMixin):
    """The message's sentence embedding, then the role and channel one-hot (an embedding of the tags would
    blur them, and the role decides refusals). Embeddings are kept by text: a benchmark reads the same
    message under several models."""

    _seen: dict[str, np.ndarray] = {}

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        from agents.policy import embedder

        parts = [_TAGS.match(s) for s in x]
        texts = [s[m.end():] if m else s for s, m in zip(x, parts, strict=True)]
        if todo := list(dict.fromkeys(t for t in texts if t not in self._seen)):
            for t, v in zip(todo, embedder().embed(todo), strict=True):
                self._seen[t] = np.asarray(v, dtype=np.float32)
        tags = np.zeros((len(x), len(_ROLES) + len(_CHANNELS)), dtype=np.float32)
        for i, m in enumerate(parts):
            if m and m[1] in _ROLES:
                tags[i, _ROLES.index(m[1])] = 1
            if m and m[2] in _CHANNELS:
                tags[i, len(_ROLES) + _CHANNELS.index(m[2])] = 1
        return sparse.csr_matrix(np.hstack([np.stack([self._seen[t] for t in texts]), tags]))


def _tfidf() -> list[tuple[str, TfidfVectorizer]]:
    return [
        ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
    ]


def _pipeline(c: float, model: str = "logreg") -> Pipeline:
    if model == "logreg":
        return Pipeline([("features", FeatureUnion(_tfidf())), ("clf", LogisticRegression(C=c, max_iter=5000))])
    if model == "svm":
        # LinearSVC gives no probabilities: sigmoid calibration over 3 folds (the rarest label has ~30 examples)
        svm = CalibratedClassifierCV(LinearSVC(C=c / 8), method="sigmoid", cv=3)
        return Pipeline([("features", FeatureUnion(_tfidf())), ("clf", svm)])
    if model == "embed":
        return Pipeline([("features", Embeddings()), ("clf", LogisticRegression(C=c, max_iter=5000))])
    if model == "hybrid":
        return Pipeline([("features", FeatureUnion([*_tfidf(), ("meaning", Embeddings())])),
                         ("clf", LogisticRegression(C=c, max_iter=5000))])
    raise ValueError(f"unknown System One model {model!r}; one of {', '.join(MODELS)}")


class SystemOne:
    def __init__(self, c: float = 4.0, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self.type_model = _pipeline(c, model)
        self.action_model = _pipeline(c, model)

    def fit(self, examples: Sequence[CorpusExample | Labelled]) -> SystemOne:
        """Each model on the examples that have its label: one with no action (``intent_examples``) only
        teaches the type."""
        typed = [ex for ex in examples if ex.request_type is not None]
        self.type_model.fit([_features(ex.request) for ex in typed], [ex.request_type.value for ex in typed])
        acted = [ex for ex in examples if ex.expected_action is not None]
        self.action_model.fit([_features(ex.request) for ex in acted], [ex.expected_action.value for ex in acted])
        return self

    def decide(self, request: Request) -> Decision:
        start = time.perf_counter()
        x = [_features(request)]
        tp = self.type_model.predict_proba(x)[0]
        ap = self.action_model.predict_proba(x)[0]
        ti, ai = int(np.argmax(tp)), int(np.argmax(ap))
        return Decision(
            request_type=RequestType(self.type_model.classes_[ti]),
            type_p=float(tp[ti]),
            action=ExpectedAction(self.action_model.classes_[ai]),
            action_p=float(ap[ai]),
            latency_ms=(time.perf_counter() - start) * 1000,
        )


def tune_tau(
    decisions: Sequence[Decision],
    examples: Sequence[CorpusExample],
    *,
    min_precision: float = 0.98,
    grid: Sequence[float] = tuple(np.round(np.arange(0.30, 1.0, 0.02), 2)),
) -> float:
    """Smallest threshold whose fast-path routes are right at least
    ``min_precision`` of the time on the validation set. An example with no
    action (a kind the fast path cannot read) is never right there.

    Until the QLoRA compiler exists, "right" means System One's type and
    action are both correct; the proposal's criterion (compilation F1 within
    2 points of the LLM-only path) replaces this once it does.
    """
    for tau in sorted(grid):
        routed = [(d, ex) for d, ex in zip(decisions, examples, strict=True) if d.fast_path(tau)]
        if not routed:
            return tau
        correct = sum(d.request_type == ex.request_type and d.action == ex.expected_action for d, ex in routed)
        if correct / len(routed) >= min_precision:
            return tau
    return 1.0
