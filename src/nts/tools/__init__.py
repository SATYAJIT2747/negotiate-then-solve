"""Typed tools the model calls (``Registry``), grouped by what they may touch (``Scope``)."""

from .draft import DRAFT_TOOLS
from .end import END_TOOLS
from .evaluate import EVALUATE_TOOLS
from .read import READ_TOOLS
from .registry import Context, Refusal, Registry, Scope, Tool, tool

ALL_TOOLS = [*READ_TOOLS, *DRAFT_TOOLS, *EVALUATE_TOOLS, *END_TOOLS]

__all__ = ["ALL_TOOLS", "DRAFT_TOOLS", "END_TOOLS", "EVALUATE_TOOLS", "READ_TOOLS", "Context", "Refusal", "Registry",
           "Scope", "Tool", "tool"]
