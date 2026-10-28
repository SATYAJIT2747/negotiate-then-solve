"""Typed tools the model calls (``Registry``), grouped by what they may touch (``Scope``)."""

from .draft import DRAFT_TOOLS
from .end import END_TOOLS
from .evaluate import EVALUATE_TOOLS
from .extend import EXTEND_TOOLS
from .read import READ_TOOLS
from .registry import Context, Refusal, Registry, Scope, Tool, tool
from .rules import POLICY_TOOLS
from .talk import TALK_TOOLS, Ask, Reply, settle

ALL_TOOLS = [*READ_TOOLS, *POLICY_TOOLS, *DRAFT_TOOLS, *EXTEND_TOOLS, *EVALUATE_TOOLS, *TALK_TOOLS, *END_TOOLS]

__all__ = ["ALL_TOOLS", "DRAFT_TOOLS", "END_TOOLS", "EVALUATE_TOOLS", "EXTEND_TOOLS", "POLICY_TOOLS", "READ_TOOLS", "TALK_TOOLS", "Ask", "Context",
           "Refusal", "Registry", "Reply", "Scope", "Tool", "settle", "tool"]
