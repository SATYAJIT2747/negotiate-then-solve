"""Models behind one interface (``Model``): llama.cpp, Gemini, replayed and scripted ones."""

from .base import Model
from .scripted import ScriptedModel, ScriptExhausted, call, say

__all__ = ["Model", "ScriptExhausted", "ScriptedModel", "call", "say"]
