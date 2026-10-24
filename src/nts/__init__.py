"""Negotiate, Then Solve, rebuilt as an agent loop (docs/agentic.md).

kernel     the timetabling model, the CP-SAT solver and the validator (the only code that decides a timetable)
state      the department's state as an append-only event log, with drafts the agent works in
tools      typed tools the model calls: read the state, change a draft, check it, talk, finish
models     one interface for every model: llama.cpp, Gemini, replayed and scripted ones
protocol   the messages, tool calls and results every layer passes around (and telemetry records)
"""
