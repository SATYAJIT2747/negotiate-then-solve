"""Negotiate, Then Solve, rebuilt as an agent loop (docs/agentic.md).

kernel     the timetabling model, the CP-SAT solver and the validator (the only code that decides a timetable)
state      the department's state as an append-only event log, with drafts the agent works in
tools      typed tools the model calls: read the state, change a draft, check it, talk, finish
models     one interface for every model: llama.cpp, Gemini, replayed and scripted ones
harness    the loop: ask the model, run its tool calls, feed back the results, within budgets
telemetry  every episode, turn, model call and tool call as spans (JSONL + SQLite)
agents     the agents, as configurations: prompt, tool scopes, budgets
protocol   the messages, tool calls and results every layer passes around (and telemetry records)
"""
