import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from google.genai import types

from nts.models import GeminiChat, ModelUnavailable, OpenAIChat
from nts.models.gemini import from_gemini, to_gemini
from nts.protocol import UNPARSED, Message, ToolCall


@pytest.fixture
def server():
    """A fake llama-server: records each request, answers with the next canned reply."""
    seen, replies = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            seen.append((self.path, None))
            self._send([{"id": 0, "path": "loras\\nts-read100.gguf"}, {"id": 1, "path": "loras\\nts-agent-v1.gguf"}])

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            self._send(replies.pop(0))

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", seen, replies
    httpd.shutdown()


def test_openai_chat_sends_tools_and_parses_calls(server):
    url, seen, replies = server
    replies.append({"model": "qwen", "usage": {"prompt_tokens": 50, "completion_tokens": 9},
                    "choices": [{"message": {"content": "", "tool_calls": [
                        {"id": "a", "type": "function", "function": {"name": "find", "arguments": '{"text": "iyer"}'}},
                        {"id": "b", "type": "function", "function": {"name": "find", "arguments": '{"text": '}}]}}]})
    model = OpenAIChat(url, adapter="nts-agent-v1")
    history = [Message(role="system", content="s"), Message(role="user", content="q"),
               Message(role="assistant", tool_calls=[ToolCall(id="x", name="describe", arguments={"what": "rooms"})]),
               Message(role="tool", call_id="x", name="describe", content="[]")]
    turn = model.chat(history, [{"type": "function", "function": {"name": "find"}}])
    assert [c.arguments for c in turn.tool_calls] == [{"text": "iyer"}, {UNPARSED: '{"text": '}]
    assert turn.usage.prompt_tokens == 50 and json.loads(turn.raw)["tool_calls"][0]["id"] == "a"
    assert seen[0][0] == "/lora-adapters"
    payload = seen[1][1]
    assert payload["lora"] == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}]
    assert payload["tools"][0]["function"]["name"] == "find"
    assert payload["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"what": "rooms"}'
    assert payload["messages"][3] == {"role": "tool", "tool_call_id": "x", "content": "[]"}
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_openai_chat_reports_a_missing_adapter_and_a_dead_server(server):
    url, _, _ = server
    with pytest.raises(ModelUnavailable, match="not loaded"):
        OpenAIChat(url, adapter="nts-other").chat([Message(role="user", content="q")], [])
    with pytest.raises(ModelUnavailable, match="not reachable"):
        OpenAIChat("http://127.0.0.1:9/v1", timeout=2).chat([Message(role="user", content="q")], [])


def test_gemini_messages_and_signed_parts_round_trip():
    signed = types.Part(function_call=types.FunctionCall(id="g1", name="find", args={"text": "iyer"}),
                        thought_signature=b"sig")
    resp = types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[signed]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=40, candidates_token_count=6))
    turn = from_gemini(resp, "gemini-x", 1.0)
    assert turn.tool_calls == [ToolCall(id="g1", name="find", arguments={"text": "iyer"})]
    assert turn.usage.prompt_tokens == 40

    history = [Message(role="system", content="be brief"), Message(role="user", content="q"), turn.as_message(),
               Message(role="tool", call_id="g1", name="find", content='[{"kind":"teacher","name":"Dr. Iyer"}]'),
               Message(role="tool", call_id="g2", name="find", content='{"error":{"code":"unknown_name"}}')]
    system, contents = to_gemini(history)
    assert system == "be brief" and [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].thought_signature == b"sig"  # sent back exactly as it came
    answers = contents[2].parts
    assert answers[0].function_response.response == {"output": [{"kind": "teacher", "name": "Dr. Iyer"}]}
    assert answers[1].function_response.response == {"error": {"code": "unknown_name"}}


def test_gemini_chat_with_a_fake_client():
    class Models:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config):
            self.calls.append((model, contents, config))
            part = types.Part(text="Hello")
            return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model",
                                                                                                    parts=[part]))])

    class Client:
        models = Models()

    model = GeminiChat("gemini-x", client=Client())
    tools = [{"type": "function", "function": {"name": "find", "description": "d",
                                               "parameters": {"type": "object", "properties": {}}}}]
    turn = model.chat([Message(role="system", content="s"), Message(role="user", content="hi")], tools)
    assert turn.text == "Hello" and turn.tool_calls == []
    _, _, config = Client.models.calls[0]
    assert config.tools[0].function_declarations[0].name == "find"
    assert config.automatic_function_calling.disable is True


class _Models:
    def __init__(self, name, fail=None):
        self.name, self.fail, self.calls = name, list(fail or []), 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        if self.fail:
            raise self.fail.pop(0)
        return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part(text=self.name)]))])


class _Client:
    def __init__(self, name, fail=None):
        self.models = _Models(name, fail)


def _limit(day: bool):
    from google.genai import errors

    quota = "GenerateRequestsPerDayPerProjectPerModel-FreeTier" if day else "GenerateRequestsPerMinutePerProjectPerModel"
    return errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                              "message": f"Quota exceeded: {quota}. Please retry in 0.01s."}})


def test_gemini_keys_rotate_and_spent_keys_drop_out():
    a, b = _Client("a", [_limit(day=True)]), _Client("b")
    model = GeminiChat("gemini-x", clients=[a, b], rpm=60_000)
    q = [Message(role="user", content="hi")]
    assert model.chat(q, []).text == "b"  # a's daily quota is spent: b answers
    assert [k.spent for k in model.keys] == [True, False]
    assert {model.chat(q, []).text for _ in range(3)} == {"b"}
    assert a.models.calls == 1


def test_gemini_per_minute_limit_moves_to_another_key():
    a, b = _Client("a", [_limit(day=False)]), _Client("b")
    model = GeminiChat("gemini-x", clients=[a, b], rpm=60_000)
    q = [Message(role="user", content="hi")]
    assert model.chat(q, []).text == "b"
    assert not any(k.spent for k in model.keys)
    a_key, b_key = model.keys
    assert a_key.ready_at - b_key.ready_at > 0.9  # a rests for the delay it asked for (+1 s); b takes the calls


def test_gemini_all_keys_spent():
    model = GeminiChat("gemini-x", clients=[_Client("a", [_limit(day=True)])], rpm=60_000)
    with pytest.raises(RuntimeError, match="daily quota of all 1 key"):
        model.chat([Message(role="user", content="hi")], [])


def test_api_keys_from_the_environment(monkeypatch):
    from nts.models.gemini import api_keys

    for k in [k for k in __import__("os").environ if k.startswith("GEMINI_API_KEY")]:
        monkeypatch.delenv(k)
    monkeypatch.setenv("GEMINI_API_KEY", "k1, k2;k3")
    monkeypatch.setenv("GEMINI_API_KEY_2", "k5")
    monkeypatch.setenv("GEMINI_API_KEY_1", "k4")
    monkeypatch.setenv("GEMINI_API_KEYS", "k1")
    assert api_keys() == ["k1", "k2", "k3", "k4", "k5"]
