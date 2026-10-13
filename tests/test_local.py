"""Local client against a fake OpenAI-compatible server (no GPU needed)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agents.negotiation import _ParsedReply
from agents.policy import PolicyOutput
from core.generator import generate_department
from evaluation.metrics import exact_match
from language.compiler import CompilerParser, completion, gold_output
from language.corpus import ExpectedAction, generate_corpus
from language.llm import LLMError
from language.local import LocalClient, grammar_schema
from language.parsing import ParseOutput


class FakeServer:
    """Serves /v1/models (and /lora-adapters when ``loras`` is set) and answers every chat request with
    ``reply``."""

    def __init__(self) -> None:
        self.reply = "{}"
        self.models = ["/models/qwen3.5-4b-nts.Q5_K_M.gguf"]
        self.loras: list[str] | None = None  # adapter paths, as llama-server --lora lists them
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, body, status: int = 200) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/lora-adapters":
                    if fake.loras is None:
                        return self._send({"error": "not found"}, 404)
                    return self._send([{"id": i, "path": p, "scale": 0.0} for i, p in enumerate(fake.loras)])
                self._send({"data": [{"id": m} for m in fake.models]})

            def do_POST(self):
                fake.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self._send({"choices": [{"message": {"content": fake.reply}}],
                            "usage": {"prompt_tokens": 800, "completion_tokens": 60}})

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.httpd.shutdown()


def test_compiler_parser_through_local_client(server, tmp_path):
    instance = generate_department(seed=0)
    ex = next(e for e in generate_corpus(instance, n=100, seed=0) if e.expected_action == ExpectedAction.COMPILE)
    server.reply = completion(gold_output(ex))
    client = LocalClient(base_url=server.url, cache_dir=tmp_path)
    assert client.model == "local:qwen3.5-4b-nts.Q5_K_M.gguf"

    result = CompilerParser(instance, client).parse(ex.request)
    assert result.action == ExpectedAction.COMPILE and exact_match(result.constraints, ex.targets, instance)

    sent = server.requests[0]
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["response_format"]["type"] == "json_schema" and sent["temperature"] == 0.0
    assert ex.request.raw_text in sent["messages"][1]["content"]
    assert client.usage.calls == 1 and client.usage.prompt_tokens == 800

    CompilerParser(instance, client).parse(ex.request)  # second time from the cache
    assert len(server.requests) == 1 and client.usage.cache_hits == 1


def test_grammar_keeps_training_key_order():
    # llama.cpp writes required keys first; cited_rules must stay before explanation
    policy = grammar_schema(PolicyOutput.model_json_schema())
    assert policy["required"] == ["verdict", "cited_rules", "obligations", "explanation"]
    for schema in (ParseOutput, _ParsedReply):  # already in order: unchanged
        assert grammar_schema(schema.model_json_schema()) == schema.model_json_schema()


def test_unreachable_server_is_a_clear_error(tmp_path):
    with pytest.raises(LLMError, match="not reachable"):
        LocalClient(base_url="http://127.0.0.1:9/v1", cache_dir=tmp_path, timeout=2)


def test_router_mode_needs_a_model_and_names_it(server, tmp_path, monkeypatch):
    monkeypatch.delenv("NTS_LOCAL_MODEL", raising=False)
    server.models = ["qwen3.5-2b-v2.Q5_K_M", "qwen3.5-4b-nts.Q8_0"]
    with pytest.raises(LLMError, match="router mode"):
        LocalClient(base_url=server.url, cache_dir=tmp_path)
    with pytest.raises(LLMError, match="not served"):
        LocalClient(base_url=server.url, cache_dir=tmp_path, server_model="qwen3.5-9b")
    monkeypatch.setenv("NTS_LOCAL_MODEL", "qwen3.5-4b-nts.Q8_0.gguf")  # the file name works too
    client = LocalClient(base_url=server.url, cache_dir=tmp_path)
    assert client.model == "local:qwen3.5-4b-nts.Q8_0.gguf"  # as with one model
    server.reply = '{"verdict": "allowed", "explanation": "x"}'
    client.generate("s", "p", PolicyOutput)
    assert server.requests[-1]["model"] == "qwen3.5-4b-nts.Q8_0"
