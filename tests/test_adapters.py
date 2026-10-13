"""LoRA adapters per task (language.adapters): the registry, and the local client switching adapters on
a fake server (no GPU needed)."""

import pytest
from test_local import FakeServer

from agents.policy import PolicyOutput
from language.adapters import Registry, TaskClients, local_client, task_clients
from language.llm import LLMError

REGISTRY = """
base = "Qwen3.5-4B-Base-Q8_0"
[adapters]
nts-intents = ["compile"]
nts-policy = ["policy"]
[domains.iitx]
policy = "iitx-policy"
"""


@pytest.fixture
def reg(tmp_path):
    path = tmp_path / "adapters.toml"
    path.write_text(REGISTRY, encoding="utf-8")
    return Registry.load(path)


@pytest.fixture
def server():
    s = FakeServer()
    s.models = ["Qwen3.5-4B-Base-Q8_0"]
    s.loras = ["/models/nts-intents.gguf", "/models/nts-policy.f16.gguf", "/models/iitx-policy.gguf"]
    s.reply = '{"verdict":"allowed","explanation":"Fine."}'
    yield s
    s.httpd.shutdown()


def test_registry_gives_each_task_its_adapter_and_a_domain_its_own(reg, tmp_path):
    assert reg.adapter_for("compile") == "nts-intents" and reg.adapter_for("policy") == "nts-policy"
    assert reg.adapter_for("reply") is None  # not adapted: the served model itself
    assert reg.adapter_for("policy", "iitx") == "iitx-policy" and reg.adapter_for("compile", "iitx") == "nts-intents"
    with pytest.raises(ValueError):
        reg.adapter_for("policy", "nowhere")
    bad = tmp_path / "bad.toml"
    bad.write_text('[adapters]\na = ["policy"]\nb = ["policy"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="more than one adapter"):
        Registry.load(bad)


def test_each_request_switches_on_its_own_adapter(server, reg, tmp_path):
    policy = local_client("policy", base_url=server.url, reg=reg, cache_dir=tmp_path)
    policy.generate("system", "prompt", PolicyOutput)
    sent = server.requests[-1]
    assert sent["model"] == "Qwen3.5-4B-Base-Q8_0"
    assert sent["lora"] == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}, {"id": 2, "scale": 0.0}]
    assert policy.model.endswith("+nts-policy.gguf")
    domain = local_client("policy", base_url=server.url, reg=reg, domain="iitx", cache_dir=tmp_path)
    domain.generate("system", "prompt", PolicyOutput)  # same prompt, another adapter: not a cache hit
    assert len(server.requests) == 2 and server.requests[-1]["lora"][2]["scale"] == 1.0


def test_an_adapter_served_as_a_model_of_its_own_name(server, tmp_path):
    server.loras = None  # a router with a preset (base + adapter) or a merged GGUF per adapter
    server.models = ["Qwen3.5-4B-Base-Q8_0", "nts-policy"]
    client = local_client("policy", base_url=server.url, adapter="nts-policy", cache_dir=tmp_path)
    client.generate("system", "prompt", PolicyOutput)
    assert server.requests[-1]["model"] == "nts-policy" and "lora" not in server.requests[-1]
    with pytest.raises(LLMError, match="nts-missing"):
        local_client("policy", base_url=server.url, adapter="nts-missing", cache_dir=tmp_path)


def test_task_clients_share_one_client_per_adapter(server, reg, tmp_path):
    clients = task_clients(base_url=server.url, reg=reg, cache_dir=tmp_path)
    assert isinstance(clients, TaskClients)
    assert clients.get("policy") is clients.get("policy") and clients.get("policy") is not clients.get("compile")
    assert clients.get("reply") is clients.get("explain")  # neither adapted: the base itself
    server.models.append("another-model")  # a router: the registry's base is named, not guessed
    assert task_clients(base_url=server.url, reg=reg, cache_dir=tmp_path).get("reply").server_model == reg.base
    one = task_clients(base_url=server.url, server_model="Qwen3.5-4B-Base-Q8_0", cache_dir=tmp_path)
    assert one.get("policy") is one.get("compile")  # no registry: one model for every task, as before


@pytest.mark.slow
def test_the_web_app_gives_the_parser_and_the_policy_check_their_own_adapters(server, tmp_path, monkeypatch):
    from web.world import World, _task

    path = tmp_path / "adapters.toml"
    path.write_text(REGISTRY, encoding="utf-8")
    monkeypatch.setenv("NTS_LOCAL_URL", server.url)
    monkeypatch.setenv("NTS_ADAPTERS", str(path))
    w = World(parser_mode="local", seed_history=False, deadline=5)
    assert w.model_link.up is True and "nts-policy" in w.models["policy"]
    clients = w.model_link.client
    assert _task(clients, "policy").adapter == "nts-policy" and _task(clients, "compile").adapter == "nts-intents"
