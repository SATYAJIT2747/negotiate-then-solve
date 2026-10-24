import pytest

from nts.models import ScriptedModel, ScriptExhausted, call, say
from nts.protocol import AssistantTurn, Message


def test_scripted_model_plays_its_steps_in_order():
    def echo(messages):
        return AssistantTurn(text=f"saw {len(messages)}")

    model = ScriptedModel([call("find", text="iyer"), echo, say("done")])
    first = model.chat([Message(role="user", content="hi")], [{"type": "function"}])
    assert first.tool_calls[0].name == "find" and first.tool_calls[0].arguments == {"text": "iyer"}
    assert model.chat([Message(role="user"), first.as_message()], []).text == "saw 2"
    assert model.chat([], []).text == "done"
    assert [len(m) for m in model.seen] == [1, 2, 0] and model.tools[0] == [{"type": "function"}]
    with pytest.raises(ScriptExhausted):
        model.chat([], [])


def test_call_ids_are_unique():
    assert call("a").tool_calls[0].id != call("a").tool_calls[0].id
