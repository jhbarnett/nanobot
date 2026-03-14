import asyncio

import pytest

from nanobot.heartbeat.service import HeartbeatService
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class DummyProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]):
        super().__init__()
        self._responses = list(responses)
        self.calls = 0

    async def chat(self, *args, **kwargs) -> LLMResponse:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return LLMResponse(content="", tool_calls=[])

    def get_default_model(self) -> str:
        return "test-model"


@pytest.mark.asyncio
async def test_start_is_idempotent(tmp_path) -> None:
    provider = DummyProvider([])

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        interval_s=9999,
        enabled=True,
    )

    await service.start()
    first_task = service._task
    await service.start()

    assert service._task is first_task

    service.stop()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_decide_returns_skip_when_no_tool_call(tmp_path) -> None:
    provider = DummyProvider([LLMResponse(content="no tool call", tool_calls=[])])
    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
    )

    action, tasks, notify = await service._decide("heartbeat content")
    assert action == "skip"
    assert tasks == ""
    assert notify is True


@pytest.mark.asyncio
async def test_trigger_now_executes_when_decision_is_run(tmp_path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    provider = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "run", "tasks": "check open tasks"},
                )
            ],
        )
    ])

    called_with: list[str] = []

    async def _on_execute(tasks: str) -> str:
        called_with.append(tasks)
        return "done"

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
    )

    result = await service.trigger_now()
    assert result == "done"
    assert called_with == ["check open tasks"]


@pytest.mark.asyncio
async def test_trigger_now_returns_none_when_decision_is_skip(tmp_path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    provider = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "skip"},
                )
            ],
        )
    ])

    async def _on_execute(tasks: str) -> str:
        return tasks

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
    )

    assert await service.trigger_now() is None


@pytest.mark.asyncio
async def test_decide_retries_transient_error_then_succeeds(tmp_path, monkeypatch) -> None:
    provider = DummyProvider([
        LLMResponse(content="429 rate limit", finish_reason="error"),
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "run", "tasks": "check open tasks"},
                )
            ],
        ),
    ])

    delays: list[int] = []

    async def _fake_sleep(delay: int) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
    )

    action, tasks, notify = await service._decide("heartbeat content")

    assert action == "run"
    assert tasks == "check open tasks"
    assert notify is True
    assert provider.calls == 2
    assert delays == [1]


@pytest.mark.asyncio
async def test_tick_suppresses_notify_when_config_disabled(tmp_path) -> None:
    """Config-level notify=False should suppress message delivery."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] do thing", encoding="utf-8")

    provider = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "run", "tasks": "check tasks", "notify": True},
                )
            ],
        )
    ])

    executed = []
    notified = []

    async def _on_execute(tasks: str) -> str:
        executed.append(tasks)
        return "done"

    async def _on_notify(response: str) -> None:
        notified.append(response)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
        on_notify=_on_notify,
        notify=False,
    )

    await service._tick()
    assert executed == ["check tasks"]
    assert notified == []  # notify suppressed by config


@pytest.mark.asyncio
async def test_tick_suppresses_notify_when_llm_says_no(tmp_path) -> None:
    """LLM returning notify=false (from HEARTBEAT.md instructions) should suppress."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] silent task\nnotify: false", encoding="utf-8")

    provider = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "run", "tasks": "silent task", "notify": False},
                )
            ],
        )
    ])

    executed = []
    notified = []

    async def _on_execute(tasks: str) -> str:
        executed.append(tasks)
        return "done"

    async def _on_notify(response: str) -> None:
        notified.append(response)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
        on_notify=_on_notify,
        notify=True,  # config allows, but LLM says no
    )

    await service._tick()
    assert executed == ["silent task"]
    assert notified == []  # notify suppressed by LLM decision


@pytest.mark.asyncio
async def test_tick_sends_notify_when_both_allow(tmp_path) -> None:
    """Notification should be sent when both config and LLM allow it."""
    (tmp_path / "HEARTBEAT.md").write_text("- [ ] loud task", encoding="utf-8")

    provider = DummyProvider([
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="hb_1",
                    name="heartbeat",
                    arguments={"action": "run", "tasks": "loud task", "notify": True},
                )
            ],
        )
    ])

    executed = []
    notified = []

    async def _on_execute(tasks: str) -> str:
        executed.append(tasks)
        return "done"

    async def _on_notify(response: str) -> None:
        notified.append(response)

    service = HeartbeatService(
        workspace=tmp_path,
        provider=provider,
        model="openai/gpt-4o-mini",
        on_execute=_on_execute,
        on_notify=_on_notify,
        notify=True,
    )

    await service._tick()
    assert executed == ["loud task"]
    assert notified == ["done"]


@pytest.mark.asyncio
async def test_should_notify_logic(tmp_path) -> None:
    """Unit test for _should_notify combining config and LLM signals."""
    provider = DummyProvider([])
    service = HeartbeatService(
        workspace=tmp_path, provider=provider, model="test", notify=True,
    )
    assert service._should_notify(True) is True
    assert service._should_notify(False) is False

    service.notify = False
    assert service._should_notify(True) is False
    assert service._should_notify(False) is False
