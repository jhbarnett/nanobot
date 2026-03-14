"""Heartbeat service - periodic agent wake-up to check for tasks."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider

_HEARTBEAT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "heartbeat",
            "description": "Report heartbeat decision after reviewing tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["skip", "run"],
                        "description": "skip = nothing to do, run = has active tasks",
                    },
                    "tasks": {
                        "type": "string",
                        "description": "Natural-language summary of active tasks (required for run)",
                    },
                    "notify": {
                        "type": "boolean",
                        "description": (
                            "Whether to send the result as a message to the user's channel. "
                            "Default true. Set false if HEARTBEAT.md instructs silent/no-notify mode."
                        ),
                    },
                },
                "required": ["action"],
            },
        },
    }
]


class HeartbeatService:
    """
    Periodic heartbeat service that wakes the agent to check for tasks.

    Phase 1 (decision): reads HEARTBEAT.md and asks the LLM — via a virtual
    tool call — whether there are active tasks.  This avoids free-text parsing
    and the unreliable HEARTBEAT_OK token.

    Phase 2 (execution): only triggered when Phase 1 returns ``run``.  The
    ``on_execute`` callback runs the task through the full agent loop and
    returns the result to deliver.
    """

    def __init__(
        self,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        on_execute: Callable[[str], Coroutine[Any, Any, str]] | None = None,
        on_notify: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        interval_s: int = 30 * 60,
        enabled: bool = True,
        notify: bool = True,
    ):
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.on_execute = on_execute
        self.on_notify = on_notify
        self.interval_s = interval_s
        self.enabled = enabled
        self.notify = notify
        self._running = False
        self._task: asyncio.Task | None = None

    @property
    def heartbeat_file(self) -> Path:
        return self.workspace / "HEARTBEAT.md"

    def _read_heartbeat_file(self) -> str | None:
        if self.heartbeat_file.exists():
            try:
                return self.heartbeat_file.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    async def _decide(self, content: str) -> tuple[str, str, bool]:
        """Phase 1: ask LLM to decide skip/run via virtual tool call.

        Returns (action, tasks, notify) where action is 'skip' or 'run' and
        notify indicates whether the result should be sent to the user's channel.
        The LLM may set notify=false when HEARTBEAT.md contains silent/no-notify
        instructions; the config-level ``self.notify`` acts as a master override.
        """
        response = await self.provider.chat_with_retry(
            messages=[
                {"role": "system", "content": (
                    "You are a heartbeat agent. Call the heartbeat tool to report your decision. "
                    "If HEARTBEAT.md contains instructions about messaging or notification "
                    "(e.g. 'silent', 'no-notify', 'do not send messages'), respect them by "
                    "setting the notify parameter to false."
                )},
                {"role": "user", "content": (
                    "Review the following HEARTBEAT.md and decide whether there are active tasks.\n\n"
                    f"{content}"
                )},
            ],
            tools=_HEARTBEAT_TOOL,
            model=self.model,
        )

        if not response.has_tool_calls:
            return "skip", "", True

        args = response.tool_calls[0].arguments
        return args.get("action", "skip"), args.get("tasks", ""), args.get("notify", True)

    async def start(self) -> None:
        """Start the heartbeat service."""
        if not self.enabled:
            logger.info("Heartbeat disabled")
            return
        if self._running:
            logger.warning("Heartbeat already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Heartbeat started (every {}s)", self.interval_s)

    def stop(self) -> None:
        """Stop the heartbeat service."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _run_loop(self) -> None:
        """Main heartbeat loop."""
        while self._running:
            try:
                await asyncio.sleep(self.interval_s)
                if self._running:
                    await self._tick()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Heartbeat error: {}", e)

    def _should_notify(self, llm_notify: bool) -> bool:
        """Return True when the heartbeat result should be sent to the user.

        The config-level ``self.notify`` is a master switch.  When it is True
        the LLM's per-tick decision (derived from HEARTBEAT.md instructions)
        is also honoured.  When the config disables notifications the LLM
        decision is ignored.
        """
        return self.notify and llm_notify

    async def _tick(self) -> None:
        """Execute a single heartbeat tick."""
        content = self._read_heartbeat_file()
        if not content:
            logger.debug("Heartbeat: HEARTBEAT.md missing or empty")
            return

        logger.info("Heartbeat: checking for tasks...")

        try:
            action, tasks, llm_notify = await self._decide(content)

            if action != "run":
                logger.info("Heartbeat: OK (nothing to report)")
                return

            logger.info("Heartbeat: tasks found, executing...")
            if self.on_execute:
                response = await self.on_execute(tasks)
                if response and self.on_notify and self._should_notify(llm_notify):
                    logger.info("Heartbeat: completed, delivering response")
                    await self.on_notify(response)
                elif response:
                    logger.info("Heartbeat: completed (notify suppressed)")
        except Exception:
            logger.exception("Heartbeat execution failed")

    async def trigger_now(self) -> str | None:
        """Manually trigger a heartbeat.

        Returns the execution result.  Notification delivery (if enabled) is
        left to the caller — ``trigger_now`` only runs the task.
        """
        content = self._read_heartbeat_file()
        if not content:
            return None
        action, tasks, _llm_notify = await self._decide(content)
        if action != "run" or not self.on_execute:
            return None
        return await self.on_execute(tasks)
