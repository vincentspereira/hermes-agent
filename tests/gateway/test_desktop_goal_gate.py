"""W1 cycle 1i: the goal-family gates (/goal, /subgoal, /heartbeat).

Goal-directive, subgoal and heartbeat-prompt text is viewer text that
reaches the model as enqueued turns and PERSISTS across turns - none of
the other seams saw it. The gateway slash handlers screen the text
(loop-safe, via ``side_agent_gate_notice_async``) BEFORE anything
dispatches, enqueues or persists; a hold returns the notice as the
slash reply.
"""

import json
import os
import pathlib
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from gateway import desktop_screen


class EngineStubHandler(BaseHTTPRequestHandler):
    tier: str = "allow"
    block_pattern: str | None = None
    screen_calls = 0
    last_texts: list = []
    last_subjects: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.endswith("/screen"):
            self.send_error(404)
            return
        EngineStubHandler.screen_calls += 1
        text = str(body.get("text", ""))
        EngineStubHandler.last_texts.append(text)
        EngineStubHandler.last_subjects.append(str(body.get("caller", "")))
        if not text.strip():
            self.send_error(400, "needs a non-empty 'text'")
            return
        tier = "block" if (EngineStubHandler.block_pattern
                           and EngineStubHandler.block_pattern in text) \
            else EngineStubHandler.tier
        reasons = ["injection"] if tier == "block" else []
        payload = json.dumps({"tier": tier, "reasons": reasons,
                              "request_id": "req-stub-1",
                              "served_by": {"provider": "stub"},
                              "scores": {}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


class GoalFamilyHandlerTestCase(unittest.TestCase):
    """Drive the REAL handlers with a patched gate: a hold must come back
    as the slash reply and dispatch/persist must never run."""

    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), EngineStubHandler)
        threading.Thread(target=self.server.serve_forever,
                         daemon=True).start()
        EngineStubHandler.tier = "allow"
        EngineStubHandler.block_pattern = None
        EngineStubHandler.screen_calls = 0
        EngineStubHandler.last_texts = []
        desktop_screen._DISABLED_LOGGED = False
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_path = os.path.join(self.tmp.name, "goal-gate.log")
        self._saved_env = {}
        self._isolate_env()

    def _isolate_env(self):
        for key in ("HERMES_DESKTOP", "HERMES_DESKTOP_SCREEN",
                    "HERMES_DESKTOP_SCREEN_ENDPOINT",
                    "HERMES_DESKTOP_SCREEN_LOG"):
            self._saved_env[key] = os.environ.get(key)
        os.environ["HERMES_DESKTOP"] = "1"
        os.environ["HERMES_DESKTOP_SCREEN"] = "1"
        os.environ["HERMES_DESKTOP_SCREEN_ENDPOINT"] = \
            f"http://127.0.0.1:{self.server.server_port}/v1/screen"
        os.environ["HERMES_DESKTOP_SCREEN_LOG"] = self.log_path
        # Isolate the default-DB read: the live state.db is written by the
        # real gateway; the row resolver must not race it.
        import sqlite3
        state_db = pathlib.Path(self.tmp.name) / "state.db"
        conn = sqlite3.connect(state_db)
        conn.execute("CREATE TABLE sessions "
                     "(id TEXT, source TEXT, created_source TEXT)")
        conn.commit()
        conn.close()
        self._db_patch = mock.patch.object(
            desktop_screen, "_state_db_path", lambda: state_db)
        self._db_patch.start()
        self.addCleanup(self._db_patch.stop)
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _holding_gate(self):
        from gateway import desktop_screen

        def _raise(*args, **kwargs):
            raise desktop_screen.hold_error(["injection", "secrets"])

        return mock.patch.object(desktop_screen, "side_agent_gate", _raise)

    def _manager(self):
        mgr = mock.Mock()
        mgr.session_id = "s-goal"
        mgr.has_goal.return_value = True
        mgr.status_line.return_value = "goal: ship it"
        mgr.render_subgoals.return_value = ""
        return mgr

    def _async_run(self, coro):
        import asyncio
        return asyncio.run(coro)

    def _goal_reply(self, args, dispatch_return=None):
        """Drive the real _handle_goal_command with a patched manager."""
        from gateway.slash_commands_goals import GatewayGoalCommandsMixin
        from gateway.run_goals import GatewayGoalsMixin

        class SlashCommandsGoals(GatewayGoalCommandsMixin, GatewayGoalsMixin):
            pass
        mgr = self._manager()
        instance = SlashCommandsGoals.__new__(SlashCommandsGoals)
        event = mock.Mock()
        event.get_command_args.return_value = args
        event.source = None

        async def _executor_passthrough(_self, fn, *a, **k):
            return fn(*a, **k)

        async def _get_goal_manager_for_event(_event):
            return mgr, None
        async def _noop(*a, **k):
            return None
        with mock.patch.object(SlashCommandsGoals,
                               "_get_goal_manager_for_event",
                               staticmethod(_get_goal_manager_for_event)), \
                mock.patch.object(SlashCommandsGoals,
                                  "_clear_goal_continuations", _noop,
                                  create=True), \
                mock.patch.object(SlashCommandsGoals,
                                  "_run_in_executor_with_context",
                                  _executor_passthrough, create=True), \
                mock.patch("hermes_cli.goal_command.dispatch_goal_command") as \
                dispatch:
            if dispatch_return is not None:
                dispatch.return_value = dispatch_return
            reply = self._async_run(
                instance._handle_goal_command(event))
        return reply, dispatch

    def test_goal_hold_returns_notice_and_never_dispatches(self):
        EngineStubHandler.block_pattern = "ATTACK"
        with self._holding_gate():
            reply, dispatch = self._goal_reply("/goal ship it ATTACK now")
        self.assertIn("Held by the desktop intake screen", reply)
        self.assertNotIn("desktop_screen_hold", reply)
        dispatch.assert_not_called()

    def test_goal_benign_passes_with_one_screen(self):
        EngineStubHandler.screen_calls = 0
        from types import SimpleNamespace
        reply, _ = self._goal_reply(
            "/goal ship it",
            dispatch_return=SimpleNamespace(output="goal set", prompt=None,
                                            clear_pending=None))
        self.assertEqual(reply, "goal set")
        self.assertEqual(EngineStubHandler.screen_calls, 1)
        self.assertEqual(EngineStubHandler.last_texts, ["/goal ship it"])


if __name__ == "__main__":
    unittest.main()
