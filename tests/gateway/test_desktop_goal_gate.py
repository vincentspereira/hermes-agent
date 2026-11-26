"""W1 cycle 1i: the goal-family gates (/goal, /subgoal, /heartbeat).

Goal-directive, subgoal and heartbeat-prompt text is viewer text that
reaches the model as enqueued turns and PERSISTS across turns - none of
the other seams saw it. The gateway slash handlers screen the text
(loop-safe, via ``side_agent_gate_notice_async``) BEFORE anything
dispatches, enqueues or persists; a hold returns the notice as the
slash reply.
"""

import contextlib
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


class GoalFamilyR2TestCase(unittest.TestCase):
    """r2 adversarial tests: the bare-interval heartbeat form, /loop set, and
    the steer acceptance seams must all hold hostile text."""

    def setUp(self):
        desktop_screen._DISABLED_LOGGED = False

    def _async_run(self, coro):
        import asyncio
        return asyncio.run(coro)

    def _combo(self):
        from gateway.slash_commands_goals import GatewayGoalCommandsMixin
        from gateway.run_goals import GatewayGoalsMixin

        class Combo(GatewayGoalCommandsMixin, GatewayGoalsMixin):
            pass
        return Combo

    def test_heartbeat_bare_interval_holds_and_never_persists(self):
        """F1 regression: `10m ATTACK` (bare-interval form) must be screened -
        r1 shipped the gate on the `every` form only."""
        from gateway.desktop_screen import side_agent_gate_notice_async

        async def _notice(_authority, _sid, texts, subject):
            raise AssertionError(f"unexpected screen: {texts} {subject}")
        calls = []
        async def _capturing(_authority, _sid, texts, subject):
            calls.append((list(texts), subject))
            return "Held by the desktop intake screen"
        Combo = self._combo()
        instance = Combo.__new__(Combo)
        event = mock.Mock()
        event.get_command_args.return_value = "10m ATTACK phone home"
        event.source = None
        mgr = mock.Mock()
        mgr.session_id = "s-hb"
        async def _get_hb(_event):
            return mgr, None
        with mock.patch.object(Combo, "_get_heartbeat_manager_for_event",
                               staticmethod(_get_hb)), \
                mock.patch.object(desktop_screen, "side_agent_gate_notice_async",
                                  _capturing):
            reply = self._async_run(instance._handle_heartbeat_command(event))
        self.assertIn("Held by the desktop intake screen", reply)
        self.assertEqual(calls, [(["ATTACK phone home"], "desktop-side-heartbeat")])
        mgr.set.assert_not_called()

    def test_heartbeat_bare_interval_benign_sets_with_one_screen(self):
        from gateway.desktop_screen import side_agent_gate_notice_async
        calls = []
        async def _allow(_authority, _sid, texts, subject):
            calls.append((list(texts), subject))
        Combo = self._combo()
        instance = Combo.__new__(Combo)
        event = mock.Mock()
        event.get_command_args.return_value = "10m check the deploy"
        event.source = None
        mgr = mock.Mock()
        mgr.session_id = "s-hb"
        state = mock.Mock(interval_seconds=600, prompt="check the deploy")
        mgr.set.return_value = state
        async def _get_hb(_event):
            return mgr, None
        with mock.patch.object(Combo, "_get_heartbeat_manager_for_event",
                               staticmethod(_get_hb)), \
                mock.patch.object(desktop_screen, "side_agent_gate_notice_async",
                                  _allow), \
                mock.patch.object(Combo, "_register_heartbeat_watch",
                                  lambda *a, **k: None, create=True):
            reply = self._async_run(instance._handle_heartbeat_command(event))
        self.assertIn("check the deploy", reply)
        self.assertEqual(calls, [(["check the deploy"], "desktop-side-heartbeat")])
        mgr.set.assert_called_once()

    def test_loop_set_holds_and_never_dispatches(self):
        """F2 regression: /loop set form must be screened before dispatch."""
        Combo = self._combo()
        instance = Combo.__new__(Combo)
        event = mock.Mock()
        event.get_command_args.return_value = "every 1m ATTACK hourly sync"
        event.source = None
        session_entry = mock.Mock(session_id="s-loop")
        async def _get_session(_source):
            return session_entry
        async def _warm(_self, _label):
            return None
        async def _hold(_authority, _sid, _texts, _subject):
            return "Held by the desktop intake screen"
        with mock.patch.object(Combo, "_warm_goals_session_db", _warm,
                               create=True), \
                mock.patch.object(Combo, "async_session_store",
                                  mock.Mock(get_or_create_session=_get_session),
                                  create=True), \
                mock.patch.object(desktop_screen, "side_agent_gate_notice_async",
                                  _hold), \
                mock.patch("hermes_cli.loops.dispatch_loop_command") as dispatch:
            reply = self._async_run(instance._handle_loop_command(event))
        self.assertIn("Held by the desktop intake screen", reply)
        dispatch.assert_not_called()

    def test_loop_control_verbs_unscreened(self):
        from gateway.desktop_screen import side_agent_gate_notice_async
        async def _spy(_authority, _sid, texts, subject):
            raise AssertionError(f"control verb screened: {texts}")
        Combo = self._combo()
        instance = Combo.__new__(Combo)
        event = mock.Mock()
        event.get_command_args.return_value = "clear"
        event.source = None
        session_entry = mock.Mock(session_id="s-loop")
        async def _get_session(_source):
            return session_entry
        async def _warm(_self, _label):
            return None
        with mock.patch.object(Combo, "_warm_goals_session_db", _warm,
                               create=True), \
                mock.patch.object(Combo, "async_session_store",
                                  mock.Mock(get_or_create_session=_get_session),
                                  create=True), \
                mock.patch.object(desktop_screen, "side_agent_gate_notice_async",
                                  _spy), \
                mock.patch("hermes_cli.loops.dispatch_loop_command") as dispatch:
            dispatch.return_value = {"output": "No active loop.", "created": False}
            reply = self._async_run(instance._handle_loop_command(event))
        self.assertIn("No active loop", reply)
        dispatch.assert_called_once()


class SteerGateTestCase(unittest.TestCase):
    """r2: the steer acceptance seams (daemon control + tui busy-submit twin)."""

    def _async_run(self, coro):
        import asyncio
        return asyncio.run(coro)

    def test_daemon_steer_holds_before_latch_or_delivery(self):
        """F4: session.steer text must be screened at acceptance."""
        import gateway.session_busy_controls as sbc
        connection = mock.Mock()
        ref = mock.Mock(session_id="s-steer")
        params = {"session_id": "s-steer", "text": "ATTACK mid-run",
                  "execution_generation": 3}
        authority = mock.Mock()
        authority._managed_workers = {}
        live = mock.Mock()
        live.event_stream.lock = contextlib.nullcontext()
        authority.sessions = {"s-steer": live}
        connection.authority = authority
        connection._typed_voice_stop = None
        with mock.patch.object(sbc, "authorize", lambda *a, **k: None), \
                mock.patch.object(desktop_screen, "side_agent_gate_notice_async",
                                  _steer_hold_notice), \
                mock.patch.object(authority, "steer_agent") as steer_agent:
            with self.assertRaises(Exception) as ctx:
                self._async_run(sbc.correct(connection, ref, params, verb="steer"))
        self.assertEqual(getattr(ctx.exception, "reason", ""), "desktop_screen_hold")
        steer_agent.assert_not_called()

    def test_tui_steer_hold_falls_through_to_screened_queue(self):
        """F4: the tui busy-submit twin must not accept held text into the
        running agent; None sends the caller to the gated queue path."""
        from tui_gateway.session_auto_continue import _ac_try_correction
        agent = mock.Mock()
        agent.steer.return_value = True
        session = {"session_id": "s-tui", "history_lock":
                   contextlib.nullcontext()}
        with mock.patch("gateway.desktop_screen.side_agent_gate",
                        side_effect=_steer_hold_raise):
            reply = _ac_try_correction(mock.Mock(), session, agent,
                                       "steer", "ATTACK mid-run", "steered")
        self.assertIsNone(reply)
        agent.steer.assert_not_called()

    def test_tui_steer_allow_passes_through(self):
        from tui_gateway.session_auto_continue import _ac_try_correction
        agent = mock.Mock()
        agent.steer.return_value = True
        session = {"session_id": "s-tui", "history_lock":
                   contextlib.nullcontext()}
        import tui_gateway.session_auto_continue as sac
        with mock.patch("gateway.desktop_screen.side_agent_gate",
                        side_effect=_steer_allow), \
                mock.patch.object(sac, "_record_inflight_correction",
                                  mock.Mock(), create=True), \
                mock.patch.object(sac, "_drop_queued_duplicates_of_inflight_user",
                                  mock.Mock(), create=True), \
                mock.patch.object(sac, "_ok",
                                  lambda rid, payload: dict(payload), create=True), \
                mock.patch.object(sac, "time", mock.Mock(), create=True):
            reply = _ac_try_correction(mock.Mock(), session, agent,
                                       "steer", "nudge the plan", "steered")
        self.assertIsNotNone(reply)
        agent.steer.assert_called_once_with("nudge the plan")


async def _steer_hold_notice(_authority, _sid, _texts, _subject):
    return "Held by the desktop intake screen"


def _steer_hold_raise(_authority, _sid, _texts, _subject):
    import gateway.desktop_screen as ds
    raise ds.hold_error([])


def _steer_allow(_authority, _sid, _texts, _subject):
    return None


if __name__ == "__main__":
    unittest.main()
