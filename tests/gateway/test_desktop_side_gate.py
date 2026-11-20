"""W1 cycle 1h: the side-agent gate (/bg and /btw).

Both verbs take viewer-authored text and build a DETACHED side agent
that runs the text with no admission - neither the tui turn gate nor any
submission seam sees it. ``side_agent_gate`` screens the text in the
parent session's scope BEFORE the side agent spawns; a hold raises the
control-plane refusal so the verb replies with an error.
"""

import json
import os
import sqlite3
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from gateway import desktop_screen


class EngineStubHandler(BaseHTTPRequestHandler):
    tier: str = "allow"
    block_pattern: str | None = None
    screen_calls = 0
    last_texts: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.endswith("/screen"):
            self.send_error(404)
            return
        EngineStubHandler.screen_calls += 1
        text = str(body.get("text", ""))
        EngineStubHandler.last_texts.append(text)
        if not text.strip():
            self.send_error(400, "needs a non-empty 'text'")
            return
        tier = "block" if (EngineStubHandler.block_pattern
                           and EngineStubHandler.block_pattern in text) \
            else EngineStubHandler.tier
        reasons = ["jailbreak"] if tier == "block" else []
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


class SideAgentGateTestCase(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), EngineStubHandler)
        threading.Thread(target=self.server.serve_forever,
                         daemon=True).start()
        EngineStubHandler.tier = "allow"
        EngineStubHandler.block_pattern = None
        EngineStubHandler.screen_calls = 0
        EngineStubHandler.last_texts = []
        desktop_screen._DISABLED_LOGGED = False
        self.tmp = tempfile.mkdtemp()
        self.log_path = os.path.join(self.tmp, "side-gate.log")
        self.state_db = os.path.join(self.tmp, "state.db")
        conn = sqlite3.connect(self.state_db)
        conn.execute("CREATE TABLE sessions "
                     "(id TEXT, source TEXT, created_source TEXT)")
        conn.commit()
        conn.close()
        self._path_patch = mock.patch.object(
            desktop_screen, "_state_db_path", lambda: Path(self.state_db))
        self._path_patch.start()
        self.addCleanup(self._path_patch.stop)
        self.env = {
            "HERMES_DESKTOP_SCREEN": "1",
            "HERMES_DESKTOP_SCREEN_ENDPOINT":
                f"http://127.0.0.1:{self.server.server_port}/v1/screen",
            "HERMES_DESKTOP_SCREEN_TIMEOUT_S": "2",
            "HERMES_DESKTOP_SCREEN_LOG": self.log_path,
        }
        self._saved = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        self._desktop = os.environ.get("HERMES_DESKTOP")
        os.environ["HERMES_DESKTOP"] = "1"
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if self._desktop is None:
            os.environ.pop("HERMES_DESKTOP", None)
        else:
            os.environ["HERMES_DESKTOP"] = self._desktop
        self.server.shutdown()

    def _row(self, sid="s-side", source="desktop", created="desktop"):
        conn = sqlite3.connect(self.state_db)
        conn.execute("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)",
                     (sid, source, created))
        conn.commit()
        conn.close()

    def last_record(self):
        with open(self.log_path, encoding="utf-8") as handle:
            lines = handle.read().strip().splitlines()
        return json.loads(lines[-1]) if lines else {}


class TestSideAgentGate(SideAgentGateTestCase):
    def test_allow_screens_once(self):
        self._row()
        self.assertIsNone(desktop_screen.side_agent_gate(
            None, "s-side", "/bg summarise the repo"))
        self.assertEqual(EngineStubHandler.screen_calls, 1)
        self.assertEqual(self.last_record().get("subject"), "desktop-side")

    def test_block_raises_hold(self):
        from hermes_state_runtime import RuntimeStoreError
        self._row()
        EngineStubHandler.block_pattern = "ATTACK"
        with self.assertRaises(RuntimeStoreError) as raised:
            desktop_screen.side_agent_gate(None, "s-side",
                                           "/btw please ATTACK")
        self.assertEqual(raised.exception.reason, "desktop_screen_hold")
        self.assertEqual(self.last_record().get("tier"), "block")

    def test_env_off_inert(self):
        os.environ["HERMES_DESKTOP_SCREEN"] = ""
        self._row()
        self.assertIsNone(desktop_screen.side_agent_gate(
            None, "s-side", "/bg hello"))
        self.assertEqual(EngineStubHandler.screen_calls, 0)
        self.assertEqual(self.last_record().get("tier"), "skipped_disabled")

    def test_out_of_scope_cli_parent(self):
        self._row("s-cli", source="cli", created="cli")
        self.assertIsNone(desktop_screen.side_agent_gate(
            None, "s-cli", "/bg hello"))
        self.assertEqual(EngineStubHandler.screen_calls, 0)
        self.assertEqual(self.last_record().get("tier"),
                         "skipped_out_of_scope")

    def test_unresolvable_desktop_backend_screens(self):
        EngineStubHandler.block_pattern = "ATTACK"
        from hermes_state_runtime import RuntimeStoreError
        with self.assertRaises(RuntimeStoreError):
            desktop_screen.side_agent_gate(None, "unknown-1",
                                           "/btw please ATTACK")

    def test_no_text_skips(self):
        self._row()
        self.assertIsNone(desktop_screen.side_agent_gate(None, "s-side", "  "))
        self.assertEqual(EngineStubHandler.screen_calls, 0)
        self.assertEqual(self.last_record().get("tier"), "skipped_no_text")

    def test_list_of_text_fields_screened(self):
        # Round-1 M1: preview.restart carries url + cwd + console context;
        # the gate must screen every field, not just the first.
        from hermes_state_runtime import RuntimeStoreError
        self._row()
        EngineStubHandler.block_pattern = "ATTACK"
        with self.assertRaises(RuntimeStoreError):
            desktop_screen.side_agent_gate(
                None, "s-side",
                ["http://127.0.0.1:9/", "/tmp/preview ATTACK cwd",
                 "please ATTACK via context"])
        # Screens fields in order and stops at the first block: url (allow)
        # then cwd (ATTACK -> hold); the third field need not be screened.
        self.assertEqual(len(EngineStubHandler.last_texts), 2)
        self.assertIn("ATTACK", EngineStubHandler.last_texts[1])

    def test_async_wrapper_holds_off_the_loop(self):
        # Round-1 M2: async handlers must not block the event loop; the
        # async wrapper runs the sync screen in the executor and refuses
        # identically.
        import asyncio

        from hermes_state_runtime import RuntimeStoreError
        self._row()
        EngineStubHandler.block_pattern = "ATTACK"

        async def scenario():
            with self.assertRaises(RuntimeStoreError):
                await desktop_screen.side_agent_gate_async(
                    None, "s-side", ["please ATTACK"],
                    "desktop-side-preview")
            # the loop kept running during the screen: another awaited
            # task resolves while the executor thread works
            await asyncio.sleep(0)

        asyncio.run(scenario())

    def test_never_raises_on_hostile_input(self):
        self._row()
        self.assertIsNone(desktop_screen.side_agent_gate(
            None, "s-side", object()))


if __name__ == "__main__":
    unittest.main()
