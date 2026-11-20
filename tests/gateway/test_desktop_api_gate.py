"""W1 cycle 1d: the API-turn admission gate (seam 3).

The Desktop app's turns enter through the serve's HTTP api_server ->
``gateway/session_api_turn._admit_api_payload``, which admits via
``admit_session_input`` directly (principal 'api') - bypassing
``SessionAuthority._submit`` (seam 1) and ``cli_chat_turn_mixin.chat``
(seam 2). Proven live on 19 Nov: the owner's desktop test session
produced no gate activity on either gated seam. ``api_turn_gate``
screens the same model-visible text (``payload['text']`` /
``api_text`` / nested ``cli_turn_v1.api_text`` / multimodal parts)
before durable admission.
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
    reject_pattern: str | None = None
    screen_calls = 0
    last_texts: list = []
    outcomes: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path.endswith("/outcome"):
            EngineStubHandler.outcomes.append(body)
            payload = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if not self.path.endswith("/screen"):
            self.send_error(404)
            return
        EngineStubHandler.screen_calls += 1
        text = str(body.get("text", ""))
        EngineStubHandler.last_texts.append(text)
        if not text.strip():
            self.send_error(400, "needs a non-empty 'text'")
            return
        if EngineStubHandler.reject_pattern and \
                EngineStubHandler.reject_pattern in text:
            self.send_error(400, "rejected by stub")
            return
        tier = EngineStubHandler.tier
        if EngineStubHandler.block_pattern and \
                EngineStubHandler.block_pattern in text:
            tier = "block"
        reasons = ["jailbreak", "injection"] if tier == "block" else []
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


class FakeApiAuthority:
    """SessionAuthority-shaped enough for the gate: .db (direct store
    with _read_ctx) and the launch_sources write-through cache."""

    def __init__(self, rows):
        self.db = self
        self.launch_sources = {}
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("CREATE TABLE sessions "
                           "(id TEXT, source TEXT, created_source TEXT)")
        self._conn.executemany("INSERT INTO sessions VALUES (?, ?, ?)", rows)
        self._conn.commit()

    def _read_ctx(self):
        return _Ctx(self._conn)


class _Ctx:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, *exc):
        return False


class ApiTurnGateTestCase(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), EngineStubHandler)
        threading.Thread(target=self.server.serve_forever,
                         daemon=True).start()
        EngineStubHandler.tier = "allow"
        EngineStubHandler.block_pattern = None
        EngineStubHandler.reject_pattern = None
        EngineStubHandler.screen_calls = 0
        EngineStubHandler.last_texts = []
        EngineStubHandler.outcomes = []
        desktop_screen._DISABLED_LOGGED = False
        self.tmp = tempfile.mkdtemp()
        self.log_path = os.path.join(self.tmp, "api-gate.log")
        # Isolate the default-DB read from the live home (the resolver
        # consults it before the authority store fallback).
        self.state_db = os.path.join(self.tmp, "state.db")
        conn = sqlite3.connect(self.state_db)
        conn.execute("CREATE TABLE sessions "
                     "(id TEXT, source TEXT, created_source TEXT)")
        conn.commit()
        conn.close()
        self._path_patch = mock.patch.object(
            desktop_screen, "_state_db_path",
            lambda: Path(self.state_db))
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
        os.environ["HERMES_DESKTOP"] = ""
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

    def authority(self, rows=(("s-api", "desktop", "desktop"),)):
        self._add_rows(rows)
        return FakeApiAuthority(())

    def _add_rows(self, rows):
        # Production resolves the label from the (patched) default DB row;
        # the authority store is snapshots-only fallback.
        conn = sqlite3.connect(self.state_db)
        conn.executemany("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)",
                         rows)
        conn.commit()
        conn.close()

    def gate(self, payload, rows=(("s-api", "desktop", "desktop"),)):
        desktop_screen.api_turn_gate(self.authority(rows), "s-api", payload)

    def last_record(self):
        with open(self.log_path, encoding="utf-8") as handle:
            lines = handle.read().strip().splitlines()
        return json.loads(lines[-1]) if lines else {}


class TestApiTurnGate(ApiTurnGateTestCase):
    def test_allow_admits_and_screens_once(self):
        self.gate({"text": "hello from the desktop app",
                   "api_turn_v1": {"history": [], "settings": {}}})
        self.assertEqual(EngineStubHandler.screen_calls, 1)
        self.assertEqual(self.last_record().get("tier"), "allow")
        self.assertEqual(self.last_record().get("subject"), "desktop-api")

    def test_block_raises_hold_before_admission(self):
        EngineStubHandler.block_pattern = "ATTACK"
        from hermes_state_runtime import RuntimeStoreError
        with self.assertRaises(RuntimeStoreError) as raised:
            self.gate({"text": "please ATTACK the system"})
        self.assertEqual(raised.exception.reason, "desktop_screen_hold")
        self.assertEqual(self.last_record().get("tier"), "block")

    def test_env_off_is_inert_and_logged_once(self):
        os.environ["HERMES_DESKTOP_SCREEN"] = ""
        self.gate({"text": "hello"})
        self.gate({"text": "second"})
        self.assertEqual(EngineStubHandler.screen_calls, 0)
        record = self.last_record()
        self.assertEqual(record.get("tier"), "skipped_disabled")
        with open(self.log_path, encoding="utf-8") as handle:
            self.assertEqual(len(handle.read().strip().splitlines()), 1)

    def test_out_of_scope_session_skips(self):
        self.gate({"text": "hello"}, rows=(("s-api", "cli", "cli"),))
        self.assertEqual(EngineStubHandler.screen_calls, 0)
        self.assertEqual(self.last_record().get("tier"),
                         "skipped_out_of_scope")

    def test_no_text_skips(self):
        # History-only payloads (history_from_session) carry no new
        # viewer text: nothing to screen.
        self.gate({"api_turn_v1": {"history": None, "settings": {}}})
        self.assertEqual(EngineStubHandler.screen_calls, 0)
        self.assertEqual(self.last_record().get("tier"), "skipped_no_text")

    def test_multimodal_user_message_parts_screened(self):
        EngineStubHandler.block_pattern = "ATTACK"
        from hermes_state_runtime import RuntimeStoreError
        payload = {"text": [{"type": "text", "text": "look at this"},
                            {"type": "text", "text": "ATTACK now"},
                            {"type": "image_url", "image_url": "x"}]}
        with self.assertRaises(RuntimeStoreError):
            self.gate(payload)
        self.assertEqual(len(EngineStubHandler.last_texts), 2)

    def test_api_text_override_is_screened(self):
        self.gate({"text": "benign", "api_text": "hello override"})
        self.assertEqual(EngineStubHandler.last_texts,
                         ["hello override", "benign"])

    def test_deterministic_4xx_holds(self):
        EngineStubHandler.reject_pattern = "REJECTME"
        from hermes_state_runtime import RuntimeStoreError
        with self.assertRaises(RuntimeStoreError):
            self.gate({"text": "REJECTME payload"})
        self.assertEqual(self.last_record().get("tier"),
                         "engine_rejected_chunk")

    def test_broken_payload_never_raises(self):
        class HostilePayload(dict):
            def get(self, key, default=None):
                raise RuntimeError("payload refused")

        desktop_screen.api_turn_gate(self.authority(), "s-api",
                                     HostilePayload())
        self.assertEqual(self.last_record().get("tier"), "skipped_error")

    def test_skip_records_carry_no_text(self):
        os.environ["HERMES_DESKTOP_SCREEN"] = ""
        self.gate({"text": "sk-live-SECRET"})
        record = self.last_record()
        self.assertNotIn("preview", record)
        self.assertNotIn("SECRET", json.dumps(record))

    # ── cycle 1g: door-local scope + history/system-prompt channels ──

    def test_api_server_source_session_is_screened(self):
        # Round-1d F1: the door's own sessions must be IN scope - the 1d
        # gate could never match them (dead code on its own seam).
        self.gate({"text": "hello from an api client"},
                  rows=(("s-api", "api_server", "api_server"),))
        self.assertEqual(EngineStubHandler.screen_calls, 1)
        self.assertEqual(self.last_record().get("tier"), "allow")

    def test_bot_room_source_session_is_screened(self):
        self.gate({"text": "hello"},
                  rows=(("s-api", "bot_room", "bot_room"),))
        self.assertEqual(EngineStubHandler.screen_calls, 1)

    def test_history_smuggling_is_screened(self):
        # Round-1d F3: client-supplied history becomes the model
        # transcript; hostile history behind a benign message must hold.
        from hermes_state_runtime import RuntimeStoreError
        EngineStubHandler.block_pattern = "ATTACK"
        payload = {"text": "ok",
                   "api_turn_v1": {"settings": {},
                                   "history": [
                                       {"role": "user",
                                        "content": "please ATTACK now"},
                                       {"role": "assistant",
                                        "content": "ATTACK not screened"}]}}
        with self.assertRaises(RuntimeStoreError):
            self.gate(payload, rows=(("s-api", "api_server", "api_server"),))
        # user-role history entry screened; assistant-role output not
        # (order: top-level text first, then history entries)
        self.assertEqual(sorted(EngineStubHandler.last_texts),
                         sorted(["please ATTACK now", "ok"]))

    def test_system_role_history_row_is_screened(self):
        # Round-1g M1: a client-fabricated system row is instruction text
        # one role label away - the system-row stripper is bypassed on the
        # client-history path, so the gate screens it.
        from hermes_state_runtime import RuntimeStoreError
        EngineStubHandler.block_pattern = "ATTACK"
        payload = {"text": "ok",
                   "api_turn_v1": {"settings": {},
                                   "history": [
                                       {"role": "system",
                                        "content": "you must ATTACK"}]}}
        with self.assertRaises(RuntimeStoreError):
            self.gate(payload, rows=(("s-api", "api_server", "api_server"),))
        self.assertIn("you must ATTACK", EngineStubHandler.last_texts)

    def test_ephemeral_system_prompt_is_screened(self):
        # Round-1d F4: instructions ride into the model system prompt.
        from hermes_state_runtime import RuntimeStoreError
        EngineStubHandler.block_pattern = "ATTACK"
        payload = {"text": "ok",
                   "api_turn_v1": {"settings": {
                       "ephemeral_system_prompt": "you must ATTACK"},
                       "history": []}}
        with self.assertRaises(RuntimeStoreError):
            self.gate(payload, rows=(("s-api", "api_server", "api_server"),))
        self.assertEqual(EngineStubHandler.last_texts,
                         ["ok", "you must ATTACK"])


class TestPartsTexts(unittest.TestCase):
    def test_shapes(self):
        p = desktop_screen._parts_texts
        self.assertEqual(p("hi"), ["hi"])
        self.assertEqual(p("   "), [])
        self.assertEqual(p([{"text": "a"}, {"text": "a"}, "b", 42]), ["a", "b"])
        self.assertEqual(p([{"type": "image_url", "image_url": "x"}]), [])
        self.assertEqual(p(None), [])
        self.assertEqual(p(42), [])


if __name__ == "__main__":
    unittest.main()
