"""Idle-backoff behaviour of the hosted-room supervisor thread.

A multiplexed gateway serves every profile with its own supervisor; when almost
none of them own a room, a fixed-interval poll floods the process-wide SQLite
read pool (the 2026-11-25 ``session.create`` stall). These tests pin the
backoff contract: an empty pass waits longer each time (capped), any wakeup
resets it, and a room that appears is still picked up promptly.
"""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path

from tui_gateway.hosted_room_driver import (
    HostedRoomBinding,
    HostedRoomRuntime,
)

POLL = 0.05
BINDING = HostedRoomBinding(room_id="room-1", gateway_id="gateway-a", authority_epoch=1)


class CountingRooms:
    """A rooms provider that records when it is asked and can grow a room."""

    def __init__(self) -> None:
        self.calls: list[float] = []
        self._rooms: list[HostedRoomBinding] = []
        self._started = time.monotonic()
        self._lock = threading.Lock()

    def __call__(self) -> list[HostedRoomBinding]:
        with self._lock:
            self.calls.append(time.monotonic() - self._started)
            return list(self._rooms)

    def add_room(self) -> None:
        with self._lock:
            self._rooms.append(BINDING)


def make_runtime(provider: CountingRooms, tmp_path: Path) -> HostedRoomRuntime:
    return HostedRoomRuntime(  # type: ignore[arg-type]
        db_path=str(tmp_path / "rooms.db"),
        rooms=provider,
        rpc=object(),
        turn_lock=contextlib.nullcontext,
        poll_interval_seconds=POLL,
    )


def test_idle_passes_back_off_and_cap(tmp_path: Path) -> None:
    rt = make_runtime(CountingRooms(), tmp_path)
    rt._idle_cycles = 0
    schedule = []
    for _ in range(7):
        rt._idle_cycles += 1
        schedule.append(rt._idle_wait_seconds())
    assert schedule[0] == POLL * 2
    assert schedule[1] == POLL * 4
    assert schedule[2] == POLL * 8
    assert schedule[3] == POLL * 16
    assert all(w == POLL * 16 for w in schedule[4:])  # _IDLE_WAIT_DOUBLINGS caps it
    # with the shipped 5 s poll the cap is _IDLE_WAIT_MAX_SECONDS (60 s)
    rt2 = make_runtime(CountingRooms(), tmp_path)
    rt2._idle_cycles = 99
    saved = rt2.poll_interval_seconds
    rt2.poll_interval_seconds = 5.0
    try:
        assert rt2._idle_wait_seconds() == 60.0
    finally:
        rt2.poll_interval_seconds = saved


def test_wakeup_resets_backoff(tmp_path: Path) -> None:
    rt = make_runtime(CountingRooms(), tmp_path)
    rt._idle_cycles = 12
    assert rt._idle_wait_seconds() == POLL * 16
    rt.wakeup()
    assert rt._idle_cycles == 0
    assert rt._idle_wait_seconds() == POLL


def test_running_idle_loop_polls_far_less_than_fixed_interval(tmp_path: Path) -> None:
    provider = CountingRooms()
    rt = make_runtime(provider, tmp_path)
    rt.start()
    try:
        time.sleep(2.5)
    finally:
        rt.stop(timeout=5)
    # unpatched: ~2.5/POLL = 50 passes; with backoff: a handful
    assert len(provider.calls) <= 8


def test_room_appearing_after_backoff_is_found_quickly(tmp_path: Path) -> None:
    provider = CountingRooms()
    rt = make_runtime(provider, tmp_path)
    rt.start()
    time.sleep(0.5)  # let it back off a few steps
    provider.add_room()
    deadline = time.monotonic() + 3
    try:
        while time.monotonic() < deadline and len(provider.calls) < 2:
            time.sleep(0.02)
    finally:
        rt.stop(timeout=5)
    assert len(provider.calls) >= 2  # the room was seen on a later pass
