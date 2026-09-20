"""Low-level connection to the shim: connect, handshake, receive frames, send commands.

Message flow per match (the shim repeats it for every game when bwapi.ini auto_restart is on):

    shim -> Hello
    bot  -> ClientConfig
    shim -> GameStart
    loop: shim -> Frame ; bot -> Commands
    shim -> GameEnd
"""
from __future__ import annotations

import logging
import socket
import time
from typing import Optional

from .generated import bw
from .observation import GameInfo, Observation
from .protocol import (
    PROTOCOL_VERSION,
    ClientConfig,
    Disconnected,
    Hello,
    ProtocolError,
    envelope_as,
    parse_envelope,
    recv_message,
    send_message,
)

log = logging.getLogger("bwbot.client")


class ShimClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.hello: Optional[Hello] = None
        self.game: Optional[GameInfo] = None
        self._buf: Optional[bytearray] = None

    # ------------------------------------------------------------------ connection
    def connect(self, timeout: Optional[float] = None, retry_interval: float = 1.0) -> None:
        """Connect to the shim, retrying until `timeout` seconds elapse (None = forever)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        attempt = 0
        while True:
            try:
                s = socket.create_connection((self.host, self.port), timeout=5.0)
                s.settimeout(None)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock = s
                log.info("connected to shim at %s:%d", self.host, self.port)
                return
            except OSError as e:
                attempt += 1
                if deadline is not None and time.monotonic() >= deadline:
                    raise ConnectionError(f"could not reach shim at {self.host}:{self.port}: {e}") from e
                if attempt == 1 or attempt % 10 == 0:
                    log.info("waiting for shim at %s:%d (%s)", self.host, self.port, e)
                time.sleep(retry_interval)

    def close(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            finally:
                self.sock = None

    @property
    def connected(self) -> bool:
        return self.sock is not None

    # ------------------------------------------------------------------ messages
    def _recv(self) -> bw.Envelope:
        assert self.sock is not None
        self._buf = recv_message(self.sock)
        return parse_envelope(self._buf)

    def _send(self, payload: bytes | bytearray) -> None:
        assert self.sock is not None
        send_message(self.sock, payload)

    def wait_for_game(self, config: ClientConfig) -> GameInfo:
        """Block until a match starts: Hello -> ClientConfig -> GameStart."""
        if self._pending_hello:
            # A Hello already arrived (see next_frame); answer it now.
            self._pending_hello = False
            self._send(config.to_bytes())
        while True:
            env = self._recv()
            kind = env.MsgType()
            if kind == bw.Message.Hello:
                self.hello = Hello.from_fb(envelope_as(env, bw.Hello))
                if self.hello.protocol_version != PROTOCOL_VERSION:
                    raise ProtocolError(
                        f"protocol mismatch: shim={self.hello.protocol_version} bot={PROTOCOL_VERSION}")
                log.info("shim %s (%s) rev %d", self.hello.shim_version, self.hello.backend, self.hello.revision)
                self._send(config.to_bytes())
            elif kind == bw.Message.GameStart:
                self.game = GameInfo.from_fb(envelope_as(env, bw.GameStart))
                log.info("game start: %s (%dx%d) as %s vs %s", self.game.map_name, self.game.map_width,
                         self.game.map_height, self.game.self_player.name,
                         ", ".join(p.name for p in self.game.enemies) or "nobody")
                return self.game
            elif kind == bw.Message.GameEnd:
                log.debug("stale GameEnd before GameStart; ignoring")
            else:
                log.warning("unexpected %s while waiting for game", bw.Message.__dict__.get(kind, kind))

    def next_frame(self) -> Optional[Observation]:
        """Receive the next Frame. Returns None on GameEnd (check `.last_result`)."""
        assert self.game is not None
        while True:
            env = self._recv()
            kind = env.MsgType()
            if kind == bw.Message.Frame:
                return Observation(self.game, envelope_as(env, bw.Frame))
            if kind == bw.Message.GameEnd:
                ge = envelope_as(env, bw.GameEnd)
                self.last_result = bool(ge.IsWinner())
                self.last_end_frame = ge.FrameCount()
                return None
            if kind == bw.Message.Hello:
                # Shim restarted a match without us seeing GameEnd; treat as an aborted game.
                log.warning("Hello received mid-game; previous match aborted")
                self.hello = Hello.from_fb(envelope_as(env, bw.Hello))
                self._pending_hello = True
                self.last_result = False
                return None
            log.warning("unexpected message kind %s in game", kind)

    def send_commands(self, payload: bytes | bytearray) -> None:
        self._send(payload)

    def send_config(self, config: ClientConfig) -> None:
        """Change frame_skip/speed/etc. mid-game (applied by the shim before the next Frame)."""
        self._send(config.to_bytes())

    last_result: bool = False
    last_end_frame: int = 0
    _pending_hello: bool = False


__all__ = ["ShimClient", "ClientConfig", "Disconnected", "ProtocolError"]
