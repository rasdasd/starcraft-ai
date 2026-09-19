"""Wire framing and message (de)serialization for the shim protocol.

Every message on the socket is a little-endian uint32 length followed by a FlatBuffer with
root type `bw.Envelope` (file identifier "BWB1"). See ../../proto/bw.fbs.
"""
from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Optional

import flatbuffers

from .generated import bw

PROTOCOL_VERSION = 1
_HDR = struct.Struct("<I")
MAX_MESSAGE = 256 << 20


class ProtocolError(RuntimeError):
    pass


class Disconnected(ConnectionError):
    pass


def recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise Disconnected("shim closed the connection")
        got += r
    return buf


def recv_message(sock: socket.socket) -> bytearray:
    """Read one length-prefixed payload."""
    (n,) = _HDR.unpack(recv_exact(sock, 4))
    if n > MAX_MESSAGE:
        raise ProtocolError(f"message too large: {n}")
    return recv_exact(sock, n)


def send_message(sock: socket.socket, payload: bytes | bytearray | memoryview) -> None:
    sock.sendall(_HDR.pack(len(payload)) + bytes(payload))


def parse_envelope(buf: bytearray) -> bw.Envelope:
    if not bw.Envelope.EnvelopeBufferHasIdentifier(buf, 0):
        raise ProtocolError("bad file identifier (not a BWB1 envelope)")
    return bw.Envelope.GetRootAs(buf, 0)


@dataclass
class ClientConfig:
    """Bot-side settings sent to the shim after Hello (see bw.fbs ClientConfig)."""

    frame_skip: int = 1
    include_bullets: bool = True
    include_tiles: bool = True
    include_players: bool = True
    local_speed: int = -1
    gui: bool = True
    complete_map_information: bool = False
    user_input: bool = True

    def to_bytes(self) -> bytearray:
        b = flatbuffers.Builder(64)
        bw.ClientConfigStart(b)
        bw.ClientConfigAddFrameSkip(b, int(self.frame_skip))
        bw.ClientConfigAddIncludeBullets(b, bool(self.include_bullets))
        bw.ClientConfigAddIncludeTiles(b, bool(self.include_tiles))
        bw.ClientConfigAddIncludePlayers(b, bool(self.include_players))
        bw.ClientConfigAddLocalSpeed(b, int(self.local_speed))
        bw.ClientConfigAddGui(b, bool(self.gui))
        bw.ClientConfigAddCompleteMapInformation(b, bool(self.complete_map_information))
        bw.ClientConfigAddUserInput(b, bool(self.user_input))
        cfg = bw.ClientConfigEnd(b)
        bw.EnvelopeStart(b)
        bw.EnvelopeAddMsgType(b, bw.Message.ClientConfig)
        bw.EnvelopeAddMsg(b, cfg)
        env = bw.EnvelopeEnd(b)
        b.Finish(env, b"BWB1")
        return b.Output()


@dataclass
class Hello:
    protocol_version: int
    shim_version: str
    backend: str
    client_version: int
    revision: int

    @classmethod
    def from_fb(cls, h: bw.Hello) -> "Hello":
        def s(x: Optional[bytes]) -> str:
            return x.decode("utf-8", "replace") if x else ""

        return cls(h.ProtocolVersion(), s(h.ShimVersion()), s(h.Backend()), h.ClientVersion(), h.Revision())


def envelope_kind(env: bw.Envelope) -> int:
    return env.MsgType()


def envelope_as(env: bw.Envelope, cls):
    """Return the union payload as the given generated table class."""
    tab = env.Msg()
    if tab is None:
        raise ProtocolError("empty envelope")
    obj = cls()
    obj.Init(tab.Bytes, tab.Pos)
    return obj
