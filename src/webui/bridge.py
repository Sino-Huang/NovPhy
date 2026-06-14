from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from enum import IntEnum


class GameState(IntEnum):
    UNKNOWN = 0
    MAIN_MENU = 1
    EPISODE_MENU = 2
    LEVEL_SELECTION = 3
    LOADING = 4
    PLAYING = 5
    WON = 6
    LOST = 7
    NEWTESTSET = 8
    NEWTRAININGSET = 9
    RESUMETRAINING = 10
    NEWTRIAL = 11
    REQUESTNOVELTYLIKELIHOOD = 12
    EVALUATION_TERMINATED = 13


class PlayingMode(IntEnum):
    COMPETITION = 0
    TRAINING = 1


class RequestCode(IntEnum):
    CONFIGURE = 1
    SET_SPEED = 2
    SCREENSHOT = 11
    GET_STATE = 12
    GET_CURRENT_LEVEL = 14
    GET_NUMBER_OF_LEVELS = 15
    GET_GROUND_TRUTH_WITHOUT_SCREENSHOT = 62
    SHOOT = 31
    FULLY_ZOOM_OUT = 34
    FULLY_ZOOM_IN = 35
    FAST_SHOOT = 41
    LOAD_LEVEL = 51
    RESTART_LEVEL = 52
    LOAD_NEXT_AVAILABLE_LEVEL = 53
    READY_FOR_NEW_SET = 68
    NOVELTY_INFO = 69
    GET_CURRENT_SCORE = 65


@dataclass(frozen=True)
class Screenshot:
    width: int
    height: int
    rgb: bytes


class ScienceBirdsBridge:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2004,
        timeout: float = 300.0,
        socket_factory=socket.socket,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = timeout
        self._socket_factory = socket_factory
        self._socket: socket.socket | None = None
        self._buffer = bytearray()

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        if self._socket is not None:
            return
        sock = self._socket_factory(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        self._socket = sock

    def disconnect(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close()
        finally:
            self._socket = None
            self._buffer.clear()

    def configure(self, agent_id: int = 28888, mode: PlayingMode = PlayingMode.TRAINING) -> tuple[int, int, int]:
        self._send(RequestCode.CONFIGURE, "IB", int(agent_id), int(mode))
        return self._read("BBB")

    def set_speed(self, speed: int) -> int:
        self._send(RequestCode.SET_SPEED, "I", int(speed))
        return self._read("B")[0]

    def get_game_state(self) -> GameState:
        self._send(RequestCode.GET_STATE)
        value = self._read("B")[0]
        try:
            return GameState(value)
        except ValueError:
            return GameState.UNKNOWN

    def get_current_level(self) -> int:
        self._send(RequestCode.GET_CURRENT_LEVEL)
        return self._read("I")[0]

    def get_number_of_levels(self) -> int:
        self._send(RequestCode.GET_NUMBER_OF_LEVELS)
        return self._read("I")[0]

    def get_current_score(self) -> int:
        self._send(RequestCode.GET_CURRENT_SCORE)
        return self._read("I")[0]

    def get_symbolic_state_without_screenshot(self):
        self._send(RequestCode.GET_GROUND_TRUTH_WITHOUT_SCREENSHOT)
        return self._read_ground_truth()

    def load_level(self, level: int) -> int:
        self._send(RequestCode.LOAD_LEVEL, "I", max(1, int(level)))
        return self._read("B")[0]

    def load_next_available_level(self) -> int:
        self._send(RequestCode.LOAD_NEXT_AVAILABLE_LEVEL)
        return self._read("I")[0]

    def ready_for_new_set(self) -> tuple[int, int, int, int, int, int, int]:
        self._send(RequestCode.READY_FOR_NEW_SET)
        return self._read("IIIIBBB")

    def get_novelty_info(self) -> int:
        self._send(RequestCode.NOVELTY_INFO)
        return self._read("i")[0]

    def restart_level(self) -> int:
        self._send(RequestCode.RESTART_LEVEL)
        return self._read("B")[0]

    def fully_zoom_out(self) -> int:
        self._send(RequestCode.FULLY_ZOOM_OUT)
        return self._read("B")[0]

    def fully_zoom_in(self) -> int:
        self._send(RequestCode.FULLY_ZOOM_IN)
        return self._read("B")[0]

    def screenshot(self) -> Screenshot:
        self._send(RequestCode.SCREENSHOT)
        width, height = self._read("II")
        rgb = self._read_exact(width * height * 3)
        return Screenshot(width=width, height=height, rgb=bytes(rgb))

    def shoot(self, x: int, y: int, tap_time: int = 0, fast: bool = False, release_time: int = 0) -> int:
        code = RequestCode.FAST_SHOOT if fast else RequestCode.SHOOT
        self._send(code, "iiii", int(x), int(y), int(release_time), int(tap_time))
        return self._read("B")[0]

    def _send(self, code: RequestCode, fmt: str = "", *values: int) -> None:
        sock = self._require_socket()
        sock.sendall(struct.pack("!B" + fmt, int(code), *values))

    def _read(self, fmt: str):
        return struct.unpack("!" + fmt, self._read_exact(struct.calcsize("!" + fmt)))

    def _read_ground_truth(self):
        payload_length = self._read("I")[0]
        payload = self._read_exact(payload_length)
        return json.loads(payload.decode("utf-8")[:-5])

    def _read_exact(self, size: int) -> bytes:
        sock = self._require_socket()
        while len(self._buffer) < size:
            chunk = sock.recv(size - len(self._buffer))
            if not chunk:
                raise ConnectionError("Science Birds socket closed while reading")
            self._buffer.extend(chunk)
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def _require_socket(self):
        if self._socket is None:
            raise ConnectionError("Science Birds bridge is not connected")
        return self._socket
