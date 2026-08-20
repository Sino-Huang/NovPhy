from __future__ import annotations
# noqa: SIZE_OK - legacy and request-70 wire surfaces must remain in this owned module.

import json
import socket
import struct
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Final, Mapping, TypeAlias


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | tuple["JsonValue", ...]
PhysicsStateV1: TypeAlias = Mapping[str, JsonValue]
PhysicsEventV1: TypeAlias = Mapping[str, JsonValue]
PhysicsViolationEngineEvidenceV1: TypeAlias = Mapping[str, JsonValue]
PhysicsCaptureV2EngineRecord: TypeAlias = Mapping[str, JsonValue]


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
    GET_PHYSICS_CAPTURE_V1 = 70
    GET_PHYSICS_CAPTURE_V2 = 71
    GT_SHOOT = 38


class PhysicsCaptureV1ProtocolError(ConnectionError):
    """The request-70 stream was malformed or could not be completed."""


class PhysicsCaptureV2ProtocolError(ConnectionError):
    """The request-71 stream was malformed or could not be completed."""


@dataclass(frozen=True, slots=True)
class LegacyGroundTruthProtocolError(ConnectionError):
    request_code: int
    field: str
    detail: str
    value: int | None = None
    limit: int | None = None

    def __post_init__(self) -> None:
        ConnectionError.__init__(self, str(self))

    def __str__(self) -> str:
        message = "request-%d %s: %s" % (self.request_code, self.field, self.detail)
        if self.value is None:
            return message
        return "%s (value=%d, limit=%d)" % (message, self.value, self.limit)


@dataclass(frozen=True, slots=True)
class PhysicsCaptureV1Failure(PhysicsCaptureV1ProtocolError):
    code: int
    message: str

    def __post_init__(self) -> None:
        PhysicsCaptureV1ProtocolError.__init__(
            self, "physics capture failed (%d): %s" % (self.code, self.message)
        )


@dataclass(frozen=True, slots=True)
class PhysicsCaptureV2Failure(PhysicsCaptureV2ProtocolError):
    code: int
    message: str

    def __post_init__(self) -> None:
        PhysicsCaptureV2ProtocolError.__init__(
            self, "physics capture v2 failed (%d): %s" % (self.code, self.message)
        )


@dataclass(frozen=True, slots=True)
class Screenshot:
    width: int
    height: int
    rgb: bytes


@dataclass(frozen=True, slots=True)
class PhysicsCaptureV1:
    png: bytes
    state: PhysicsStateV1
    events: tuple[PhysicsEventV1, ...]
    evidence: PhysicsViolationEngineEvidenceV1 | None = None


@dataclass(frozen=True, slots=True)
class PhysicsCaptureV2Engine:
    record: PhysicsCaptureV2EngineRecord


_PHYSICS_MAGIC = b"SBPV"
_PHYSICS_VERSION = 1
_PHYSICS_FAILURE_FLAG = 1
_PHYSICS_MAX_ENVELOPE = 64 * 1024 * 1024
_PHYSICS_MAX_PNG = 32 * 1024 * 1024
_PHYSICS_MAX_JSON = 16 * 1024 * 1024
_PHYSICS_V2_MAGIC = b"SBP2"
_PHYSICS_V2_VERSION = 1
_PHYSICS_V2_FAILURE_FLAG = 1
_PHYSICS_V2_MAX_ENVELOPE = 64 * 1024 * 1024
_PHYSICS_V2_MAX_JSON = _PHYSICS_V2_MAX_ENVELOPE - 12
_LEGACY_MAX_GROUND_TRUTH_RECORDS: Final = 10_000
_LEGACY_MAX_GROUND_TRUTH_PAYLOAD: Final = 16 * 1024 * 1024
_LEGACY_GROUND_TRUTH_SUFFIX_BYTES: Final = 5


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
        try:
            return self._read_ground_truth(RequestCode.GET_GROUND_TRUTH_WITHOUT_SCREENSHOT)
        except LegacyGroundTruthProtocolError:
            self.disconnect()
            raise

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

    def get_physics_capture_v1(self) -> PhysicsCaptureV1:
        if not self.connected:
            self.connect()
        try:
            self._send(RequestCode.GET_PHYSICS_CAPTURE_V1)
            envelope_length = struct.unpack("!I", self._read_exact(4))[0]
            if envelope_length < 16 or envelope_length > _PHYSICS_MAX_ENVELOPE:
                raise PhysicsCaptureV1ProtocolError("invalid request-70 envelope length")
            body = self._read_exact(envelope_length)
            return _decode_physics_capture_v1(body)
        except PhysicsCaptureV1ProtocolError:
            self.disconnect()
            raise
        except (ConnectionError, OSError, ValueError, struct.error, UnicodeError, RecursionError) as exc:
            raise PhysicsCaptureV1ProtocolError("invalid request-70 envelope") from exc
        finally:
            self.disconnect()

    def get_physics_capture_v2(self) -> PhysicsCaptureV2Engine:
        if not self.connected:
            self.connect()
        try:
            self._send(RequestCode.GET_PHYSICS_CAPTURE_V2)
            envelope_length = struct.unpack("!I", self._read_exact(4))[0]
            if envelope_length < 12 or envelope_length > _PHYSICS_V2_MAX_ENVELOPE:
                raise PhysicsCaptureV2ProtocolError("invalid request-71 envelope length")
            body = self._read_exact(envelope_length)
            return _decode_physics_capture_v2_engine(body)
        except PhysicsCaptureV2ProtocolError:
            self.disconnect()
            raise
        except (ConnectionError, OSError, ValueError, struct.error, UnicodeError, RecursionError) as exc:
            raise PhysicsCaptureV2ProtocolError("invalid request-71 envelope") from exc
        finally:
            self.disconnect()

    def shoot(self, x: int, y: int, tap_time: int = 0, fast: bool = False, release_time: int = 0) -> int:
        code = RequestCode.FAST_SHOOT if fast else RequestCode.SHOOT
        self._send(code, "iiii", int(x), int(y), int(release_time), int(tap_time))
        return self._read("B")[0]

    def shoot_and_record_ground_truth(
        self,
        x: int,
        y: int,
        tap_time: int = 0,
        release_time: int = 0,
        frequency: int = 1,
    ) -> int:
        """Execute the recorder-backed legacy request-38 action."""
        self._send(
            RequestCode.GT_SHOOT,
            "iiiii",
            int(x),
            int(y),
            int(release_time),
            int(tap_time),
            int(frequency),
        )
        try:
            ground_truth_count = self._read("I")[0]
            if ground_truth_count > _LEGACY_MAX_GROUND_TRUTH_RECORDS:
                raise LegacyGroundTruthProtocolError(
                    int(RequestCode.GT_SHOOT),
                    "record_count",
                    "exceeds maximum",
                    ground_truth_count,
                    _LEGACY_MAX_GROUND_TRUTH_RECORDS,
                )
            for _ in range(ground_truth_count):
                self._read_ground_truth(RequestCode.GT_SHOOT)
            return ground_truth_count
        except LegacyGroundTruthProtocolError:
            self.disconnect()
            raise
        except (ConnectionError, OSError, struct.error) as exc:
            self.disconnect()
            raise LegacyGroundTruthProtocolError(
                int(RequestCode.GT_SHOOT), "record_count", "is truncated"
            ) from exc

    def _send(self, code: RequestCode, fmt: str = "", *values: int) -> None:
        sock = self._require_socket()
        sock.sendall(struct.pack("!B" + fmt, int(code), *values))

    def _read(self, fmt: str):
        return struct.unpack("!" + fmt, self._read_exact(struct.calcsize("!" + fmt)))

    def _read_ground_truth(self, request_code: RequestCode):
        try:
            payload_length = self._read("I")[0]
        except (ConnectionError, OSError, struct.error) as exc:
            raise LegacyGroundTruthProtocolError(
                int(request_code), "payload_length", "is truncated"
            ) from exc
        if payload_length > _LEGACY_MAX_GROUND_TRUTH_PAYLOAD:
            raise LegacyGroundTruthProtocolError(
                int(request_code),
                "payload_length",
                "exceeds maximum",
                payload_length,
                _LEGACY_MAX_GROUND_TRUTH_PAYLOAD,
            )
        if payload_length < _LEGACY_GROUND_TRUTH_SUFFIX_BYTES:
            raise LegacyGroundTruthProtocolError(
                int(request_code),
                "payload_length",
                "is smaller than the legacy suffix",
                payload_length,
                _LEGACY_GROUND_TRUTH_SUFFIX_BYTES,
            )
        try:
            payload = self._read_exact(payload_length)
            return json.loads(payload.decode("utf-8")[:-_LEGACY_GROUND_TRUTH_SUFFIX_BYTES])
        except (ConnectionError, OSError, UnicodeError, ValueError, RecursionError) as exc:
            raise LegacyGroundTruthProtocolError(
                int(request_code), "payload", "is malformed or truncated"
            ) from exc

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


def encode_physics_capture_v1(
    png: bytes,
    state: dict,
    events: list[dict],
    evidence: dict | None = None,
) -> bytes:
    """Build the canonical request-70 response, useful for protocol fixtures."""
    png = bytes(png)
    state_bytes = json.dumps(state, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    events_bytes = json.dumps(events, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    if evidence is None:
        payload = struct.pack("!III", len(png), len(state_bytes), len(events_bytes))
        payload += png + state_bytes + events_bytes
        return _encode_envelope(0, 0, payload)
    evidence_bytes = json.dumps(evidence, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    payload = struct.pack(
        "!IIII", len(png), len(state_bytes), len(events_bytes), len(evidence_bytes)
    )
    payload += png + state_bytes + events_bytes + evidence_bytes
    return _encode_envelope(2, 0, payload)


def encode_physics_capture_v2_engine(record: Mapping[str, JsonValue]) -> bytes:
    """Build the canonical request-71 engine response for protocol fixtures."""
    payload = json.dumps(
        record, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    body = struct.pack(
        "!4sBBHI", _PHYSICS_V2_MAGIC, _PHYSICS_V2_VERSION, 0, 0, len(payload)
    ) + payload
    return struct.pack("!I", len(body)) + body


def _encode_envelope(flags: int, failure_code: int, payload: bytes) -> bytes:
    body = struct.pack("!4sBBHI", _PHYSICS_MAGIC, _PHYSICS_VERSION, flags,
                       failure_code, len(payload)) + payload
    return struct.pack("!I", len(body)) + body


def _decode_physics_capture_v1(body: bytes) -> PhysicsCaptureV1:
    if len(body) < 16:
        raise PhysicsCaptureV1ProtocolError("request-70 envelope header is truncated")
    magic, version, flags, failure_code, payload_length = struct.unpack("!4sBBHI", body[:12])
    if magic != _PHYSICS_MAGIC:
        raise PhysicsCaptureV1ProtocolError("request-70 magic mismatch")
    if version != _PHYSICS_VERSION:
        raise PhysicsCaptureV1ProtocolError("unsupported request-70 protocol version")
    if payload_length != len(body) - 12:
        raise PhysicsCaptureV1ProtocolError("request-70 payload length mismatch")
    payload = body[12:]
    if flags == _PHYSICS_FAILURE_FLAG:
        if len(payload) < 4:
            raise PhysicsCaptureV1ProtocolError("request-70 failure payload is truncated")
        message_length = struct.unpack("!I", payload[:4])[0]
        if message_length > _PHYSICS_MAX_JSON or message_length != len(payload) - 4:
            raise PhysicsCaptureV1ProtocolError("request-70 failure message length mismatch")
        try:
            message = payload[4:].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PhysicsCaptureV1ProtocolError("request-70 failure message is not UTF-8") from exc
        raise PhysicsCaptureV1Failure(failure_code, message)
    if flags not in (0, 2) or failure_code != 0:
        raise PhysicsCaptureV1ProtocolError("invalid request-70 success flags")
    component_header_length = 16 if flags == 2 else 12
    if len(payload) < component_header_length:
        raise PhysicsCaptureV1ProtocolError("request-70 success payload is truncated")
    if flags == 2:
        png_length, state_length, events_length, evidence_length = struct.unpack("!IIII", payload[:16])
    else:
        png_length, state_length, events_length = struct.unpack("!III", payload[:12])
        evidence_length = 0
    if (png_length > _PHYSICS_MAX_PNG or state_length > _PHYSICS_MAX_JSON
            or events_length > _PHYSICS_MAX_JSON or evidence_length > _PHYSICS_MAX_JSON):
        raise PhysicsCaptureV1ProtocolError("request-70 payload exceeds bounds")
    if png_length + state_length + events_length + evidence_length != len(payload) - component_header_length:
        raise PhysicsCaptureV1ProtocolError("request-70 record lengths mismatch")
    offset = component_header_length
    png = payload[offset:offset + png_length]
    offset += png_length
    state = _parse_json_record(payload[offset:offset + state_length], dict, "state")
    offset += state_length
    events = _parse_json_record(payload[offset:offset + events_length], list, "events")
    offset += events_length
    evidence = None
    if flags == 2:
        evidence = _parse_json_record(payload[offset:offset + evidence_length], (dict, type(None)), "evidence")
    render_frame = state.get("render_frame")
    if isinstance(render_frame, bool) or not isinstance(render_frame, int):
        raise PhysicsCaptureV1ProtocolError("state render_frame is missing or invalid")
    for event in events:
        if not isinstance(event, dict) or event.get("render_frame") != render_frame:
            raise PhysicsCaptureV1ProtocolError("state and event render_frame mismatch")
    if evidence is not None:
        if evidence.get("schema_version") != "physics_violation_engine_evidence_v1":
            raise PhysicsCaptureV1ProtocolError("request-70 evidence schema is unsupported")
        if evidence.get("capture_id") != state.get("capture_id") or evidence.get("sequence") != state.get("sequence"):
            raise PhysicsCaptureV1ProtocolError("state and evidence identity mismatch")
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise PhysicsCaptureV1ProtocolError("request-70 payload is not a PNG")
    return PhysicsCaptureV1(
        bytes(png),
        _freeze(state),
        tuple(_freeze(event) for event in events),
        None if evidence is None else _freeze(evidence),
    )


def _decode_physics_capture_v2_engine(body: bytes) -> PhysicsCaptureV2Engine:
    if len(body) < 12:
        raise PhysicsCaptureV2ProtocolError("request-71 envelope header is truncated")
    magic, version, flags, failure_code, payload_length = struct.unpack("!4sBBHI", body[:12])
    if magic != _PHYSICS_V2_MAGIC:
        raise PhysicsCaptureV2ProtocolError("request-71 magic mismatch")
    if version != _PHYSICS_V2_VERSION:
        raise PhysicsCaptureV2ProtocolError("unsupported request-71 protocol version")
    payload = body[12:]
    if payload_length > _PHYSICS_V2_MAX_JSON or payload_length != len(payload):
        raise PhysicsCaptureV2ProtocolError("request-71 payload length is invalid")
    if flags == _PHYSICS_V2_FAILURE_FLAG:
        if len(payload) < 4:
            raise PhysicsCaptureV2ProtocolError("request-71 failure payload is truncated")
        message_length = struct.unpack("!I", payload[:4])[0]
        if message_length != len(payload) - 4:
            raise PhysicsCaptureV2ProtocolError("request-71 failure message length mismatch")
        try:
            message = payload[4:].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PhysicsCaptureV2ProtocolError("request-71 failure message is not UTF-8") from exc
        raise PhysicsCaptureV2Failure(failure_code, message)
    if flags != 0 or failure_code != 0:
        raise PhysicsCaptureV2ProtocolError("invalid request-71 success envelope")
    record = _parse_json_record(payload, dict, "request-71 engine capture")
    if record.get("schema_version") != "physics_capture_v2_engine_v1":
        raise PhysicsCaptureV2ProtocolError("request-71 engine schema is unsupported")
    stride = record.get("configured_fixed_step_capture_stride")
    if type(stride) is not int or stride <= 0:
        raise PhysicsCaptureV2ProtocolError("request-71 capture stride is missing or invalid")
    return PhysicsCaptureV2Engine(_freeze(record))


def _parse_json_record(data: bytes, expected_type: type | tuple[type, ...], name: str):
    try:
        value = json.loads(data.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PhysicsCaptureV1ProtocolError("request-70 %s JSON is invalid" % name) from exc
    if not isinstance(value, expected_type):
        raise PhysicsCaptureV1ProtocolError("request-70 %s JSON has the wrong type" % name)
    return value


def _reject_json_constant(value: str):
    raise ValueError("non-finite JSON value: " + value)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value
