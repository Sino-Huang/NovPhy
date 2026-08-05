import struct
import socket
import threading
import unittest
from dataclasses import FrozenInstanceError
from typing import Any, get_type_hints

from src.webui.bridge import (
    GameState,
    PhysicsCaptureV1,
    PhysicsCaptureV1Failure,
    PhysicsCaptureV1ProtocolError,
    ScienceBirdsBridge,
    encode_physics_capture_v1,
)


class FakeSocket:
    def __init__(self, *args):
        self.sent = bytearray()
        self.responses = bytearray()
        self.connected_to = None
        self.closed = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect(self, address):
        self.connected_to = address

    def sendall(self, data):
        self.sent.extend(data)

    def recv(self, size):
        chunk = self.responses[:size]
        del self.responses[:size]
        return bytes(chunk)

    def close(self):
        self.closed = True


class BridgeTest(unittest.TestCase):
    def make_bridge(self):
        fake = FakeSocket()
        bridge = ScienceBirdsBridge("127.0.0.1", 2004, socket_factory=lambda *args: fake)
        bridge.connect()
        return bridge, fake

    def test_configure_sends_training_mode_and_reads_response(self):
        bridge, fake = self.make_bridge()
        fake.responses.extend(struct.pack("!BBB", 0, 0, 20))

        result = bridge.configure(agent_id=28888)

        self.assertEqual(result, (0, 0, 20))
        self.assertEqual(bytes(fake.sent), struct.pack("!BIB", 1, 28888, 1))

    def test_shoot_uses_existing_cartesian_protocol(self):
        bridge, fake = self.make_bridge()
        fake.responses.extend(struct.pack("!B", 1))

        result = bridge.shoot(120, 220, tap_time=30)

        self.assertEqual(result, 1)
        self.assertEqual(bytes(fake.sent), struct.pack("!Biiii", 31, 120, 220, 0, 30))

    def test_fast_shoot_uses_fast_message_code(self):
        bridge, fake = self.make_bridge()
        fake.responses.extend(struct.pack("!B", 1))

        bridge.shoot(120, 220, tap_time=30, fast=True)

        self.assertEqual(bytes(fake.sent), struct.pack("!Biiii", 41, 120, 220, 0, 30))

    def test_screenshot_reads_exact_rgb_payload(self):
        bridge, fake = self.make_bridge()
        rgb = bytes([255, 0, 0, 0, 255, 0])
        fake.responses.extend(struct.pack("!II", 2, 1) + rgb)

        screenshot = bridge.screenshot()

        self.assertEqual(screenshot.width, 2)
        self.assertEqual(screenshot.height, 1)
        self.assertEqual(screenshot.rgb, rgb)
        self.assertEqual(bytes(fake.sent), struct.pack("!B", 11))

    def test_state_maps_known_state_values(self):
        bridge, fake = self.make_bridge()
        fake.responses.extend(struct.pack("!B", 5))

        self.assertEqual(bridge.get_game_state(), GameState.PLAYING)
        self.assertEqual(bytes(fake.sent), struct.pack("!B", 12))

    def test_ready_for_new_set_sends_protocol_and_reads_metadata(self):
        bridge, fake = self.make_bridge()
        fake.responses.extend(struct.pack("!IIIIBBB", 9000, 60000, 200, 300, 1, 0, 0))

        result = bridge.ready_for_new_set()

        self.assertEqual(result, (9000, 60000, 200, 300, 1, 0, 0))
        self.assertEqual(bytes(fake.sent), struct.pack("!B", 68))

    def test_get_novelty_info_sends_protocol_and_reads_flag(self):
        bridge, fake = self.make_bridge()
        fake.responses.extend(struct.pack("!i", 0))

        result = bridge.get_novelty_info()

        self.assertEqual(result, 0)
        self.assertEqual(bytes(fake.sent), struct.pack("!B", 69))

    def test_load_next_available_level_sends_protocol_and_reads_level(self):
        bridge, fake = self.make_bridge()
        fake.responses.extend(struct.pack("!I", 1))

        result = bridge.load_next_available_level()

        self.assertEqual(result, 1)
        self.assertEqual(bytes(fake.sent), struct.pack("!B", 53))


class PhysicsCaptureV1Tests(unittest.TestCase):
    def make_bridge(self, response):
        fake = FakeSocket()
        fake.responses.extend(response)
        bridge = ScienceBirdsBridge("127.0.0.1", 2004, socket_factory=lambda *args: fake)
        bridge.connect()
        return bridge, fake

    def test_request_70_round_trips_png_state_and_events_at_one_render_frame(self):
        state = {"schema_version": "physics_capture_v1", "render_frame": 42, "nodes": []}
        events = [{"schema_version": "physics_capture_v1", "render_frame": 42, "sequence": 1}]
        png = b"\x89PNG\r\n\x1a\nframe"
        bridge, fake = self.make_bridge(encode_physics_capture_v1(png, state, events))

        capture = bridge.get_physics_capture_v1()

        self.assertIsInstance(capture, PhysicsCaptureV1)
        self.assertEqual(capture.png, png)
        self.assertEqual(capture.state["render_frame"], 42)
        self.assertEqual(capture.events[0]["render_frame"], 42)
        self.assertEqual(bytes(fake.sent), b"\x46")

    def test_request_70_reconnects_for_each_one_response_direct_socket(self):
        response = encode_physics_capture_v1(
            b"\x89PNG\r\n\x1a\nframe",
            {"schema_version": "physics_capture_v1", "render_frame": 42, "nodes": []},
            [],
        )
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(2)
        requests = []

        def serve_two_requests():
            with server:
                for _ in range(2):
                    connection, _ = server.accept()
                    with connection:
                        requests.append(connection.recv(1))
                        connection.sendall(response)

        worker = threading.Thread(target=serve_two_requests, daemon=True)
        worker.start()
        bridge = ScienceBirdsBridge("127.0.0.1", server.getsockname()[1], timeout=1.0)
        bridge.connect()
        try:
            captures = [bridge.get_physics_capture_v1() for _ in range(2)]
        finally:
            bridge.disconnect()
            worker.join(timeout=2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(requests, [b"\x46", b"\x46"])
        self.assertEqual([capture.state["render_frame"] for capture in captures], [42, 42])

    def test_request_70_preserves_nested_decoded_values(self):
        state = {
            "schema_version": "physics_capture_v1",
            "render_frame": 42,
            "nodes": [{"entity_id": "bird:1", "velocity": [1.5, -2.0]}],
        }
        events = [{"render_frame": 42, "participants": ["bird:1", "pig:1"]}]
        bridge, _ = self.make_bridge(
            encode_physics_capture_v1(b"\x89PNG\r\n\x1a\nframe", state, events)
        )

        capture = bridge.get_physics_capture_v1()

        self.assertEqual(capture.state["nodes"][0]["entity_id"], "bird:1")
        self.assertEqual(capture.state["nodes"][0]["velocity"], (1.5, -2.0))
        self.assertEqual(capture.events[0]["participants"], ("bird:1", "pig:1"))

    def test_request_70_public_records_have_recursive_typed_annotations(self):
        annotations = get_type_hints(PhysicsCaptureV1)

        self.assertNotEqual(annotations["state"], Any)
        self.assertNotEqual(annotations["events"], tuple[Any, ...])

    def test_request_70_failure_fields_are_immutable(self):
        failure = PhysicsCaptureV1Failure(7, "capture failed")

        with self.assertRaises(FrozenInstanceError):
            failure.code = 8
        with self.assertRaises(FrozenInstanceError):
            failure.message = "changed"

    def test_legacy_request_38_and_62_fixture_bytes_remain_unchanged(self):
        bridge, fake = self.make_bridge(struct.pack("!I", 7) + b"{}xxxxx")
        bridge._send(38, "iiiii", 1, 2, 3, 4, 5)
        bridge.get_symbolic_state_without_screenshot()
        self.assertEqual(bytes(fake.sent), struct.pack("!BiiiiiB", 38, 1, 2, 3, 4, 5, 62))

    def test_recorder_backed_shoot_preserves_request_38_framing_and_consumes_ground_truth(self):
        response = struct.pack("!I", 2) + struct.pack("!I", 7) + b"{}xxxxx" + struct.pack("!I", 7) + b"{}xxxxx"
        bridge, fake = self.make_bridge(response)

        ground_truth_count = bridge.shoot_and_record_ground_truth(1, 2, tap_time=4, release_time=3)

        self.assertEqual(ground_truth_count, 2)
        self.assertEqual(bytes(fake.sent), struct.pack("!Biiiii", 38, 1, 2, 3, 4, 1))
        self.assertEqual(bytes(fake.responses), b"")


class PhysicsCaptureV1MalformedEnvelopeTests(unittest.TestCase):
    def make_bridge(self, response):
        fake = FakeSocket()
        fake.responses.extend(response)
        bridge = ScienceBirdsBridge(socket_factory=lambda *args: fake)
        bridge.connect()
        return bridge

    def assert_rejected(self, response):
        bridge = self.make_bridge(response)
        with self.assertRaises(PhysicsCaptureV1ProtocolError):
            bridge.get_physics_capture_v1()
        self.assertFalse(bridge.connected)

    def test_rejects_bad_magic(self):
        self.assert_rejected(_envelope(b"NOPE", 1, 0, 0, b""))

    def test_rejects_bad_version(self):
        self.assert_rejected(_envelope(b"SBPV", 99, 0, 0, b""))

    def test_rejects_bad_length(self):
        self.assert_rejected(struct.pack("!I", 100) + b"short")

    def test_rejects_bad_json(self):
        payload = struct.pack("!III", 8, 2, 2) + b"\x89PNG\r\n\x1a\n" + b"{}" + b"[]"
        self.assert_rejected(_envelope(b"SBPV", 1, 0, 0, payload))

    def test_rejects_overflow(self):
        self.assert_rejected(struct.pack("!I", 64 * 1024 * 1024 + 1))

    def test_rejects_invalid_flags(self):
        self.assert_rejected(_envelope(b"SBPV", 1, 2, 0, b""))

    def test_rejects_failure_length_mismatch(self):
        self.assert_rejected(_envelope(b"SBPV", 1, 1, 3, struct.pack("!I", 9) + b"no"))

    def test_rejects_render_frame_mismatch(self):
        payload = struct.pack("!III", 8, 54, 54)
        payload += b"\x89PNG\r\n\x1a\n"
        payload += b'{"schema_version":"physics_capture_v1","render_frame":1}'
        payload += b'[{"schema_version":"physics_capture_v1","render_frame":2}]'
        self.assert_rejected(_envelope(b"SBPV", 1, 0, 0, payload))

    def test_rejects_non_png_payload(self):
        state = b'{"schema_version":"physics_capture_v1","render_frame":1}'
        events = b'[]'
        payload = struct.pack("!III", 4, len(state), len(events)) + b"nope" + state + events
        self.assert_rejected(_envelope(b"SBPV", 1, 0, 0, payload))


def _envelope(magic, version, flags, failure_code, payload):
    body = struct.pack("!4sBBHI", magic, version, flags, failure_code, len(payload)) + payload
    return struct.pack("!I", len(body)) + body


if __name__ == "__main__":
    unittest.main()
