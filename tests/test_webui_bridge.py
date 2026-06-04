import struct
import unittest

from src.webui.bridge import GameState, ScienceBirdsBridge


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


if __name__ == "__main__":
    unittest.main()
