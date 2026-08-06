import json
import socket
import struct
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.webui.bridge import ScienceBirdsBridge, encode_physics_capture_v1


png = b"\x89PNG\r\n\x1a\n" + struct.pack("!I", 13) + b"IHDR" + b"fixture"
state = {
    "schema_version": "physics_capture_v1",
    "render_frame": 913,
    "nodes": [{"entity_id": "bird:1", "velocity": [1.5, -2.0]}],
}
events = [
    {
        "schema_version": "physics_capture_v1",
        "render_frame": 913,
        "participants": ["bird:1"],
    }
]
envelope = encode_physics_capture_v1(png, state, events)
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("127.0.0.1", 0))
server.listen(1)
port = server.getsockname()[1]
request = bytearray()


def serve() -> None:
    connection, _ = server.accept()
    with connection:
        request.extend(connection.recv(1))
        connection.sendall(envelope)
    server.close()


thread = threading.Thread(target=serve)
thread.start()
bridge = ScienceBirdsBridge("127.0.0.1", port)
bridge.connect()
capture = bridge.get_physics_capture_v1()
bridge.disconnect()
thread.join()

immutable = False
try:
    capture.state["nodes"][0]["velocity"] = (0.0, 0.0)
except TypeError:
    immutable = True

body_length = struct.unpack("!I", envelope[:4])[0]
result = {
    "request_code": request[0],
    "request_byte_hex": bytes(request).hex(),
    "png_signature": capture.png[:8].hex(),
    "png_ihdr": capture.png[12:16].decode("ascii"),
    "state_render_frame": capture.state["render_frame"],
    "event_render_frames": [event["render_frame"] for event in capture.events],
    "nested_velocity": list(capture.state["nodes"][0]["velocity"]),
    "nested_participants": list(capture.events[0]["participants"]),
    "outer_length_equal": body_length == len(envelope) - 4,
    "recursive_immutable": immutable,
}
expected = {
    "request_code": 70,
    "request_byte_hex": "46",
    "png_signature": "89504e470d0a1a0a",
    "png_ihdr": "IHDR",
    "state_render_frame": 913,
    "event_render_frames": [913],
    "nested_velocity": [1.5, -2.0],
    "nested_participants": ["bird:1"],
    "outer_length_equal": True,
    "recursive_immutable": True,
}
print(json.dumps({"passed": result == expected, "observed": result}, indent=2))
raise SystemExit(0 if result == expected else 1)
