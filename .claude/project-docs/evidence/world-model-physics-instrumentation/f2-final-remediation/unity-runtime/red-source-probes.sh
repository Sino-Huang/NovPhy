#!/usr/bin/env bash
set -u

python3 - "tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsCaptureProtocol.cs" \
  "tasks/task_template_designer/Assets/Scripts/GroundTruth/PhysicsShotRecorder.cs" \
  "tasks/task_template_designer/Assets/Scripts/GameWorld/Characters/Birds/ABBird.cs" <<'PY'
import re
import sys
from pathlib import Path


def method_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening:index + 1]
    raise ValueError("unterminated method: " + signature)


protocol = Path(sys.argv[1]).read_text(encoding="utf-8-sig")
recorder = Path(sys.argv[2]).read_text(encoding="utf-8-sig")
bird = Path(sys.argv[3]).read_text(encoding="utf-8-sig")
failures = []

serve = method_body(protocol, "private IEnumerator Serve")
if re.search(r"\bstream\.Write\s*\(", serve) or "BeginWrite" not in protocol:
    failures.append(
        "RED transport: Serve still performs synchronous NetworkStream.Write and has no bounded BeginWrite transmitter"
    )

append_string = method_body(protocol, "private static void AppendString")
if not re.search(r"(?:c|character)\s*<\s*32", append_string) or "\\\\b" not in append_string or "\\\\f" not in append_string:
    failures.append(
        "RED JSON: PhysicsCaptureV1Protocol.AppendString does not escape every U+0000-U+001F character"
    )

record_contacts = method_body(
    recorder,
    "public void RecordContacts(long fixedStep, float fixedTime, PhysicalContactInput[] contacts)",
)
if "fixedTime > limits.TimeoutSeconds" in record_contacts or "shotStartFixedTime" not in recorder:
    failures.append(
        "RED timeout: RecordContacts compares absolute fixedTime to TimeoutSeconds instead of elapsed shot time"
    )

one_shot = method_body(recorder, "private void AddOneShotEvent")
terminal = method_body(recorder, "private void AddTerminalEvent")
fail = method_body(recorder, "private void Fail(")
finalize = method_body(recorder, "public PhysicalCaptureResult FinalizeShot")
finalized_assignment = finalize.find("finalized = true")
truncated_failure = finalize.find("TruncatedFinalization")
if (
    "finalized" not in one_shot
    or "finalized" not in terminal
    or "finalized" not in fail
    or (finalized_assignment >= 0 and truncated_failure >= 0 and finalized_assignment < truncated_failure)
):
    failures.append(
        "RED finality: helper/failure mutation bypasses finalized, and truncated failure is attempted after finalization"
    )

launch = method_body(bird, "public void LaunchBird")
callback_position = launch.find("RecordLaunchCallback")
assigned_position = launch.find("_rigidBody.velocity = direction")
if callback_position < 0 or assigned_position < 0 or callback_position < assigned_position:
    failures.append(
        "RED launch: bird_launched callback occurs before the computed Rigidbody2D velocity assignment"
    )

for failure in failures:
    print(failure)
print("RED_DEFECT_COUNT=" + str(len(failures)))
if len(failures) != 5:
    print("ERROR: expected all five scoped defects to be present before production edits")
sys.exit(1 if failures else 0)
PY
