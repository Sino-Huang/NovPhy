#!/usr/bin/env bash
set -uo pipefail

repo_root="$(GIT_MASTER=1 git rev-parse --show-toplevel)"
python3 - "$repo_root" <<'PY'
import re
import subprocess
import sys
from pathlib import Path


root = Path(sys.argv[1])
source_root = root / "tasks/task_template_designer/Assets/Scripts"
test_root = root / "tasks/task_template_designer/Assets/Tests/Editor"
protocol_path = source_root / "GroundTruth/PhysicsCaptureProtocol.cs"
recorder_path = source_root / "GroundTruth/PhysicsShotRecorder.cs"
runtime_path = source_root / "GroundTruth/PhysicalSnapshotRuntime.cs"
bird_path = source_root / "GameWorld/Characters/Birds/ABBird.cs"
dispatch_path = source_root / "AIBirdsConnection.cs"
version_path = root / "tasks/task_template_designer/ProjectSettings/ProjectVersion.txt"
protocol_tests_path = test_root / "PhysicsCaptureProtocolTests.cs"
recorder_tests_path = test_root / "PhysicsShotRecorderTests.cs"
bird_tests_path = test_root / "ABBirdLaunchTests.cs"


def read(path):
    return path.read_text(encoding="utf-8-sig")


def method_body(source, signature):
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


def check(condition, label):
    if condition:
        print("PASS " + label)
    else:
        failures.append(label)
        print("FAIL " + label)


failures = []
protocol = read(protocol_path)
recorder = read(recorder_path)
runtime = read(runtime_path)
bird = read(bird_path)
dispatch = read(dispatch_path)
protocol_tests = read(protocol_tests_path)
recorder_tests = read(recorder_tests_path)
bird_tests = read(bird_tests_path)

serve = method_body(protocol, "private IEnumerator Serve")
transmit = method_body(protocol, "private static IEnumerator TransmitResponse")
queue_end_write = method_body(protocol, "private static void QueueEndWrite")
append_string = method_body(protocol, "private static void AppendString")
record_contacts = method_body(recorder, "public void RecordContacts(long fixedStep, float fixedTime, PhysicalContactInput[] contacts)")
record_unity_contacts = method_body(recorder, "public void RecordUnityContacts")
record_event = method_body(recorder, "public void RecordEvent")
add_one_shot = method_body(recorder, "private void AddOneShotEvent")
add_terminal = method_body(recorder, "private void AddTerminalEvent")
add_event = method_body(recorder, "private void AddEvent")
fail = method_body(recorder, "private void Fail(")
finalize = method_body(recorder, "public PhysicalCaptureResult FinalizeShot")
launch = method_body(bird, "public void LaunchBird")
assign_launch = method_body(bird, "protected void AssignLaunchVelocityAndRecord")

response_calls = list(re.finditer(r"TransmitResponse\s*\(\s*client\s*,\s*[^,]+,\s*ResponseWriteTimeoutMilliseconds\s*\)", serve))
check(
    "stream.Write(" not in serve and len(response_calls) == 2,
    "transport Serve has no synchronous NetworkStream.Write and both response paths use TransmitResponse with fixed ResponseWriteTimeoutMilliseconds",
)
begin_write = transmit.index("BeginWrite")
bound_check = transmit.index("response == null || response.Length > PhysicsCaptureV1Protocol.MaxEnvelopeBytes")
check(
    bound_check < begin_write and "client.Close();" in transmit[bound_check:begin_write],
    "transport response bound rejects null or over-MaxEnvelopeBytes before BeginWrite",
)
check(
    "BeginWrite(response, 0, response.Length, QueueEndWrite, completion)" in transmit
    and "Stopwatch deadline = Stopwatch.StartNew();" in transmit
    and "!completion.EndWriteFinished" in transmit
    and "if (!completion.EndWriteFinished)" in transmit
    and "client.Close();" in transmit[transmit.index("if (!completion.EndWriteFinished)"):],
    "transport BeginWrite is completion-callback based and deadline closes an incomplete client",
)
check(
    "ThreadPool.QueueUserWorkItem" in queue_end_write
    and "completion.Stream.EndWrite(write);" in queue_end_write
    and "catch (IOException)" in queue_end_write
    and "catch (ObjectDisposedException)" in queue_end_write
    and "completion.EndWriteFinished = true;" in queue_end_write,
    "transport callback queues EndWrite off coroutine and catches close-induced exceptions",
)

controls = [r'''case '"':''', r'''case '\\':''', r'''case '\b':''', r'''case '\f':''', r'''case '\n':''', r'''case '\r':''', r'''case '\t':''']
check(
    all(control in append_string for control in controls)
    and "if (c < 32)" in append_string
    and r'''json.Append("\\u")''' in append_string
    and ".ToString(\"x4\", CultureInfo.InvariantCulture)" in append_string,
    "JSON AppendString handles quote/backslash/b/f/n/r/t and emits unicode escapes for default c<32",
)

check(
    "shotStartFixedTime" in record_contacts
    and "if (!hasShotStartFixedTime)" in record_contacts
    and "shotStartFixedTime = fixedTime;" in record_contacts
    and "Mathf.Max(0f, fixedTime - shotStartFixedTime)" in record_contacts
    and "elapsedFixedTime > limits.TimeoutSeconds" in record_contacts
    and "fixedTime > limits.TimeoutSeconds" not in record_contacts,
    "timeout uses stored shotStartFixedTime, nonnegative elapsed difference, and elapsed comparison only",
)

record_unity_alloc = record_unity_contacts.index("List<PhysicalContactInput> inputs")
check(
    record_unity_contacts.index("if (Failure != null || finalized)") < record_unity_alloc
    and "foreach" in record_unity_contacts[record_unity_alloc:],
    "RecordUnityContacts guards finalized state before allocation and collider traversal",
)
check(
    "if (finalized)" in add_one_shot
    and "Failure != null || finalized" in add_terminal
    and "if (finalized)" in add_event
    and "!finalized && Failure == null" in fail
    and "if (Failure != null || finalized)" in record_event,
    "event, terminal, failure, and public mutation paths retain finalized guards",
)
check(
    finalize.index("TruncatedFinalization") < finalize.index("finalized = true"),
    "truncated finalization failure is recorded before finalized assignment",
)

assignment = assign_launch.index("_rigidBody.velocity = launchVelocity")
callback = assign_launch.index("RecordLaunchCallback")
recorded_field = assign_launch.index("_rigidBody.velocity);", callback)
check(
    assignment < callback < recorded_field,
    "ABBird assignment seam sets _rigidBody.velocity then records that exact field",
)
launch_velocity = launch.index("Vector2 launchVelocity")
check(
    launch_velocity < launch.index("AssignLaunchVelocityAndRecord(launchVelocity)"),
    "LaunchBird calls the velocity assignment seam after computing launch velocity",
)

check(
    version_path.read_text(encoding="utf-8") == "m_EditorVersion: 2019.4.41f2\nm_EditorVersionWithRevision: 2019.4.41f2 (6b23d448b533)\n",
    "Unity ProjectVersion.txt is exactly 2019.4.41f2 revision 6b23d448b533",
)
dispatch_diff = subprocess.run(
    ["git", "diff", "--quiet", "--", "tasks/task_template_designer/Assets/Scripts/AIBirdsConnection.cs"],
    cwd=root,
    env={**dict(__import__("os").environ), "GIT_MASTER": "1"},
).returncode
check(
    dispatch_diff == 0
    and "shootAndRecordGroundTruth" in dispatch
    and "GroundTruthWithoutScreenshot" in dispatch
    and "NoisyGroundTruthWithoutScreenshot" in dispatch,
    "request 38/62 dispatch source path is unchanged and legacy handlers remain present",
)
check(
    "RequestCode = 70" in protocol
    and "Version = 1" in protocol
    and "MaxEnvelopeBytes = 64 * 1024 * 1024" in protocol
    and "private static readonly byte[] Magic = { (byte)'S', (byte)'B', (byte)'P', (byte)'V' };" in protocol
    and "int bodyLength = 12 + payload.Length" in protocol
    and "WriteUInt32(result, 0, (uint)bodyLength)" in protocol
    and "result[8] = Version" in protocol
    and "WriteUInt32(result, 12, (uint)payload.Length)" in protocol,
    "request-70 envelope Magic/Version/layout constants remain expected",
)

transport_test = method_body(protocol_tests, "public void Request70ResponseTransmissionYieldsAndExpiresWhenClientStopsReading")
json_test = method_body(protocol_tests, "public void Request70JsonEscapesEveryControlCharacterAndRoundTrips")
silent_test = method_body(protocol_tests, "public void SilentLoopbackClientDoesNotBlockTheRequestCoroutineBeforeItsFirstYield")
capacity_test = method_body(protocol_tests, "public void SilentInFlightClientsHoldCapacityUntilTimeoutThenReleaseIt")
elapsed_test = method_body(recorder_tests, "public void TimeoutUsesElapsedFixedTimeFromFirstAuthoritativeSample")
mutation_test = method_body(recorder_tests, "public void FinalizedRecorderRejectsEveryPublicMutationPath")
truncated_test = method_body(recorder_tests, "public void TruncatedFinalizationSetsFailureBeforeBecomingTerminal")
registry_test = method_body(recorder_tests, "private static void AssertFinalizedUnityContactsReturnBeforeRegistryMutation")
bird_test = method_body(bird_tests, "public void BirdLaunchRecordsTheExactVelocityAfterItIsAssigned")
check(
    "GetMethod(\n                \"TransmitResponse\"" in transport_test
    and "new byte[4 * 1024 * 1024]" in transport_test
    and "transmission.MoveNext()" in transport_test
    and "Assert.IsFalse(running" in transport_test
    and "IsLocallyClosed(server)" in transport_test
    and "SilentLoopbackClient" in protocol_tests
    and "SilentInFlightClientsHoldCapacityUntilTimeoutThenReleaseIt" in protocol_tests,
    "focused tests assert asynchronous transport deadline, nonreading client, silent request, and capacity cleanup",
)
check(
    "new char[32]" in protocol_tests
    and "JSONNode.Parse(json)" in protocol_tests
    and "Assert.AreEqual(value, parsed" in protocol_tests
    and "Assert.IsFalse(json.Contains(controls[i].ToString())," in protocol_tests
    and "StringAssert.Contains(expected, json" in protocol_tests,
    "focused test behaviorally round-trips all 32 controls and rejects raw control bytes",
)
check(
    "50f" in elapsed_test
    and "49f" in elapsed_test
    and "50.1f" in elapsed_test
    and "50.101f" in elapsed_test
    and "Assert.IsNull(recorder.Failure" in elapsed_test
    and "CaptureTimeout" in elapsed_test,
    "focused test asserts late-start elapsed timeout with nonnegative clock behavior and boundary",
)
check(
    mutation_test.count("AssertFinalizedMutationIsNoOp") >= 20
    and "AssertFinalizedUnityContactsReturnBeforeRegistryMutation();" in mutation_test
    and "lifetimes.Count" in registry_test
    and "Assert.AreEqual(0, lifetimes.Count" in registry_test
    and "Assert.IsFalse(result.IsValid)" in truncated_test
    and "TruncatedFinalization" in truncated_test,
    "focused tests cover all mutation paths, finality, registry side-effect prevention, and truncated failure ordering",
)
check(
    "staleVelocity = new Vector2(-3f, 4f)" in bird_test
    and "launchVelocity = new Vector2(12.5f, 7.25f)" in bird_test
    and "Payload.LaunchVelocity.Value" in bird_test
    and "Assert.AreNotEqual" in bird_test,
    "focused ABBird test asserts exact assigned velocity is the recorded payload",
)

print("GREEN_CHECK_COUNT=" + str(17 - len(failures)))
print("GREEN_FAILURE_COUNT=" + str(len(failures)))
sys.exit(1 if failures else 0)
PY
