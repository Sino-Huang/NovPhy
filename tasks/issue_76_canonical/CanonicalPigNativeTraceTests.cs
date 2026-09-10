using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using NUnit.Framework;
using UnityEngine;

public class CanonicalPigNativeTraceTests
{
    [Serializable] private class Descriptor { public string path; }
    [Serializable] private class Frame { public int fixed_step; }
    [Serializable] private class Event { public string event_id; }
    [Serializable] private class Chunk { public Frame[] fixed_step_samples; public Event[] events; }
    [Serializable] private class Manifest
    {
        public string status;
        public int sample_count;
        public Descriptor[] chunks;
        public Frame[] frame_records;
        public Frame terminal_evidence;
    }
    [Test]
    public void NativeTimeWindowIncludesDragAndUsesTwentyMillisecondObservations()
    {
        Assert.AreEqual(0.0004f, Time.fixedDeltaTime);
        Assert.Less(600 * Time.fixedDeltaTime, 1f);
        Assert.AreEqual(12f, CanonicalNativeTrace.MaximumShotSteps * Time.fixedDeltaTime, 0.00001f);
        Assert.AreEqual(0.02f, CanonicalNativeTrace.ObservationStride * Time.fixedDeltaTime, 0.000001f);
    }

    [Test]
    public void ChunkDrainPreservesEveryStepLateEventsAndRealTerminalWithoutFakeChunkTerminals()
    {
        string root = Path.Combine(Path.GetTempPath(), "novphy-native-fixture-" + Guid.NewGuid().ToString("N"));
        string previous = Environment.GetEnvironmentVariable("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE");
        GameObject host = new GameObject("native-recorder-fixture");
        GameObject entity = new GameObject("native-entity-fixture");
        try
        {
            Environment.SetEnvironmentVariable("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE", "50");
            entity.AddComponent<BoxCollider2D>();
            entity.AddComponent<Rigidbody2D>();
            ScenarioObjectIdentity.Assign(entity, "block:0000");
            var recorder = host.AddComponent<PhysicsCaptureV2FixedStepRecorder>();
            GameObject[] objects = { entity };
            recorder.BeginPreIntervention(0, objects);
            var trace = new CanonicalNativeTrace(root, recorder.CaptureId, 0);
            for (int step = 1; step <= 501; step++)
            {
                recorder.RecordFixedStep(step, objects);
                Assert.LessOrEqual(recorder.NativeBufferedSampleCount, 251);
                if (step == 249) recorder.RecordMacroEvent(step, "stable_exited", new string[0], "{}");
                trace.Observe(recorder);
                // Callback after a flush still belongs to the retained latest step.
                if (step == 250) recorder.RecordMacroEvent(step, "stable_exited", new string[0], "{}");
            }
            recorder.FinalizeTerminal(501, "stable_entered");
            trace.Finish(recorder);
            var manifest = JsonUtility.FromJson<Manifest>(File.ReadAllText(Path.Combine(root, "native-manifest.json")));
            Assert.AreEqual("complete", manifest.status);
            Assert.AreEqual(502, manifest.sample_count);
            Assert.AreEqual(3, manifest.chunks.Length);
            Assert.AreEqual(12, manifest.frame_records.Length);
            Assert.AreEqual(501, manifest.terminal_evidence.fixed_step);
            int expectedStep = 0;
            HashSet<string> eventIds = new HashSet<string>();
            foreach (Descriptor descriptor in manifest.chunks)
            {
                string content;
                using (var file = File.OpenRead(Path.Combine(root, descriptor.path)))
                using (var gzip = new GZipStream(file, CompressionMode.Decompress))
                using (var reader = new StreamReader(gzip)) content = reader.ReadToEnd();
                Assert.IsFalse(content.Contains("\"terminal_evidence\""));
                var chunk = JsonUtility.FromJson<Chunk>(content);
                Assert.LessOrEqual(chunk.fixed_step_samples.Length, 250);
                foreach (Frame sample in chunk.fixed_step_samples)
                    Assert.AreEqual(expectedStep++, sample.fixed_step);
                foreach (Event item in chunk.events)
                    Assert.IsTrue(eventIds.Add(item.event_id), "event ordinal repeated after buffer drain");
            }
            Assert.AreEqual(502, expectedStep);
            Assert.AreEqual(3, eventIds.Count);
            Assert.AreEqual(0, recorder.NativeBufferedSampleCount);
            File.WriteAllText(Path.Combine(root, "unit-fixture.txt"), "Synthetic recorder unit fixture, not a research-data episode.");
            Debug.Log("[native-unit-fixture] " + root);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(entity);
            UnityEngine.Object.DestroyImmediate(host);
            Environment.SetEnvironmentVariable("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE", previous);
            // Retain the successful synthetic fixture for the Python-reader
            // cross-language check. Its unique temp path is printed above.
        }
    }

    [Test]
    public void FailedNativeCaptureDoesNotInventGameplayTerminalEvidence()
    {
        string root = Path.Combine(Path.GetTempPath(), "novphy-native-failure-" + Guid.NewGuid().ToString("N"));
        string previous = Environment.GetEnvironmentVariable("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE");
        GameObject host = new GameObject("native-failed-recorder");
        GameObject entity = new GameObject("native-failed-entity");
        try
        {
            Environment.SetEnvironmentVariable("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE", "50");
            entity.AddComponent<BoxCollider2D>();
            ScenarioObjectIdentity.Assign(entity, "block:0000");
            var recorder = host.AddComponent<PhysicsCaptureV2FixedStepRecorder>();
            recorder.BeginPreIntervention(0, new[] { entity });
            var trace = new CanonicalNativeTrace(root, recorder.CaptureId, 0);
            trace.Finish(recorder, "native_time_window_limit");
            string text = File.ReadAllText(Path.Combine(root, "native-manifest.json"));
            Assert.IsTrue(text.Contains("\"terminal_evidence\":null"));
            Assert.AreEqual("failed", JsonUtility.FromJson<Manifest>(text).status);
            Assert.IsFalse(recorder.IsFinalized);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(entity);
            UnityEngine.Object.DestroyImmediate(host);
            Environment.SetEnvironmentVariable("NOVPHY_PHYSICS_CAPTURE_V2_STRIDE", previous);
            if (Directory.Exists(root)) Directory.Delete(root, true);
        }
    }
}
