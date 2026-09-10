using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Text;

// Native microsteps are losslessly streamed. A chunk boundary is not a game
// terminal, and neither a missing sample nor an unobserved frame is invented.
public sealed class CanonicalNativeTrace
{
    public const int ChunkSamples = 250;
    public const int MaximumChunkBytes = 16 * 1024 * 1024;
    public const long MaximumShotSteps = 30000;
    public const int ObservationStride = 50;
    public const float FixedDeltaSeconds = 0.0004f;
    private readonly string root;
    private readonly string captureId;
    private readonly long firstStep;
    private readonly List<string> chunks = new List<string>();
    private long sampleCount;
    public bool Closed { get; private set; }

    public CanonicalNativeTrace(string root, string captureId, long firstStep)
    {
        this.root = root;
        this.captureId = captureId;
        this.firstStep = firstStep;
        Directory.CreateDirectory(Path.Combine(root, "native"));
    }

    public void Observe(PhysicsCaptureV2FixedStepRecorder recorder)
    {
        if (Closed) return;
        // Retain the latest step until the next step: late callbacks can still
        // merge contact evidence into that sample in the existing recorder.
        if (recorder.NativeBufferedSampleCount > ChunkSamples)
            Flush(recorder, false);
    }

    private void Flush(PhysicsCaptureV2FixedStepRecorder recorder, bool final)
    {
        PhysicsCaptureV2EngineSnapshot snapshot = recorder.NativeChunkSnapshot(final);
        if (snapshot == null) return;
        string json = PhysicsCaptureV2EngineProtocol.BuildNativeChunkJson(snapshot);
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        if (bytes.Length > MaximumChunkBytes)
            throw new InvalidOperationException("native trace chunk exceeds 16MiB; capture must fail");
        string relative = "native/chunk-" + (chunks.Count + 1).ToString("D6") + ".json.gz";
        string path = Path.Combine(root, relative);
        using (FileStream file = new FileStream(path + ".partial", FileMode.CreateNew))
        using (GZipStream gzip = new GZipStream(file, CompressionMode.Compress))
            gzip.Write(bytes, 0, bytes.Length);
        File.Move(path + ".partial", path);
        long last = snapshot.FixedStepSamples[snapshot.FixedStepSamples.Count - 1].FixedStep;
        chunks.Add("{\"path\":" + Quote(relative)
            + ",\"first_fixed_step\":" + snapshot.FixedStepSamples[0].FixedStep
            + ",\"last_fixed_step\":" + last
            + ",\"sample_count\":" + snapshot.FixedStepSamples.Count
            + ",\"event_count\":" + snapshot.Events.Count
            + ",\"uncompressed_bytes\":" + bytes.Length
            + ",\"compressed_bytes\":" + new FileInfo(path).Length + "}");
        sampleCount += snapshot.FixedStepSamples.Count;
        recorder.CommitNativeChunk(snapshot.FixedStepSamples.Count, last);
    }

    public void Finish(PhysicsCaptureV2FixedStepRecorder recorder, string failure = null)
    {
        if (Closed) return;
        PhysicsCaptureV2EngineSnapshot metadata = recorder.NativeMetadataSnapshot();
        if (failure == null && !recorder.IsFinalized)
            throw new InvalidOperationException("native capture cannot complete without a real terminal record");
        while (recorder.NativeBufferedSampleCount > ChunkSamples) Flush(recorder, false);
        Flush(recorder, true);
        StringBuilder json = new StringBuilder("{\"schema\":\"canonical_native_trace_v1\",\"capture_id\":");
        json.Append(Quote(captureId)).Append(",\"shot_id\":").Append(Quote(metadata.ShotId))
            .Append(",\"status\":").Append(Quote(failure == null ? "complete" : "failed"))
            .Append(",\"failure\":").Append(failure == null ? "null" : Quote(failure))
            .Append(",\"fixed_delta_seconds\":0.0004,\"observation_stride\":50")
            .Append(",\"first_fixed_step\":").Append(firstStep)
            .Append(",\"last_fixed_step\":").Append(recorder.LastFixedStep)
            .Append(",\"sample_count\":").Append(sampleCount)
            .Append(",\"complete_every_native_step\":").Append(sampleCount == recorder.LastFixedStep - firstStep + 1 ? "true" : "false")
            .Append(",\"engine_seed\":").Append(Quote(Environment.GetEnvironmentVariable("NOVPHY_ENVIRONMENT_SEED")))
            .Append(",\"chunks\":[").Append(string.Join(",", chunks.ToArray())).Append("]")
            .Append(",\"frame_records\":[");
        for (int i = 0; i < metadata.FrameRecords.Count; i++)
        {
            if (i > 0) json.Append(',');
            json.Append("{\"fixed_step\":").Append(metadata.FrameRecords[i].FixedStep)
                .Append(",\"forced_terminal\":").Append(metadata.FrameRecords[i].ForcedTerminal ? "true" : "false")
                .Append('}');
        }
        json.Append("],\"terminal_evidence\":");
        if (failure == null)
            json.Append("{\"reason\":").Append(Quote(metadata.TerminalReason))
                .Append(",\"fixed_step\":").Append(metadata.TerminalFixedStep)
                .Append(",\"event_id\":").Append(Quote(metadata.TerminalEventId)).Append('}');
        else
            json.Append("null");
        json.Append('}');
        string manifest = Path.Combine(root, "native-manifest.json");
        File.WriteAllText(manifest + ".partial", json.ToString(), new UTF8Encoding(false));
        File.Move(manifest + ".partial", manifest);
        Closed = true;
    }

    private static string Quote(string value)
    {
        if (value == null) return "null";
        return "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"")
            .Replace("\r", "\\r").Replace("\n", "\\n") + "\"";
    }
}
