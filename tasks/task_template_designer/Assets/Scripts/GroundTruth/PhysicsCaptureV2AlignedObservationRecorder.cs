using System;
using System.Globalization;
using System.IO;
using UnityEngine;

public sealed class PhysicsCaptureV2AlignedObservationRecorder
{
    public const string RootEnvironmentVariable =
        "NOVPHY_ALIGNED_OBSERVATION_CAPTURE_ROOT";

    private readonly string root;
    private readonly string captureId;
    private long sequence;

    private PhysicsCaptureV2AlignedObservationRecorder(string root, string captureId)
    {
        this.root = root;
        this.captureId = captureId;
        Directory.CreateDirectory(root);
    }

    public static PhysicsCaptureV2AlignedObservationRecorder Create(string captureId)
    {
        string root = Environment.GetEnvironmentVariable(
            RootEnvironmentVariable, EnvironmentVariableTarget.Process);
        return string.IsNullOrEmpty(root)
            ? null
            : new PhysicsCaptureV2AlignedObservationRecorder(root, captureId);
    }

    public void Capture(PhysicalSnapshotRuntime runtime)
    {
        Camera camera = Camera.main;
        int width = Screen.width;
        int height = Screen.height;
        if (camera == null || width <= 0 || height <= 0)
            throw new InvalidOperationException(
                "aligned observation capture requires the main camera and viewport");

        RenderTexture priorTarget = camera.targetTexture;
        RenderTexture priorActive = RenderTexture.active;
        RenderTexture render = RenderTexture.GetTemporary(
            width, height, 24, RenderTextureFormat.ARGB32);
        Texture2D texture = new Texture2D(width, height, TextureFormat.RGB24, false);
        byte[] png;
        try
        {
            camera.targetTexture = render;
            RenderTexture.active = render;
            camera.Render();
            texture.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
            texture.Apply(false, false);
            png = texture.EncodeToPNG();
        }
        finally
        {
            camera.targetTexture = priorTarget;
            RenderTexture.active = priorActive;
            RenderTexture.ReleaseTemporary(render);
            UnityEngine.Object.Destroy(texture);
        }

        sequence++;
        PhysicalSceneSnapshot snapshot = runtime.CaptureCurrent(
            new SymbolicGameState(false), Time.frameCount, Time.time);
        string metadata = ObservationCaptureProtocol.BuildMetadata(
            snapshot, camera, captureId, sequence, width, height,
            "synchronized_fixed_step_camera_render");
        string stem = "frame_" + sequence.ToString("D6", CultureInfo.InvariantCulture);
        File.WriteAllBytes(Path.Combine(root, stem + ".png"), png);
        File.WriteAllText(Path.Combine(root, stem + ".json"), metadata);
    }
}
