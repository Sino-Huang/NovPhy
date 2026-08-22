using System;
using NUnit.Framework;
using SimpleJSON;
using UnityEngine;

public sealed class ObservationCaptureProtocolTests
{
    [Test]
    public void Request72BindsCanonicalPngToCameraViewportAndSourceFrame()
    {
        GameObject cameraObject = new GameObject("Main Camera");
        Camera camera = cameraObject.AddComponent<Camera>();
        camera.orthographic = true;
        camera.orthographicSize = 5f;
        camera.transform.position = new Vector3(1f, 2f, -10f);
        PhysicalSceneSnapshot snapshot = new PhysicalSceneSnapshot(
            30, 1.5f, 40, 0.8f, new PhysicalNodeSnapshot[0]);
        byte[] png = new byte[] {
            0x89, (byte)'P', (byte)'N', (byte)'G', 0x0d, 0x0a, 0x1a, 0x0a, 1
        };

        byte[] envelope = ObservationCaptureProtocol.BuildCaptureEnvelope(
            png, snapshot, camera, "capture-1", 2, 4, 3);

        Assert.AreEqual(72, ObservationCaptureProtocol.RequestCode);
        Assert.AreEqual(envelope.Length - 4, ReadUInt32(envelope, 0));
        CollectionAssert.AreEqual(
            new byte[] { (byte)'S', (byte)'B', (byte)'O', (byte)'1' },
            new ArraySegment<byte>(envelope, 4, 4));
        Assert.AreEqual(1, envelope[8]);
        Assert.AreEqual(0, envelope[9]);
        Assert.AreEqual(0, envelope[10]);
        Assert.AreEqual(0, envelope[11]);
        int payloadLength = ReadUInt32(envelope, 12);
        Assert.AreEqual(envelope.Length - 16, payloadLength);
        int pngLength = ReadUInt32(envelope, 16);
        int metadataLength = ReadUInt32(envelope, 20);
        Assert.AreEqual(png.Length, pngLength);
        string metadataText = System.Text.Encoding.UTF8.GetString(
            envelope, 24 + pngLength, metadataLength).Replace(":null", ":\"null\"");
        JSONNode metadata = JSONNode.Parse(metadataText);
        Assert.AreEqual("observation_capture_engine_v1", metadata["schema_version"].Value);
        Assert.AreEqual("synchronized_observation_endpoint", metadata["source"].Value);
        Assert.AreEqual("source-frame-v1:capture-1:2:30:40",
            metadata["source_frame_identity"].Value);
        Assert.AreEqual(4, metadata["viewport"]["width_pixels"].AsInt);
        Assert.AreEqual(3, metadata["viewport"]["height_pixels"].AsInt);
        Assert.AreEqual("unity_unit", metadata["coordinates"]["world_units"].Value);
        Assert.AreEqual(16, metadata["camera"]["world_to_camera_matrix"].Count);
        Assert.AreEqual(9,
            metadata["world_to_observation_transform"]["ndc_to_observation_matrix"].Count);
        UnityEngine.Object.DestroyImmediate(cameraObject);
    }

    [Test]
    public void Request72FailsClosedWhenCameraOrSynchronizedStateIsMissing()
    {
        byte[] png = new byte[] {
            0x89, (byte)'P', (byte)'N', (byte)'G', 0x0d, 0x0a, 0x1a, 0x0a, 1
        };
        GameObject cameraObject = new GameObject("Main Camera");
        Camera camera = cameraObject.AddComponent<Camera>();
        PhysicalSceneSnapshot snapshot = new PhysicalSceneSnapshot(
            1, 0.1f, 2, 0.04f, new PhysicalNodeSnapshot[0]);

        byte[] missingCamera = ObservationCaptureProtocol.BuildCaptureEnvelope(
            png, snapshot, null, "capture-1", 1, 4, 3);
        byte[] missingState = ObservationCaptureProtocol.BuildCaptureEnvelope(
            png, null, camera, "capture-1", 1, 4, 3);

        Assert.AreEqual(1, missingCamera[9]);
        Assert.AreEqual(1, missingState[9]);
        Assert.AreNotEqual(0, ReadUInt16(missingCamera, 10));
        Assert.AreNotEqual(0, ReadUInt16(missingState, 10));
        UnityEngine.Object.DestroyImmediate(cameraObject);
    }

    private static int ReadUInt16(byte[] buffer, int offset)
    {
        return (buffer[offset] << 8) | buffer[offset + 1];
    }

    private static int ReadUInt32(byte[] buffer, int offset)
    {
        return (buffer[offset] << 24) | (buffer[offset + 1] << 16)
            | (buffer[offset + 2] << 8) | buffer[offset + 3];
    }
}
