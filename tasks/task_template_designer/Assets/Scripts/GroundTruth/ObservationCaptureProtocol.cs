using System;
using System.Globalization;
using System.Text;
using UnityEngine;

public enum ObservationCaptureFailureCode
{
    CanonicalObservationMissing = 1,
    SynchronizedStateMissing = 2,
    CameraStateMissing = 3,
    ViewportInvalid = 4,
    NonFiniteMetadata = 5,
    EnvelopeLimitExceeded = 6
}

public static class ObservationCaptureProtocol
{
    public const byte RequestCode = 72;
    public const byte Version = 1;
    public const int MaxEnvelopeBytes = 64 * 1024 * 1024;
    private static readonly byte[] Magic = {
        (byte)'S', (byte)'B', (byte)'O', (byte)'1'
    };

    public static byte[] EncodeCanonicalRgbPng(Texture2D source)
    {
        if (source == null) return null;
        Texture2D rgb = new Texture2D(
            source.width, source.height, TextureFormat.RGB24, false);
        try
        {
            rgb.SetPixels32(source.GetPixels32());
            rgb.Apply(false, false);
            return rgb.EncodeToPNG();
        }
        finally
        {
            if (Application.isPlaying) UnityEngine.Object.Destroy(rgb);
            else UnityEngine.Object.DestroyImmediate(rgb);
        }
    }

    public static byte[] BuildCaptureEnvelope(byte[] canonicalPng,
        PhysicalSceneSnapshot snapshot, Camera camera, string captureId,
        long sequence, int width, int height)
    {
        if (canonicalPng == null || canonicalPng.Length < 8
            || canonicalPng[0] != 0x89 || canonicalPng[1] != (byte)'P'
            || canonicalPng[2] != (byte)'N' || canonicalPng[3] != (byte)'G')
            return BuildFailureEnvelope(
                ObservationCaptureFailureCode.CanonicalObservationMissing,
                "canonical observation is missing or is not PNG");
        if (snapshot == null)
            return BuildFailureEnvelope(
                ObservationCaptureFailureCode.SynchronizedStateMissing,
                "synchronized physical state is missing");
        if (camera == null)
            return BuildFailureEnvelope(
                ObservationCaptureFailureCode.CameraStateMissing,
                "main camera state is missing");
        if (width <= 0 || height <= 0)
            return BuildFailureEnvelope(
                ObservationCaptureFailureCode.ViewportInvalid,
                "observation viewport is invalid");
        if (string.IsNullOrEmpty(captureId) || sequence <= 0
            || !CameraValuesAreFinite(camera))
            return BuildFailureEnvelope(
                ObservationCaptureFailureCode.NonFiniteMetadata,
                "observation identity or camera metadata is invalid");

        string metadata = BuildMetadata(snapshot, camera, captureId, sequence,
            width, height, "synchronized_observation_endpoint");
        byte[] metadataBytes = Encoding.UTF8.GetBytes(metadata);
        long payloadLength = 8L + canonicalPng.Length + metadataBytes.Length;
        if (payloadLength > MaxEnvelopeBytes - 16L)
            return BuildFailureEnvelope(
                ObservationCaptureFailureCode.EnvelopeLimitExceeded,
                "observation capture envelope exceeds bounds");
        byte[] payload = new byte[(int)payloadLength];
        WriteUInt32(payload, 0, (uint)canonicalPng.Length);
        WriteUInt32(payload, 4, (uint)metadataBytes.Length);
        Buffer.BlockCopy(canonicalPng, 0, payload, 8, canonicalPng.Length);
        Buffer.BlockCopy(metadataBytes, 0, payload, 8 + canonicalPng.Length,
            metadataBytes.Length);
        return BuildEnvelope(0, 0, payload);
    }

    public static byte[] BuildFailureEnvelope(ObservationCaptureFailureCode code,
        string message)
    {
        byte[] text = Encoding.UTF8.GetBytes(message ?? "observation capture failed");
        byte[] payload = new byte[4 + text.Length];
        WriteUInt32(payload, 0, (uint)text.Length);
        Buffer.BlockCopy(text, 0, payload, 4, text.Length);
        return BuildEnvelope(1, (int)code, payload);
    }

    public static string BuildMetadata(PhysicalSceneSnapshot snapshot,
        Camera camera, string captureId, long sequence, int width, int height,
        string source)
    {
        string sourceFrameIdentity = string.Format(CultureInfo.InvariantCulture,
            "source-frame-v1:{0}:{1}:{2}:{3}", captureId, sequence,
            snapshot.RenderFrame, snapshot.FixedStep);
        Rect pixelRect = camera.pixelRect;
        if (pixelRect.width <= 0f || pixelRect.height <= 0f)
            pixelRect = new Rect(0f, 0f, width, height);
        float aspect = camera.aspect;
        if (!IsFinite(aspect) || aspect <= 0f)
            aspect = (float)width / height;
        Matrix4x4 worldToCamera = camera.worldToCameraMatrix;
        Matrix4x4 cameraToClip = camera.projectionMatrix;
        StringBuilder json = new StringBuilder(
            "{\"schema_version\":\"observation_capture_engine_v1\",\"capture_id\":");
        AppendString(json, captureId);
        json.Append(",\"sequence\":").Append(sequence.ToString(CultureInfo.InvariantCulture));
        json.Append(",\"source_frame_identity\":"); AppendString(json, sourceFrameIdentity);
        json.Append(",\"render_frame\":").Append(snapshot.RenderFrame.ToString(CultureInfo.InvariantCulture));
        json.Append(",\"render_time_seconds\":"); AppendFloat(json, snapshot.RenderTime);
        json.Append(",\"fixed_step\":").Append(snapshot.FixedStep.ToString(CultureInfo.InvariantCulture));
        json.Append(",\"fixed_time_seconds\":"); AppendFloat(json, snapshot.FixedTime);
        json.Append(",\"source\":"); AppendString(json, source);
        json.Append(",\"camera\":{");
        json.Append("\"camera_identity\":\"unity-main-camera\",\"projection_kind\":");
        AppendString(json, camera.orthographic ? "orthographic" : "perspective");
        json.Append(",\"position_world\":"); AppendVector3(json, camera.transform.position);
        json.Append(",\"rotation_xyzw\":"); AppendQuaternion(json, camera.transform.rotation);
        json.Append(",\"orthographic_size_world_units\":");
        if (camera.orthographic) AppendFloat(json, camera.orthographicSize); else json.Append("null");
        json.Append(",\"vertical_field_of_view_degrees\":");
        if (camera.orthographic) json.Append("null"); else AppendFloat(json, camera.fieldOfView);
        json.Append(",\"near_clip_world_units\":"); AppendFloat(json, camera.nearClipPlane);
        json.Append(",\"far_clip_world_units\":"); AppendFloat(json, camera.farClipPlane);
        json.Append(",\"aspect_ratio\":"); AppendFloat(json, aspect);
        json.Append(",\"world_to_camera_matrix\":"); AppendMatrix(json, worldToCamera);
        json.Append(",\"camera_to_clip_matrix\":"); AppendMatrix(json, cameraToClip);
        json.Append("},\"viewport\":{\"width_pixels\":").Append(width);
        json.Append(",\"height_pixels\":").Append(height);
        json.Append(",\"camera_pixel_rect\":["); AppendFloat(json, pixelRect.x);
        json.Append(','); AppendFloat(json, pixelRect.y); json.Append(',');
        AppendFloat(json, pixelRect.width); json.Append(','); AppendFloat(json, pixelRect.height);
        json.Append("],\"screen_width_pixels\":").Append(width);
        json.Append(",\"screen_height_pixels\":").Append(height);
        json.Append(",\"pixel_origin\":\"bottom_left\"},\"coordinates\":{");
        json.Append("\"world_space\":\"unity_world_2d\",\"world_units\":\"unity_unit\",");
        json.Append("\"observation_space\":\"rgb_pixel\",\"observation_units\":\"pixel\",");
        json.Append("\"observation_origin\":\"top_left\",\"observation_x_axis\":\"right\",");
        json.Append("\"observation_y_axis\":\"down\",\"channel_order\":\"RGB\",");
        json.Append("\"sample_type\":\"uint8\",\"color_space\":\"sRGB\"},");
        json.Append("\"world_to_observation_transform\":{");
        json.Append("\"method\":\"unity_world_to_clip_to_top_left_pixel_v1\",");
        json.Append("\"world_to_camera_matrix\":"); AppendMatrix(json, worldToCamera);
        json.Append(",\"camera_to_clip_matrix\":"); AppendMatrix(json, cameraToClip);
        json.Append(",\"clip_to_ndc\":\"homogeneous_divide\",");
        json.Append("\"ndc_to_observation_matrix\":[");
        AppendFloat(json, pixelRect.width / 2f); json.Append(",0,");
        AppendFloat(json, pixelRect.x + pixelRect.width / 2f);
        json.Append(",0,"); AppendFloat(json, -pixelRect.height / 2f); json.Append(',');
        AppendFloat(json, height - pixelRect.y - pixelRect.height / 2f);
        json.Append(",0,0,1]}}");
        return json.ToString();
    }

    private static bool CameraValuesAreFinite(Camera camera)
    {
        if (!IsFinite(camera.nearClipPlane) || !IsFinite(camera.farClipPlane)
            || camera.nearClipPlane < 0f || camera.farClipPlane <= camera.nearClipPlane)
            return false;
        if (camera.orthographic)
        {
            if (!IsFinite(camera.orthographicSize) || camera.orthographicSize <= 0f)
                return false;
        }
        else if (!IsFinite(camera.fieldOfView) || camera.fieldOfView <= 0f
            || camera.fieldOfView >= 180f)
            return false;
        return MatrixIsFinite(camera.worldToCameraMatrix)
            && MatrixIsFinite(camera.projectionMatrix)
            && IsFinite(camera.transform.position.x)
            && IsFinite(camera.transform.position.y)
            && IsFinite(camera.transform.position.z)
            && IsFinite(camera.transform.rotation.x)
            && IsFinite(camera.transform.rotation.y)
            && IsFinite(camera.transform.rotation.z)
            && IsFinite(camera.transform.rotation.w);
    }

    private static bool MatrixIsFinite(Matrix4x4 value)
    {
        for (int row = 0; row < 4; row++)
            for (int column = 0; column < 4; column++)
                if (!IsFinite(value[row, column])) return false;
        return true;
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private static void AppendVector3(StringBuilder json, Vector3 value)
    {
        json.Append('['); AppendFloat(json, value.x); json.Append(',');
        AppendFloat(json, value.y); json.Append(','); AppendFloat(json, value.z);
        json.Append(']');
    }

    private static void AppendQuaternion(StringBuilder json, Quaternion value)
    {
        json.Append('['); AppendFloat(json, value.x); json.Append(',');
        AppendFloat(json, value.y); json.Append(','); AppendFloat(json, value.z);
        json.Append(','); AppendFloat(json, value.w); json.Append(']');
    }

    private static void AppendMatrix(StringBuilder json, Matrix4x4 value)
    {
        json.Append('[');
        for (int row = 0; row < 4; row++)
            for (int column = 0; column < 4; column++)
            {
                if (row != 0 || column != 0) json.Append(',');
                AppendFloat(json, value[row, column]);
            }
        json.Append(']');
    }

    private static void AppendFloat(StringBuilder json, float value)
    {
        json.Append(value.ToString("R", CultureInfo.InvariantCulture));
    }

    private static void AppendString(StringBuilder json, string value)
    {
        json.Append('\"').Append(value.Replace("\\", "\\\\").Replace("\"", "\\\""))
            .Append('\"');
    }

    private static byte[] BuildEnvelope(byte flags, int failureCode, byte[] payload)
    {
        int bodyLength = 12 + payload.Length;
        byte[] result = new byte[4 + bodyLength];
        WriteUInt32(result, 0, (uint)bodyLength);
        Buffer.BlockCopy(Magic, 0, result, 4, Magic.Length);
        result[8] = Version;
        result[9] = flags;
        WriteUInt16(result, 10, (ushort)failureCode);
        WriteUInt32(result, 12, (uint)payload.Length);
        Buffer.BlockCopy(payload, 0, result, 16, payload.Length);
        return result;
    }

    private static void WriteUInt16(byte[] buffer, int offset, ushort value)
    {
        buffer[offset] = (byte)(value >> 8);
        buffer[offset + 1] = (byte)value;
    }

    private static void WriteUInt32(byte[] buffer, int offset, uint value)
    {
        buffer[offset] = (byte)(value >> 24);
        buffer[offset + 1] = (byte)(value >> 16);
        buffer[offset + 2] = (byte)(value >> 8);
        buffer[offset + 3] = (byte)value;
    }
}
