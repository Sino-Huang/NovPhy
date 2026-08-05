using System.Globalization;
using System.Text;
using UnityEngine;

public static class PhysicalSnapshotJson
{
    public static string Serialize(PhysicalSceneSnapshot snapshot)
    {
        StringBuilder json = new StringBuilder(2048);
        json.Append("{\"schema_version\":\"").Append(PhysicalSceneSnapshot.SchemaVersion);
        json.Append("\",\"render_frame\":").Append(snapshot.RenderFrame.ToString(CultureInfo.InvariantCulture));
        json.Append(",\"render_time\":");
        AppendFloat(json, snapshot.RenderTime);
        json.Append(",\"fixed_step\":").Append(snapshot.FixedStep.ToString(CultureInfo.InvariantCulture));
        json.Append(",\"fixed_time\":");
        AppendFloat(json, snapshot.FixedTime);
        AppendCoordinates(json);
        json.Append(",\"nodes\":[");

        for (int index = 0; index < snapshot.Nodes.Length; index++)
        {
            if (index > 0)
            {
                json.Append(',');
            }
            AppendNode(json, snapshot.Nodes[index]);
        }

        return json.Append("]}").ToString();
    }

    private static void AppendCoordinates(StringBuilder json)
    {
        json.Append(",\"coordinates\":{");
        json.Append("\"world_space\":\"unity_world_2d\",");
        json.Append("\"world_origin\":\"scene_defined\",");
        json.Append("\"world_x_axis\":\"right\",");
        json.Append("\"world_y_axis\":\"up\",");
        json.Append("\"world_length_unit\":\"unity_unit\",");
        json.Append("\"screen_space\":\"rgb_pixel_2d\",");
        json.Append("\"screen_origin\":\"top_left\",");
        json.Append("\"screen_x_axis\":\"right\",");
        json.Append("\"screen_y_axis\":\"down\",");
        json.Append("\"screen_length_unit\":\"pixel\",");
        json.Append("\"time_unit\":\"second\",");
        json.Append("\"angle_unit\":\"degree\",");
        json.Append("\"mass_unit\":\"unity_mass_unit\",");
        json.Append("\"velocity_unit\":\"unity_unit/second\",");
        json.Append("\"angular_velocity_unit\":\"degree/second\",");
        json.Append("\"kinetic_energy_unit\":\"unity_mass_unit*unity_unit^2/second^2\",");
        json.Append("\"impulse_unit\":\"unity_mass_unit*unity_unit/second\"}");
    }

    private static void AppendNode(StringBuilder json, PhysicalNodeSnapshot node)
    {
        json.Append("{\"entity_id\":");
        AppendString(json, node.EntityId);
        json.Append(",\"unity_instance_id\":").Append(node.UnityInstanceId.ToString(CultureInfo.InvariantCulture));
        json.Append(",\"object_class\":");
        AppendString(json, node.ObjectClass);
        json.Append(",\"object_type\":");
        AppendString(json, node.ObjectType);
        json.Append(",\"screen_polygon\":[");
        for (int index = 0; index < node.ScreenPolygon.Length; index++)
        {
            if (index > 0)
            {
                json.Append(',');
            }
            AppendVector(json, node.ScreenPolygon[index]);
        }
        json.Append("],\"world_pose\":{\"position\":");
        AppendVector(json, node.WorldPosition);
        json.Append(",\"rotation_degrees\":");
        AppendFloat(json, node.RotationDegrees);
        json.Append("},\"life\":");
        AppendNullableFloat(json, node.Life);
        json.Append(",\"body\":");
        AppendBody(json, node.Body);
        json.Append('}');
    }

    private static void AppendBody(StringBuilder json, PhysicalBodySnapshot body)
    {
        json.Append("{\"present\":").Append(body.Present ? "true" : "false");
        json.Append(",\"velocity\":");
        if (body.Velocity.HasValue)
        {
            AppendVector(json, body.Velocity.Value);
        }
        else
        {
            json.Append("null");
        }
        json.Append(",\"angular_velocity_degrees_per_second\":");
        AppendNullableFloat(json, body.AngularVelocityDegreesPerSecond);
        json.Append(",\"mass_unity_units\":");
        AppendNullableFloat(json, body.MassUnityUnits);
        json.Append(",\"kinetic_energy_unity_units\":");
        AppendNullableFloat(json, body.KineticEnergyUnityUnits);
        json.Append('}');
    }

    private static void AppendVector(StringBuilder json, Vector2 value)
    {
        json.Append("{\"x\":");
        AppendFloat(json, value.x);
        json.Append(",\"y\":");
        AppendFloat(json, value.y);
        json.Append('}');
    }

    private static void AppendNullableFloat(StringBuilder json, float? value)
    {
        if (value.HasValue)
        {
            AppendFloat(json, value.Value);
        }
        else
        {
            json.Append("null");
        }
    }

    private static void AppendFloat(StringBuilder json, float value)
    {
        json.Append(value.ToString("R", CultureInfo.InvariantCulture));
    }

    private static void AppendString(StringBuilder json, string value)
    {
        json.Append('"');
        for (int index = 0; index < value.Length; index++)
        {
            char character = value[index];
            switch (character)
            {
                case '"': json.Append("\\\""); break;
                case '\\': json.Append("\\\\"); break;
                case '\b': json.Append("\\b"); break;
                case '\f': json.Append("\\f"); break;
                case '\n': json.Append("\\n"); break;
                case '\r': json.Append("\\r"); break;
                case '\t': json.Append("\\t"); break;
                default:
                    if (character < 32)
                    {
                        json.Append("\\u").Append(((int)character).ToString("x4", CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        json.Append(character);
                    }
                    break;
            }
        }
        json.Append('"');
    }
}
