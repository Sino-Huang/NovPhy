using System;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEditor;
using UnityEditor.Build.Reporting;

public static class NovPhyBuild
{
    private const string RequiredVersion = "2019.4.41f2";
    private const string RequiredRevision = "6b23d448b533";

    public static void BuildPhysicsLinux()
    {
        if (Application.unityVersion != RequiredVersion)
            throw new InvalidOperationException("NovPhy physics player requires Unity " + RequiredVersion);

        string output = Environment.GetEnvironmentVariable("NOVPHY_BUILD_OUTPUT");
        if (String.IsNullOrEmpty(output) || !Path.IsPathRooted(output))
            throw new InvalidOperationException("NOVPHY_BUILD_OUTPUT must be an absolute path");

        string projectVersion = File.ReadAllText(Path.Combine(Directory.GetCurrentDirectory(), "ProjectSettings/ProjectVersion.txt"));
        if (!projectVersion.Contains(RequiredVersion) || !projectVersion.Contains(RequiredRevision))
            throw new InvalidOperationException("ProjectVersion.txt is not pinned to the required Unity revision");

        string[] scenes = EditorBuildSettings.scenes
            .Where(scene => scene.enabled)
            .Select(scene => scene.path)
            .ToArray();
        if (scenes.Length == 0)
            throw new InvalidOperationException("No enabled scenes are configured");

        Directory.CreateDirectory(Path.GetDirectoryName(output));
        BuildPlayerOptions options = new BuildPlayerOptions
        {
            scenes = scenes,
            locationPathName = output,
            target = BuildTarget.StandaloneLinux64,
            options = BuildOptions.NoUniqueIdentifier | BuildOptions.StrictMode
        };
        BuildReport report = BuildPipeline.BuildPlayer(options);
        if (report.summary.result != BuildResult.Succeeded || !File.Exists(output))
            throw new InvalidOperationException("Linux player build failed: " + report.summary.result);
    }
}
