"""Apply the scoped observer port to a recovered working project, never the reference."""
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
OBSERVERS = ROOT / "tasks/task_template_designer/Assets/Scripts/GroundTruth"


def replace(text, before, after, count=1):
    if text.count(before) != count:
        raise ValueError(f"reference source changed at observer insertion: {before[:100]!r}")
    return text.replace(before, after)


def source_changes(sources):
    """Pure source transform so tests and the actual port use the same seam."""
    changed = dict(sources)
    for name in ("OBjData.cs", "BirdData.cs", "SlingData.cs"):
        changed[name] = replace(changed[name], "\n{\n", "\n{\n\tpublic string scenarioObjectId;\n")
    changed["SlingData.cs"] = replace(changed["SlingData.cs"], "\t\tthis.y = y;",
                                       "\t\tthis.y = y;\n\t\tthis.scenarioObjectId = null;")
    text = changed["LevelLoader.cs"]
    for expression, count in (("new BirdData(value)", 1),
                              ("new BlockData(value2, rotation, x, y, material)", 1),
                              ("new OBjData(value2, rotation, x, y)", 2),
                              ("new PlatData(value2, rotation, x, y, scaleX2, scaleY2)", 1),
                              ("new ExternalAgentData(value2, rotation, x, y, material)", 1),
                              ("new NoveltyData(value2, rotation, x, y, material, scaleX, scaleY)", 1)):
        text = replace(text, expression, expression + ' { scenarioObjectId = xmlReader.GetAttribute("scenarioObjectId") }', count)
    text = replace(text, '\t\t\taBLevel.slingshot.y = (float)Convert.ToDouble(xmlReader.Value);',
                   '\t\t\taBLevel.slingshot.y = (float)Convert.ToDouble(xmlReader.Value);\n'
                   '\t\t\taBLevel.slingshot.scenarioObjectId = xmlReader.GetAttribute("scenarioObjectId");')
    changed["LevelLoader.cs"] = text

    text = changed["ABGameWorld.cs"]
    text = replace(text, "\t\tPhysics2D.autoSimulation = false;",
                   "\t\tPhysics2D.autoSimulation = false;\n\t\tPhysicalSnapshotRuntime.Attach(gameObject).ResetLevel();")
    text = replace(text, "\tpublic void DecodeLevel(ABLevel currentLevel)\n\t{",
                   "\tpublic void DecodeLevel(ABLevel currentLevel)\n\t{\n\t\tPhysicalSnapshotRuntime.Attach(gameObject).ResetLevel();")
    text = replace(text, "\t\t\tPhysics2D.Simulate(Time.fixedDeltaTime);",
                   "\t\t\tPhysicalSnapshotRuntime runtime = PhysicalSnapshotRuntime.Attach(gameObject);\n"
                   "\t\t\truntime.BeforeManualSimulation();\n"
                   "\t\t\tPhysics2D.Simulate(Time.fixedDeltaTime);\n"
                   "\t\t\truntime.AfterManualSimulation();")
    text = replace(text, "\tpublic bool LevelFailed()", "\tpublic bool IsLevelClearPending() { return _levelCleared; }\n\n\tpublic bool LevelFailed()")
    start, end = text.index("\tpublic void DecodeLevel("), text.index("\tprivate IEnumerator GTTest(")
    decode = text[start:end]
    for kind, next_kind, variable, factory in (
            ("BirdData bird", "OBjData pig", "bird", "AddBird"),
            ("OBjData pig", "BlockData block", "pig", "AddPig"),
            ("BlockData block", "PlatData platform", "block", "AddBlock"),
            ("PlatData platform", "OBjData tnt", "platform", "AddPlatform"),
            ("OBjData tnt", "ExternalAgentData externalagent", "tnt", "AddBlock"),
            ("ExternalAgentData externalagent", "NoveltyData novelty", "externalagent", "AddExternalAgent"),
            ("NoveltyData novelty", None, "novelty", "AddNovelty")):
        first = decode.index("foreach (" + kind)
        last = decode.index("foreach (" + next_kind) if next_kind else len(decode)
        section = decode[first:last]
        # Existing factory return values stay in their original statements.
        section, assigned = re.subn(r"(GameObject (\w+) = " + factory + r"\([^;]+;)",
                                    lambda m: m[1] + f"\n\t\t\t\tScenarioObjectIdentity.Assign({m[2]}, {variable}.scenarioObjectId);", section)
        section, bare = re.subn(r"(?m)^(\s*)" + factory + r"\(([^;]+)\);$",
                                lambda m: m[1] + f"ScenarioObjectIdentity.Assign({factory}({m[2]}), {variable}.scenarioObjectId);", section)
        if assigned + bare == 0:
            raise ValueError(f"no canonical {factory} identity insertion for {kind}")
        decode = decode[:first] + section + decode[last:]
    decode = replace(decode, '\t\t_slingshot.name = "Slingshot";',
                     '\t\t_slingshot.name = "Slingshot";\n\t\tScenarioObjectIdentity.Assign(_slingshot, currentLevel.slingshot.scenarioObjectId);')
    decode = replace(decode,
                     "UnityEngine.Object.Instantiate(ABWorldAssets.LANDSCAPE, position2, Quaternion.identity).transform.parent = base.transform;",
                     "GameObject landscape = UnityEngine.Object.Instantiate(ABWorldAssets.LANDSCAPE, position2, Quaternion.identity);\n"
                     "\t\t\tlandscape.transform.parent = base.transform;\n"
                     '\t\t\tScenarioObjectIdentity.Assign(landscape, "world:landscape:" + i.ToString("D4"));')
    decode = replace(decode, "UnityEngine.Object.Instantiate(ABWorldAssets.GROUND_EXTENSION, position2 + vector, Quaternion.identity);",
                     "ScenarioObjectIdentity.Assign(UnityEngine.Object.Instantiate(ABWorldAssets.GROUND_EXTENSION, position2 + vector, Quaternion.identity), "
                     '\n\t\t\t\t\t"world:ground_extension:" + i.ToString("D4") + ":" + j.ToString("D4"));')
    text = text[:start] + decode + text[end:]
    for verdict, event in (("Fail", 'RecordLevelFailCallback("no_playable_birds")'),
                           ("Pass", "RecordLevelClearCallback(ABSingleton<HUD>.Instance.GetScore())")):
        anchor = f'ABSingleton<EvaluationHandler>.Instance.RecordEvaluationScore("{verdict}");'
        text = replace(text, anchor, anchor + f"\n\t\t\tPhysicalSnapshotRuntime.{event};\n\t\t\tPhysicalSnapshotRuntime.FinalizeTerminalCallback();")
    anchor = "\t\t_birds.Remove(bird);"
    text = replace(text, anchor, anchor + "\n\t\tif (_birds.Count == 0) PhysicalSnapshotRuntime.RecordBirdExhaustionCallback();")
    changed["ABGameWorld.cs"] = text

    text = changed["ABGameObject.cs"]
    text = replace(text, "\tpublic virtual void Die(bool withEffect = true)\n\t{",
                   "\tpublic virtual void Die(bool withEffect = true)\n\t{\n"
                   "\t\tPhysicalSnapshotRuntime.RecordDeathCallback(PhysicalSnapshotRuntime.EntityIdForCallback(gameObject));")
    text = replace(text, "\tprivate void WaitParticlesAndDestroy()", "\tprivate void OnDestroy()\n\t{\n"
                   "\t\tPhysicalSnapshotRuntime.RecordDestroyedCallback(PhysicalSnapshotRuntime.EntityIdForCallback(gameObject));\n\t}\n\n"
                   "\tprivate void WaitParticlesAndDestroy()")
    changed["ABGameObject.cs"] = text
    changed["ABPig.cs"] = replace(changed["ABPig.cs"], "\t\tABSingleton<ABGameWorld>.Instance.KillPig(this);",
                                  "\t\tPhysicalSnapshotRuntime.RecordPigRemovedCallback(PhysicalSnapshotRuntime.EntityIdForCallback(gameObject));\n"
                                  "\t\tABSingleton<ABGameWorld>.Instance.KillPig(this);")
    changed["ABBird.cs"] = replace(changed["ABBird.cs"], "\t\t_rigidBody.AddForce(force, ForceMode2D.Impulse);",
                                   "\t\t_rigidBody.AddForce(force, ForceMode2D.Impulse);\n"
                                   "\t\tPhysicalSnapshotRuntime.RecordLaunchCallback(PhysicalSnapshotRuntime.EntityIdForCallback(gameObject), _rigidBody.velocity);")
    # Record the collision before any original lethal callback can disable it.
    for name in ("ABGameObject.cs", "ABPig.cs", "DieOnBirdHitCountPig.cs", "ABBird.cs", "ABBlock.cs", "ABBirdBlack.cs", "ABEgg.cs"):
        changed[name], count = re.subn(r"(public (?:override|virtual) void OnCollisionEnter2D\(Collision2D collision\)\n\t\{\n)",
                                      r"\1\t\tPhysicalSnapshotRuntime.RecordCollisionCallback(collision);\n", changed[name])
        if count != 1:
            raise ValueError(f"canonical collision callback changed in {name}")
    text = changed["AIBirdsConnection.cs"]
    text = replace(text, "ABSingleton<HUD>.Instance.shootDone = false;",
                   "PhysicalSnapshotRuntime.BeginV2ShotCallback();\n\t\tABSingleton<HUD>.Instance.shootDone = false;", 3)
    anchor = '\n\t\tsocket = new WebSocket(new Uri("ws://localhost:" + port + "/"));'
    text = replace(text, anchor, '\n\t\tstring physicsPort = Environment.GetEnvironmentVariable("NOVPHY_PHYSICS_CAPTURE_PORT");\n'
                   '\t\tif (!string.IsNullOrEmpty(physicsPort)) PhysicsCaptureDirectSocket.Attach(gameObject, int.Parse(physicsPort));\n' + anchor)
    changed["AIBirdsConnection.cs"] = text
    return {name: value for name, value in changed.items() if value != sources[name]}


def instrument(project):
    project = Path(project)
    receipt = project.parent / "instrumentation.json"
    if receipt.exists():
        raise ValueError("observer port already applied; preserve its recorded version")
    scripts = project / "Assets/Scripts/Assembly-CSharp"
    sources = {p.name: p.read_text() for p in scripts.glob("*.cs")}
    changes = source_changes(sources)
    target = project / "Assets/Scripts/CanonicalCapture"
    target.mkdir()
    observer_names = []
    for source in sorted(OBSERVERS.glob("*.cs")):
        if not source.name.startswith(("Physical", "Physics", "ObservationCapture", "ScenarioObjectIdentity")):
            continue
        value = source.read_text()
        if source.name == "PhysicalSnapshotExporter.cs":
            value = replace(value, "symbolicState.GetGTJson(false)", "symbolicState.GetGTJson()")
        if source.name == "PhysicsCaptureV2AlignedObservationRecorder.cs":
            value = replace(value, "new PhysicsCaptureV2AlignedObservationRecorder(root, captureId)",
                            "new PhysicsCaptureV2AlignedObservationRecorder(Path.Combine(root, captureId), captureId)")
        if source.name == "PhysicalSnapshotRuntime.cs":
            start = value.index("    private void FixedUpdate()")
            end = value.index("    public void RecordCollision(", start)
            value = value[:start] + '''    public void BeforeManualSimulation()
    {
        if (ABGameWorld.SimulationSpeed != 1f)
            throw new InvalidOperationException("canonical aligned capture requires simulation speed 1");
        Clock.ObserveFixedStep(Time.fixedTime);
    }

    public void AfterManualSimulation()
    {
        CaptureV2PostPhysicsStep();
    }

''' + value[end:]
            value = replace(value, "        v2PostPhysicsLoop = StartCoroutine(CaptureV2PostPhysicsSteps());\n", "")
            value = replace(value, "    private Coroutine v2PostPhysicsLoop;\n", "")
            value = replace(value, "        if (v2PostPhysicsLoop != null) StopCoroutine(v2PostPhysicsLoop);\n        v2PostPhysicsLoop = null;\n", "")
            start = value.index("    private IEnumerator CaptureV2PostPhysicsSteps()")
            end = value.index("    private void CaptureV2PostPhysicsStep()", start)
            value = value[:start] + value[end:]
        (target / source.name).write_text(value)
        observer_names.append(source.name)
    shutil.copy2(ROOT / "tasks/issue_76_canonical/CanonicalCaptureSeed.cs", target)
    restore_solid_color_shader(project)
    before = project.parent / "pre-instrumentation-source"
    before.mkdir()
    for name, value in changes.items():
        (before / name).write_text(sources[name])
        (scripts / name).write_text(value)
    report = {"schema": "issue_76_canonical_observer_port_v1", "project": str(project),
              "changed_gameplay_sources": sorted(changes), "observer_sources": observer_names,
              "source_before": {name: sources[name] for name in changes}, "source_after": changes,
              "manual_physics_step": "before clock, original Simulate call, after capture",
              "simulation_speed": 1, "multi_shot_rgb": "separate capture-ID directory per shot",
              "seed_variable": "NOVPHY_ENVIRONMENT_SEED", "fresh_access": False,
              "supported_scope": "normal and built-in appearance pigs; other novelty cells untested"}
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[canonical-player] instrumented {len(changes)} original files; prior source retained in {before}", flush=True)
    return report


def restore_solid_color_shader(project):
    """Retain the dummy export and restore the repository's existing shader source."""
    project = Path(project)
    target = project / "Assets/Shader/Custom_Solid Color.shader"
    source = ROOT / "tasks/task_template_designer/Assets/Scripts/Shaders/SolidColor.shader"
    receipt = project.parent / "shader-restoration.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if target.read_text() != saved["source_after"] or source.read_text() != saved["source_after"]:
            raise ValueError("recorded solid-color shader restoration changed")
        return saved
    before, after = target.read_text(), source.read_text()
    if "DummyShaderTextExporter" not in before or 'Shader "Custom/Solid Color"' not in after:
        raise ValueError("expected recovered dummy and existing solid-color shader source")
    report = {"schema": "issue_76_solid_color_shader_restoration_v1", "source": str(source),
              "source_before": before, "source_after": after,
              "reason": "ABSlingshot actually uses this shader; do not publish AssetRipper's dummy",
              "render_equivalence_established": False}
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    target.write_text(after)
    return report
