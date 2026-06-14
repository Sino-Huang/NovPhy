from pathlib import Path
import json
import subprocess
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "webui" / "static"


class WebUIStaticTests(unittest.TestCase):
    def read_static(self, name: str) -> str:
        return (STATIC / name).read_text(encoding="utf-8")

    def test_app_uses_pointer_drag_for_local_aim_preview(self) -> None:
        app = self.read_static("app.js")

        for event_name in ("pointerdown", "pointermove", "pointerup", "pointercancel"):
            with self.subTest(event_name=event_name):
                self.assertIn(event_name, app)

        self.assertIn("setPointerCapture", app)
        self.assertIn("releasePointerCapture", app)
        self.assertIn("drawAimPreview", app)
        self.assertIn("drawTrajectoryPreview", app)
        self.assertIn("aimStartPoint", app)
        self.assertIn("aimCurrentPoint", app)

    def test_trajectory_preview_matches_unity_launch_integration(self) -> None:
        app = self.read_static("app.js")

        self.assertIn("WEB_TRAJECTORY_STEPS = 500", app)
        self.assertIn("WEB_TRAJECTORY_TIME_STEP = 0.02", app)
        self.assertIn("WEB_TRAJECTORY_MAX_LAUNCH_SPEED = 10", app)
        self.assertIn("WEB_TRAJECTORY_LAUNCH_GRAVITY = 0.48", app)
        self.assertIn("WEB_TRAJECTORY_DRAG_RADIUS_WORLD = 1", app)
        self.assertIn("let trajectoryWorldWidth = 17.5", app)
        self.assertIn("trajectoryWorldWidth = Number(frame.trajectoryWorldWidth)", app)
        self.assertIn("trajectorySlingCenter = frame.trajectorySlingCenter || null", app)
        self.assertIn("canvas.width / trajectoryWorldWidth", app)
        self.assertIn("function previewSlingCenterPoint", app)
        self.assertIn("function cappedReleasePoint", app)
        self.assertIn("function buildTrajectoryPreviewPoints", app)
        self.assertIn("position = { ...release }", app)
        self.assertIn("0.5 * gravityY * timeStep * timeStep", app)
        self.assertIn("velocityY += gravityY * timeStep", app)
        self.assertIn("const releasePoint = cappedReleasePoint(slingCenter, aimCurrentPoint)", app)
        self.assertIn("buildAgentActionFromDrag(slingCenter, releasePoint)", app)
        self.assertIn("fillShotFields(releasePoint)", app)
        self.assertNotIn("velocityX * time * 0.18", app)
        self.assertNotIn("time * time * 5", app)

    def test_trajectory_math_uses_real_sling_center_for_strength(self) -> None:
        script = textwrap.dedent(
            f"""
            const fs = require('fs');
            const vm = require('vm');
            const app = fs.readFileSync({json.dumps(str(STATIC / 'app.js'))}, 'utf8');
            const canvas = {{
              width: 600,
              height: 480,
              getContext: () => ({{
                save() {{}}, restore() {{}}, beginPath() {{}}, moveTo() {{}}, lineTo() {{}}, stroke() {{}},
                clearRect() {{}}, putImageData() {{}}, createImageData: (w, h) => ({{ data: new Uint8ClampedArray(w * h * 4) }}),
            }}),
              addEventListener() {{}}, setPointerCapture() {{}}, hasPointerCapture: () => false,
              releasePointerCapture() {{}}, classList: {{ add() {{}}, remove() {{}} }},
            }};
            const elements = new Map();
            function element(id) {{
              if (!elements.has(id)) elements.set(id, {{ textContent: '', value: '0', checked: false, classList: {{ add() {{}}, remove() {{}}, toggle() {{}} }}, addEventListener() {{}} }});
              return elements.get(id);
            }}
            const context = {{
              document: {{ getElementById: (id) => id === 'gameCanvas' ? canvas : element(id) }},
              fetch() {{}}, atob: (value) => Buffer.from(value, 'base64').toString('binary'),
              setInterval() {{}}, clearInterval() {{}}, console,
              Uint8ClampedArray, Buffer, Math, Number, Date, JSON,
            }};
            vm.createContext(context);
            vm.runInContext(app, context);
            vm.runInContext('updateTelemetry({{ trajectoryWorldWidth: 30, trajectorySlingCenter: {{ canvasX: 260, canvasY: 300 }}, state: {{ name: "PLAYING" }} }});', context);
            const result = vm.runInContext(`(() => {{
              const start = {{ canvasX: 300, canvasY: 300 }};
              const target = {{ canvasX: 100, canvasY: 300 }};
              const release = cappedReleasePoint(previewSlingCenterPoint(start), target);
              const points = buildTrajectoryPreviewPoints(start, target);
              return {{
                length: points.length,
                release,
                first: points[0],
                second: points[1],
                action: buildAgentActionFromDrag(previewSlingCenterPoint(start), release),
              }};
            }})()`, context);
            console.log(JSON.stringify(result));
            """
        )

        completed = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
        result = json.loads(completed.stdout)

        self.assertEqual(result["length"], 500)
        self.assertAlmostEqual(result["release"]["canvasX"], 240.0)
        self.assertAlmostEqual(result["release"]["canvasY"], 299.0)
        self.assertAlmostEqual(result["first"]["canvasX"], 240.0)
        self.assertAlmostEqual(result["first"]["canvasY"], 299.0)
        self.assertAlmostEqual(result["second"]["canvasX"], 244.0)
        self.assertAlmostEqual(result["second"]["canvasY"], 299.018816)
        self.assertEqual(result["action"]["drag_start"], [260, 179])
        self.assertEqual(result["action"]["drag_release"], [-20, 0])

    def test_draw_frame_applies_runtime_sling_center_to_preview_origin(self) -> None:
        script = textwrap.dedent(
            f"""
            const fs = require('fs');
            const vm = require('vm');
            const app = fs.readFileSync({json.dumps(str(STATIC / 'app.js'))}, 'utf8');
            const canvas = {{
              width: 640,
              height: 480,
              getContext: () => ({{
                save() {{}}, restore() {{}}, beginPath() {{}}, moveTo() {{}}, lineTo() {{}}, stroke() {{}},
                clearRect() {{}}, putImageData() {{}}, createImageData: (w, h) => ({{ data: new Uint8ClampedArray(w * h * 4) }}),
              }}),
              addEventListener() {{}}, setPointerCapture() {{}}, hasPointerCapture: () => false,
              releasePointerCapture() {{}}, classList: {{ add() {{}}, remove() {{}} }},
            }};
            const elements = new Map();
            function element(id) {{
              if (!elements.has(id)) elements.set(id, {{ textContent: '', value: '0', checked: false, classList: {{ add() {{}}, remove() {{}}, toggle() {{}} }}, addEventListener() {{}} }});
              return elements.get(id);
            }}
            const context = {{
              document: {{ getElementById: (id) => id === 'gameCanvas' ? canvas : element(id) }},
              fetch() {{}},
              atob: (value) => Buffer.from(value, 'base64').toString('binary'),
              setInterval() {{}}, clearInterval() {{}}, console,
              Uint8ClampedArray, Buffer, Math, Number, Date, JSON,
            }};
            vm.createContext(context);
            vm.runInContext(app, context);
            const rgbBase64 = Buffer.alloc(640 * 480 * 3).toString('base64');
            vm.runInContext(`drawFrame({{
              width: 640,
              height: 480,
              rgbBase64: "${{rgbBase64}}",
              trajectoryWorldWidth: 30,
              trajectorySlingCenter: {{ canvasX: 260, canvasY: 300 }},
              state: {{ name: 'PLAYING' }},
              currentLevel: 1,
              numberOfLevels: 20,
              score: 0,
            }});`, context);
            const result = vm.runInContext(`(() => {{
              const start = {{ canvasX: 300, canvasY: 300 }};
              const end = {{ canvasX: 100, canvasY: 300 }};
              const release = cappedReleasePoint(previewSlingCenterPoint(start), end);
              return buildAgentActionFromDrag(previewSlingCenterPoint(start), release);
            }})()`, context);
            console.log(JSON.stringify(result));
            """
        )

        completed = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
        result = json.loads(completed.stdout)

        self.assertEqual(result["drag_start"], [260, 179])
        self.assertAlmostEqual(result["drag_release"][0], -21.333333333333343)
        self.assertEqual(result["drag_release"][1], 0)

    def test_app_has_shared_coordinate_conversion_and_game_y_mapping(self) -> None:
        app = self.read_static("app.js")

        self.assertIn("function clientToCanvasPoint", app)
        self.assertIn("getBoundingClientRect", app)
        self.assertIn("canvas.height - 1 - canvasY", app)
        self.assertIn("Math.max(0, Math.min", app)

    def test_drag_preview_is_local_and_shot_api_stays_final_only(self) -> None:
        app = self.read_static("app.js")

        self.assertIn("post('/api/shot', payload)", app)
        self.assertNotIn("post('/api/drag", app)
        self.assertNotIn("post('/api/aim", app)
        self.assertIn("redrawCanvas", app)

    def test_app_builds_agent_action_from_drag_hold_release_and_hold_time(self) -> None:
        app = self.read_static("app.js")

        self.assertIn("function buildAgentActionFromDrag", app)
        self.assertIn("action_type: 'drag_hold_release'", app)
        self.assertIn("coordinate_frame: 'slingshot_relative'", app)
        self.assertIn("drag_start", app)
        self.assertIn("drag_release", app)
        self.assertIn("release.x - start.x", app)
        self.assertIn("start.y - release.y", app)
        self.assertIn("holdTime", app)
        self.assertIn("document.getElementById('holdTime').value", app)
        self.assertIn("tapTime", app)

    def test_drag_hold_release_action_math_executes_with_hold_time(self) -> None:
        script = textwrap.dedent(
            f"""
            const fs = require('fs');
            const vm = require('vm');
            const app = fs.readFileSync({json.dumps(str(STATIC / 'app.js'))}, 'utf8');
            const canvas = {{
              width: 640,
              height: 480,
              getContext: () => ({{
                save() {{}}, restore() {{}}, beginPath() {{}}, moveTo() {{}}, lineTo() {{}}, stroke() {{}},
                clearRect() {{}}, putImageData() {{}}, createImageData: (w, h) => ({{ data: new Uint8ClampedArray(w * h * 4) }}),
              }}),
              addEventListener() {{}}, setPointerCapture() {{}}, hasPointerCapture: () => false,
              releasePointerCapture() {{}}, classList: {{ add() {{}}, remove() {{}} }},
            }};
            const elements = new Map();
            function element(id) {{
              if (!elements.has(id)) elements.set(id, {{ textContent: '', value: '0', checked: false, classList: {{ add() {{}}, remove() {{}}, toggle() {{}} }}, addEventListener() {{}} }});
              return elements.get(id);
            }}
            element('tapTime').value = '70';
            element('holdTime').value = '120';
            const context = {{
              document: {{ getElementById: (id) => id === 'gameCanvas' ? canvas : element(id) }},
              fetch: async () => ({{ ok: true, json: async () => ({{ ok: true, connected: false, preflightErrors: [] }}) }}),
              atob: (value) => Buffer.from(value, 'base64').toString('binary'),
              setInterval() {{}}, clearInterval() {{}}, console,
              Uint8ClampedArray, Buffer, Math, Number, Date, JSON,
            }};
            vm.createContext(context);
            vm.runInContext(app, context);
            const result = vm.runInContext(`(() => buildAgentActionFromDrag({{ canvasX: 300, canvasY: 259 }}, {{ canvasX: 250, canvasY: 299 }}))()`, context);
            console.log(JSON.stringify(result));
            """
        )

        completed = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
        result = json.loads(completed.stdout)

        self.assertEqual(result["action_type"], "drag_hold_release")
        self.assertEqual(result["coordinate_frame"], "slingshot_relative")
        self.assertEqual(result["drag_start"], [300, 220])
        self.assertEqual(result["drag_release"], [-50, 40])
        self.assertEqual(result["holdTime"], 120)
        self.assertEqual(result["tapTime"], 70)

    def test_app_transfers_agent_action_immediately_and_optionally_shoots(self) -> None:
        app = self.read_static("app.js")

        self.assertNotIn("setTimeout", app)
        self.assertNotIn("agentActionTransferTimer", app)
        self.assertNotIn("cancelPendingAgentActionTransfer", app)
        self.assertIn("post('/api/agent-action'", app)
        self.assertIn("autoExecuteAgentAction", app)
        self.assertIn("post('/api/shot', { ...result.shot, async: true })", app)

    def test_shoot_button_includes_release_time_from_hold_time(self) -> None:
        script = textwrap.dedent(
            f"""
            const fs = require('fs');
            const vm = require('vm');
            const app = fs.readFileSync({json.dumps(str(STATIC / 'app.js'))}, 'utf8');
            const canvas = {{
              width: 640,
              height: 480,
              getContext: () => ({{
                save() {{}}, restore() {{}}, beginPath() {{}}, moveTo() {{}}, lineTo() {{}}, stroke() {{}},
                clearRect() {{}}, putImageData() {{}}, createImageData: (w, h) => ({{ data: new Uint8ClampedArray(w * h * 4) }}),
              }}),
              addEventListener() {{}}, setPointerCapture() {{}}, hasPointerCapture: () => false,
              releasePointerCapture() {{}}, classList: {{ add() {{}}, remove() {{}} }},
            }};
            const listeners = new Map();
            const elements = new Map();
            function element(id) {{
              if (!elements.has(id)) elements.set(id, {{
                textContent: '', value: '0', checked: false,
                classList: {{ add() {{}}, remove() {{}}, toggle() {{}} }},
                addEventListener(type, callback) {{ listeners.set(id + ':' + type, callback); }},
              }});
              return elements.get(id);
            }}
            element('shotX').value = '250';
            element('shotY').value = '180';
            element('tapTime').value = '70';
            element('holdTime').value = '120';
            element('fastShot').checked = true;
            const calls = [];
            const context = {{
              document: {{ getElementById: (id) => id === 'gameCanvas' ? canvas : element(id) }},
              fetch: async (path, options = {{}}) => {{
                calls.push({{ path, body: options.body ? JSON.parse(options.body) : null }});
                return {{ ok: true, json: async () => ({{ ok: true, connected: false, preflightErrors: [] }}) }};
              }},
              atob: (value) => Buffer.from(value, 'base64').toString('binary'),
              setInterval() {{}}, clearInterval() {{}}, console,
              Uint8ClampedArray, Buffer, Math, Number, Date, JSON,
            }};
            vm.createContext(context);
            vm.runInContext(app, context);
            calls.length = 0;
            (async () => {{
              await listeners.get('shootBtn:click')();
              console.log(JSON.stringify(calls));
            }})().catch((error) => {{ console.error(error); process.exit(1); }});
            """
        )

        completed = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
        calls = json.loads(completed.stdout)

        self.assertEqual(calls[0]["path"], "/api/shot")
        self.assertEqual(calls[0]["body"]["x"], 250)
        self.assertEqual(calls[0]["body"]["y"], 180)
        self.assertEqual(calls[0]["body"]["tapTime"], 70)
        self.assertEqual(calls[0]["body"]["releaseTime"], 120)
        self.assertTrue(calls[0]["body"]["fast"])

    def test_ui_explains_drag_to_aim_and_release_coordinates(self) -> None:
        index = self.read_static("index.html")

        self.assertIn("Drag on the frame to preview aim", index)
        self.assertIn("release to fill", index)
        self.assertIn("origin at bottom-left", index)
        self.assertIn("Hold time", index)
        self.assertIn("autoExecuteAgentAction", index)
        self.assertIn("id=\"autoExecuteAgentAction\" checked", index)
        self.assertIn("id=\"fastShot\" checked", index)

    def test_styles_mark_canvas_as_aiming_surface(self) -> None:
        styles = self.read_static("styles.css")

        self.assertIn("canvas.aiming", styles)
        self.assertIn("touch-action: none", styles)

    def test_readme_documents_local_preview_and_final_shot(self) -> None:
        readme = (ROOT / "src" / "webui" / "README.md").read_text(encoding="utf-8")

        self.assertIn("Drag on the canvas", readme)
        self.assertIn("browser-local", readme)
        self.assertIn("release fills", readme)
        self.assertIn("drag_hold_release", readme)
        self.assertIn("holdTime", readme)
        self.assertIn("releaseTime", readme)
        self.assertIn("/api/shot", readme)
        self.assertIn("no live TCP drag", readme)


if __name__ == "__main__":
    unittest.main()
