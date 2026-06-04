from pathlib import Path
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

    def test_app_builds_agent_action_from_drag_release(self) -> None:
        app = self.read_static("app.js")

        self.assertIn("function buildAgentActionFromDrag", app)
        self.assertIn("action_type: 'drag_release'", app)
        self.assertIn("coordinate_frame: 'slingshot_relative'", app)
        self.assertIn("drag_start", app)
        self.assertIn("drag_release", app)
        self.assertIn("release.x - start.x", app)
        self.assertIn("start.y - release.y", app)
        self.assertIn("tapTime", app)

    def test_app_transfers_agent_action_immediately_and_optionally_shoots(self) -> None:
        app = self.read_static("app.js")

        self.assertNotIn("setTimeout", app)
        self.assertNotIn("500", app)
        self.assertIn("post('/api/agent-action'", app)
        self.assertIn("autoExecuteAgentAction", app)
        self.assertIn("post('/api/shot', { ...result.shot, async: true })", app)

    def test_ui_explains_drag_to_aim_and_release_coordinates(self) -> None:
        index = self.read_static("index.html")

        self.assertIn("Drag on the frame to preview aim", index)
        self.assertIn("release to fill", index)
        self.assertIn("origin at bottom-left", index)
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
        self.assertIn("/api/shot", readme)
        self.assertIn("no live TCP drag", readme)


if __name__ == "__main__":
    unittest.main()
