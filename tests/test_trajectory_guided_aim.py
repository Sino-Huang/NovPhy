from __future__ import annotations

import unittest

from world_model.planning.gameplay import SlingshotActionBounds
from world_model.planning.trajectory_guided import aim_directly_at_visible_pig


class TrajectoryGuidedAimTests(unittest.TestCase):
    def test_low_arc_uses_visible_pig_upper_edge_and_matches_preview(self) -> None:
        symbolic_state = [
            {
                "features": [
                    {
                        "properties": {"id": "pig:fixture", "label": "pig_basic_small_1"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [290, 210], [310, 210], [310, 230], [290, 230]
                            ]],
                        },
                    }
                ]
            }
        ]
        bounds = SlingshotActionBounds(
            drag_x=(-160, -40),
            drag_y=(-80, 80),
            tap_time_ms=(0, 1000),
            release_time_ms=600,
        )

        aim = aim_directly_at_visible_pig(
            symbolic_state,
            {
                "canvasX": 100.0,
                "canvasY": 300.0,
                "pixelsPerWorldUnit": 32.0,
            },
            bounds,
            target_rank=0,
            arc="low",
            aim_point="visible_polygon_upper_edge",
            tap_time_ms=0,
        )

        self.assertTrue(bounds.contains(aim.action))
        self.assertGreater(aim.action.drag_y, 0)
        self.assertEqual(aim.target_label, "pig_basic_small_1")
        self.assertEqual(aim.target_canvas, (300.0, 210.0))
        self.assertEqual(aim.aim_point, "visible_polygon_upper_edge")
        self.assertLess(aim.predicted_miss_pixels, 1.0)


if __name__ == "__main__":
    unittest.main()
