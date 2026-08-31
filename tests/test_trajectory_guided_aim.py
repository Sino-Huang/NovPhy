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
                        "properties": {"id": "bird:fixture", "label": "redBird"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [94, 294], [106, 294], [106, 306], [94, 306]
                            ]],
                        },
                    },
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
            arc="lowest_clear_full_pull",
            aim_point="visible_polygon_upper_edge",
            bird_radius_world=0.17,
            clearance_margin_world=0.34,
            clearance_margin_minimum_target_distance_world=8.0,
            tap_time_ms=0,
        )

        self.assertTrue(bounds.contains(aim.action))
        self.assertGreater(aim.action.drag_y, 0)
        self.assertEqual(aim.target_label, "pig_basic_small_1")
        self.assertEqual(aim.target_canvas, (300.0, 210.0))
        self.assertEqual(aim.aim_point, "visible_polygon_upper_edge")
        self.assertEqual(aim.arc, "low")
        self.assertLess(aim.predicted_miss_pixels, 1.0)

    def test_obstructed_low_arc_uses_collision_free_high_arc(self) -> None:
        symbolic_state = [{
            "features": [
                {
                    "properties": {"id": "bird:fixture", "label": "redBird"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [94, 294], [106, 294], [106, 306], [94, 306]
                        ]],
                    },
                },
                {
                    "properties": {"id": "pig:fixture", "label": "pig_basic_big_1"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [290, 210], [310, 210], [310, 230], [290, 230]
                        ]],
                    },
                },
                {
                    "properties": {"id": "platform:fixture", "label": "Platform"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [220, 205], [250, 205], [250, 275], [220, 275]
                        ]],
                    },
                },
            ],
        }]
        bounds = SlingshotActionBounds(
            drag_x=(-160, -10),
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
            arc="lowest_clear_full_pull",
            aim_point="visible_polygon_upper_edge",
            bird_radius_world=0.17,
            clearance_margin_world=0.34,
            clearance_margin_minimum_target_distance_world=8.0,
            tap_time_ms=0,
        )

        self.assertEqual(aim.arc, "high")
        self.assertEqual(aim.obstacle_clearance, "bird_volume_swept_clear")
        self.assertIn("platform:fixture", aim.cleared_obstacle_ids)

    def test_prediction_margin_moves_the_observed_platform_shot_higher(self) -> None:
        symbolic_state = [{
            "features": [
                {
                    "properties": {"id": "pig:observed", "label": "pig_basic_big_8"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [303, 178], [326, 178], [326, 201], [303, 201]
                        ]],
                    },
                },
                {
                    "properties": {"id": "platform:observed", "label": "Platform"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [282.7, 192.5], [298, 192.5],
                            [298, 217.5], [282.7, 217.5]
                        ]],
                    },
                },
            ],
        }]

        aim = aim_directly_at_visible_pig(
            symbolic_state,
            {
                "canvasX": 127.0,
                "canvasY": 256.0,
                "pixelsPerWorldUnit": 23.844282,
            },
            SlingshotActionBounds(
                drag_x=(-160, -10),
                drag_y=(-80, 80),
                tap_time_ms=(0, 1000),
                release_time_ms=600,
            ),
            target_rank=0,
            arc="lowest_clear_full_pull",
            aim_point="visible_polygon_upper_edge",
            bird_radius_world=0.17,
            clearance_margin_world=0.34,
            clearance_margin_minimum_target_distance_world=8.0,
            tap_time_ms=0,
        )

        self.assertGreaterEqual(
            (aim.action.drag_x ** 2 + aim.action.drag_y ** 2) ** 0.5,
            23.844282,
        )
        self.assertEqual(aim.margin_applied_obstacle_ids, ())
        self.assertEqual(aim.clearance_margin_world, 0.34)

    def test_prediction_margin_applies_to_support_adjacent_to_pig(self) -> None:
        def feature(object_id, label, bounds):
            left, top, right, bottom = bounds
            return {
                "properties": {"id": object_id, "label": label},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [left, top], [right, top],
                        [right, bottom], [left, bottom],
                    ]],
                },
            }

        aim = aim_directly_at_visible_pig(
            [{"features": [
                feature("pig:adjacent", "pig_basic_big_1", (319, 181, 342, 205)),
                feature("platform:adjacent", "Platform", (303, 195.6, 318.3, 220.6)),
            ]}],
            {
                "canvasX": 127.0,
                "canvasY": 256.0,
                "pixelsPerWorldUnit": 23.844282,
            },
            SlingshotActionBounds(
                drag_x=(-160, -10),
                drag_y=(-80, 80),
                tap_time_ms=(0, 1000),
                release_time_ms=600,
            ),
            target_rank=0,
            arc="lowest_clear_full_pull",
            aim_point="visible_polygon_upper_edge",
            bird_radius_world=0.17,
            clearance_margin_world=0.34,
            clearance_margin_minimum_target_distance_world=8.0,
            tap_time_ms=0,
        )

        self.assertEqual(
            aim.margin_applied_obstacle_ids, ("platform:adjacent",)
        )
        self.assertGreaterEqual(aim.target_distance_world, 8.0)
        self.assertGreaterEqual(
            (aim.action.drag_x ** 2 + aim.action.drag_y ** 2) ** 0.5,
            23.844282,
        )

    def test_expert_solvable_platform_layouts_allow_partial_pull(self) -> None:
        def feature(object_id, label, bounds):
            left, top, right, bottom = bounds
            return {
                "properties": {"id": object_id, "label": label},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [left, top], [right, top],
                        [right, bottom], [left, bottom],
                    ]],
                },
            }

        bounds = SlingshotActionBounds(
            drag_x=(-160, -10),
            drag_y=(-80, 80),
            tap_time_ms=(0, 1000),
            release_time_ms=600,
        )
        for pig_bounds, platform_bounds in (
            ((227, 192, 239, 203), (208, 203, 257, 218)),
            ((227, 186, 238, 197), (207, 197, 256, 212)),
        ):
            with self.subTest(pig_bounds=pig_bounds):
                aim = aim_directly_at_visible_pig(
                    [{"features": [
                        feature("pig:expert", "pig_basic_small_1", pig_bounds),
                        feature("platform:expert", "Platform", platform_bounds),
                    ]}],
                    {
                        "canvasX": 127.98783454987834,
                        "canvasY": 256.96107055961073,
                        "pixelsPerWorldUnit": 23.84428223844282,
                    },
                    bounds,
                    target_rank=0,
                    arc="lowest_clear_full_pull",
                    aim_point="visible_polygon_upper_edge",
                    bird_radius_world=0.17,
                    clearance_margin_world=0.34,
                    clearance_margin_minimum_target_distance_world=8.0,
                    tap_time_ms=0,
                )

                self.assertEqual(aim.arc, "high")
                self.assertLess(
                    (aim.action.drag_x ** 2 + aim.action.drag_y ** 2) ** 0.5,
                    23.84428223844282,
                )


if __name__ == "__main__":
    unittest.main()
