def _point_xy(point):
    if hasattr(point, "X") and hasattr(point, "Y"):
        return point.X, point.Y
    try:
        return point[0], point[1]
    except (TypeError, IndexError, KeyError):
        raise ValueError("release point must contain x and y values")


def _sequence_len(action):
    try:
        return len(action)
    except TypeError:
        return None


def _tap_time(action):
    if "tap_time" in action:
        return action["tap_time"]
    if "tapTime" in action:
        return action["tapTime"]
    return 0


def normalize_release_action(action, sling_center=None):
    if isinstance(action, dict):
        release = action.get("drag_release", action.get("release"))
        if release is None:
            raise ValueError("drag_release or release is required for dict actions")

        x, y = _point_xy(release)
        coordinate_frame = action.get("coordinate_frame", "slingshot_relative")
        if coordinate_frame == "slingshot_relative":
            dx, dy = x, y
        elif coordinate_frame == "absolute":
            if sling_center is None:
                raise ValueError("sling_center is required for absolute release actions")
            sling_x, sling_y = _point_xy(sling_center)
            dx = x - sling_x
            dy = sling_y - y
        else:
            raise ValueError("coordinate_frame must be slingshot_relative or absolute")
        return int(dx), int(dy), _tap_time(action)

    if hasattr(action, "X") and hasattr(action, "Y"):
        return int(action.X), int(action.Y), 0

    action_len = _sequence_len(action)
    if action_len is not None and action_len >= 2:
        tap_time = action[2] if action_len == 3 else 0
        return int(action[0]), int(action[1]), tap_time

    raise AssertionError("action type {} not recognized".format(type(action)))
