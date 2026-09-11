"""Prospective terminal-or-censored segments; raw capture outcomes stay unchanged."""
from scripts.canonical_native_trace import NativeTrace, MAX_STEPS, OBSERVATION_STRIDE


class NativeSegmentTrace:
    """A validated observed interval, not a claim that gameplay completed.

    Collection-plan membership determines fitting eligibility separately. This
    reader cannot authorize fitting on an engineering or held-out lineage.
    """
    def __init__(self, root):
        self.trace = NativeTrace(root)
        self.manifest = self.trace.manifest
        summary = self.trace.validate()
        self.censored = not summary["complete"]
        first, last = summary["first_fixed_step"], summary["last_fixed_step"]
        if summary["event_counts"].get("bird_launched", 0) != 1:
            raise ValueError("segment must contain exactly one observed launch")
        if self.censored:
            steps = [f["fixed_step"] for f in self.manifest["frame_records"]]
            if (summary["failure"] != "native_time_window_limit"
                    or last - first != MAX_STEPS or summary["sample_count"] != MAX_STEPS + 1
                    or self.manifest["complete_every_native_step"] is not True
                    or steps != list(range(first, last + 1, OBSERVATION_STRIDE))):
                raise ValueError("censoring requires an intact full native time window")
        self.summary = {**summary, "segment_schema": "native_observed_segment_v1",
                        "censored": self.censored, "observed_window_valid": True,
                        "terminal_observed": not self.censored}

    def observed_samples(self):
        wanted = {f["fixed_step"] for f in self.manifest["frame_records"]}
        for chunk in self.trace.chunks():
            for sample in chunk["fixed_step_samples"]:
                if sample["fixed_step"] in wanted:
                    yield sample

    def endpoint_indices(self, horizon):
        """Only exact observed endpoints; no padding, extrapolation or gap filling."""
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("endpoint horizon must be a positive native-step count")
        steps = [f["fixed_step"] for f in self.manifest["frame_records"]]
        indices = {step: index for index, step in enumerate(steps)}
        return [(index, indices[step + horizon]) for index, step in enumerate(steps)
                if step + horizon in indices]
