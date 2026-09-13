"""Publish the completed native refit's negative development evidence."""
import argparse
from collections import Counter
import gzip
import time

from scripts import run_issue_76_native_refit as fit
from scripts import run_issue_76_automated_refit as execution


OUTPUT = fit.files.ROOT / "data/issue-76-native-refit"
SOURCE = "scripts/publish_issue_76_native_refit.py"
AUDIT = "data/issue-76-native-outcome-audit/report.json"


def compact(value):
    """Keep every assigned row; replace step lists with exact work summaries."""
    rows = []
    for row in value["rows"]:
        policies = {}
        for name, policy in row["policies"].items():
            record = {k: v for k, v in policy.items() if k != "work"}
            if "work" in policy:
                work = policy["work"]
                counts = Counter((w["mode"], w["horizon_native_steps"]) for w in work)
                record["work_histogram"] = [
                    {"mode": mode, "horizon_native_steps": horizon, "count": count}
                    for (mode, horizon), count in sorted(counts.items())]
                record["work_totals"] = {k: sum(w[k] for w in work) for k in (
                    "transition_calls", "controller_calls", "symbol_decoder_calls", "linear_macs")}
            policies[name] = record
        rows.append({**row, "policies": policies})
    return {**value, "rows": rows}


def report():
    plan = fit.load_plan()
    execution.load_amendment(fit.ROOT, plan)
    original = fit.publish_diagnostic(fit.ROOT, plan, validate=True)
    value = compact(original)
    audit = fit.files.read(fit.files.ROOT / AUDIT)
    value.update(
        schema="issue_76_native_refit_compact_publication_v1",
        original_schema=original["schema"],
        complete_step_records="diagnostic-report.json.gz",
        endpoint_native_steps=fit.ENDPOINT,
        endpoint_seconds=fit.ENDPOINT * .0004,
        collection_outcome_correction={
            "source": AUDIT,
            "assigned": audit["assigned"],
            "native_terminal_successes": audit["native_terminal_successes"],
            "legacy_success_field_is_not_authoritative": True,
            "trained_policy_gameplay_evidence": False},
        publication_source_text={SOURCE: (fit.files.ROOT / SOURCE).read_text()},
        interpretation="Completed development refit, not a fresh experiment or an advancement disposition.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    value = report()
    target = OUTPUT / "report.json"
    archive = OUTPUT / "diagnostic-report.json.gz"
    raw = (fit.ROOT / "diagnostic-report.json").read_bytes()
    if args.publish:
        fit.files.write(target, value)
        if not archive.exists():
            archive.write_bytes(gzip.compress(raw, mtime=0))
    if fit.files.read(target) != value or gzip.decompress(archive.read_bytes()) != raw:
        raise ValueError("compact publication or complete diagnostic archive differs")
    fit.log(f"native refit publication validated: 600 assigned rows; elapsed={time.monotonic()-started:.2f}s")


if __name__ == "__main__":
    main()
