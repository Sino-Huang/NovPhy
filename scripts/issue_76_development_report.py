"""Progress-logged publication of the frozen collector's exact report schema."""
import argparse
import time

from scripts import run_issue_76_development as collection


def report(root, plan):
    entries = []
    started = time.monotonic()
    fragment = None
    for ordinal, member in enumerate(plan["members"], 1):
        # Reuse the frozen validation path; do not redefine capture eligibility.
        fragment = collection.report(root, {**plan, "members": [member]})
        entries.extend(fragment["entries"])
        elapsed = time.monotonic() - started
        collection.episode.old.log(f"report validation episode={ordinal}/{len(plan['members'])} elapsed={elapsed:.1f}s ETA={elapsed/ordinal*(len(plan['members'])-ordinal):.1f}s")
    if fragment is None:
        raise ValueError("cannot publish an empty collection plan")
    successes = sum(bool(e["result"] and e["result"].get("gameplay_success")) for e in entries)
    return {**fragment, "entries": entries,
            "collection_pipeline_passed": all(e["result"] and e["result"]["complete"] for e in entries),
            "assigned_episode_count": len(entries), "gameplay_successes": successes,
            "gameplay_failure_or_unattempted_count": len(entries) - successes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "development"), default="development")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--publish", action="store_true")
    modes.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    plan = collection.load_plan(args.stage)
    root = collection.SMOKE if args.stage == "smoke" else collection.DEVELOPMENT
    value = report(root, plan)
    count = sum(e["result"] is not None for e in value["entries"])
    target = collection.ROOT / f"data/issue-76-{args.stage}-censored/attempts-{count}/report.json"
    if args.publish:
        if target.exists() and collection.files.read(target) != value:
            raise ValueError("do not overwrite a different collector publication")
        collection.files.write(target, value)
    elif collection.files.read(target) != value:
        raise ValueError("published collector report differs from retained evidence")
    collection.episode.old.log(f"progress-logged report {'published' if args.publish else 'validated'}: {target}")


if __name__ == "__main__":
    main()
