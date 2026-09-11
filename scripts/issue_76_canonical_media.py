"""Combined retained-frame gallery for all eight canonical engineering captures."""
import argparse
from pathlib import Path

from scripts import issue_76_expansion as files
from scripts import issue_76_native_media as previous
from scripts import run_issue_76_development as current


def publication():
    _, old = previous.inventory()
    entries = list(old["entries"])
    plan = current.load_plan("smoke")
    for member in plan["members"]:
        path = current.c2.result_path(current.SMOKE, member)
        if not path.exists():
            raise ValueError("finish the frozen engineering assignments before final media publication")
        result = files.read(path)
        for raw in sorted((current.SMOKE / "attempts" / member["identity"] / "aligned").glob("capture-v2:*")):
            native = files.read(raw / "native-manifest.json")
            frames = []
            for metadata in sorted(raw.glob("frame_*.json")):
                frame = files.read(metadata)
                image = metadata.with_suffix(".png")
                if not image.is_file():
                    raise ValueError("finished canonical media has missing RGB evidence")
                frames.append({"path": str(image), "fixed_step": frame["fixed_step"],
                               "fixed_time_seconds": frame["fixed_time_seconds"]})
            if [f["fixed_step"] for f in frames] != [f["fixed_step"] for f in native["frame_records"]]:
                raise ValueError("finished canonical media timestamps differ from native capture")
            entries.append({"episode": member["identity"], "condition": member["novelty_level"],
                "capture_id": raw.name, "phase": plan["identity"], "frames": frames,
                "capture_status": native["status"], "failure": native["failure"] or result["failure"],
                "episode_result": str(path), "native_manifest": str(raw / "native-manifest.json"),
                "native_samples": native["sample_count"], "gameplay_success": result["gameplay_success"],
                "recorded_seconds": frames[-1]["fixed_time_seconds"] - frames[0]["fixed_time_seconds"]})
    return {"schema": "issue_76_all_canonical_retained_media_v1", "entries": entries,
            "new_captures": 0, "synthetic_frames": 0,
            "canonical_budget": files.read(current.SMOKE / "budget.json"),
            "source_text": Path(__file__).read_text(), "earlier_renderer_source": old["source_text"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--publish", action="store_true")
    modes.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    root = files.ROOT / "data/issue-76-canonical-engineering-media"
    value = publication()
    rendered = previous.page(root, value).replace("C2 and native continuation: retained frames",
        "Canonical engineering: all eight retained captures")
    rendered = rendered.replace("<script>const entries=", "<p>The later censored-policy smoke passed collection integrity, not gameplay success. Earlier failures are unchanged. C1's separate aligned-player gallery remains in its original report.</p><script>const entries=")
    if args.publish:
        if (root / "media.json").exists() and files.read(root / "media.json") != value:
            raise ValueError("do not overwrite a different canonical media publication")
        files.write(root / "media.json", value)
        (root / "index.html").write_text(rendered)
    elif files.read(root / "media.json") != value or (root / "index.html").read_text() != rendered:
        raise ValueError("canonical media publication differs from retained evidence")
    print(f"[canonical-media] {'published' if args.publish else 'validated'} {len(value['entries'])} captures: {root/'index.html'}", flush=True)


if __name__ == "__main__":
    main()
