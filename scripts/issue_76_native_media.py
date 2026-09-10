"""Retained-frame playback for C2 and its native continuation, including failures."""
import argparse
import html
import json
import os
from pathlib import Path

from scripts import issue_76_expansion as files
from scripts import run_issue_76_canonical_smoke as c2
from scripts import run_issue_76_native_continuation as native


def inventory():
    plans = ((c2.OUTPUT, files.read(c2.OUTPUT / "plan.json")),
             (native.OUTPUT, files.read(native.OUTPUT / "plan.json")))
    entries = []
    for output, plan in plans:
        for member in plan["members"]:
            result_path = c2.result_path(output, member)
            result = files.read(result_path) if result_path.exists() else None
            # Publication must not bind files still being written by an active
            # episode. Failed/interrupted attempts are included once recorded.
            if result is None:
                continue
            root = output / "attempts" / member["identity"]
            for capture_root in sorted((root / "aligned").glob("capture-v2:*")):
                frames = []
                for path in sorted(capture_root.glob("frame_*.json")):
                    frame = files.read(path)
                    image = path.with_suffix(".png")
                    if image.is_file():
                        frames.append({"path": str(image), "fixed_step": frame["fixed_step"],
                                       "fixed_time_seconds": frame["fixed_time_seconds"]})
                manifest = capture_root / "native-manifest.json"
                evidence = files.read(manifest) if manifest.exists() else None
                entries.append({"episode": member["identity"], "condition": member["novelty_level"],
                    "capture_id": capture_root.name, "phase": plan["identity"], "frames": frames,
                    "capture_status": evidence["status"] if evidence else "incomplete",
                    "failure": (evidence["failure"] if evidence else None) or (result.get("failure") if result else "episode still running"),
                    "episode_result": str(result_path) if result else None,
                    "native_manifest": str(manifest) if manifest.exists() else None,
                    "native_samples": evidence["sample_count"] if evidence else None,
                    "recorded_seconds": frames[-1]["fixed_time_seconds"] - frames[0]["fixed_time_seconds"] if frames else 0})
    completed = sum(c2.result_path(output, m).exists() for output, p in plans for m in p["members"])
    root = files.ROOT / f"data/issue-76-native-media/completed-episodes-{completed}"
    return root, {"schema": "issue_76_native_retained_media_v2", "entries": entries,
                  "new_captures": 0, "synthetic_frames": 0,
                  "source_text": (files.ROOT / "scripts/issue_76_native_media.py").read_text()}


def page(root, value):
    parts = ["<!doctype html><meta charset='utf-8'><title>Native capture engineering evidence</title>",
             "<style>body{font:16px sans-serif;max-width:900px;margin:30px auto;background:#16191d;color:#eee}img{max-width:100%}section{margin:30px 0;padding:16px;background:#242930}input{width:95%}button{margin:8px}</style>",
             "<h1>C2 and native continuation: retained frames</h1><p>Failures are retained. This is engineering evidence, not model performance. Frame timestamps are actual recorded simulation time. The short C2 fragment plays at 50× slow motion; native 50Hz observations play at nominal recorded cadence. No frames are interpolated or manufactured.</p>"]
    payload = []
    for index, entry in enumerate(value["entries"]):
        frames = entry["frames"]
        parts.append(f"<section><h2>{html.escape(entry['episode'])} — condition {entry['condition']}</h2><p>{html.escape(entry['phase'])}: {html.escape(entry['capture_status'])}; {html.escape(str(entry['failure']))}</p><p>{len(frames)} RGB frames; {entry['recorded_seconds']:.6f}s recorded; native samples: {entry['native_samples']}</p>")
        if frames:
            parts.append(f"<img id='image-{index}' src='{html.escape(os.path.relpath(frames[0]['path'],root))}'><p id='clock-{index}'></p><input id='slider-{index}' type='range' min='0' max='{len(frames)-1}' value='0'><button id='play-{index}'>Play / pause</button>")
        parts.append("</section>")
        payload.append({"frames": [{**f, "path": os.path.relpath(f["path"], root)} for f in frames],
                        "rate": .02 if entry["phase"] == "issue-76-canonical-smoke-v1" else 1})
    parts.append("<script>const entries=" + json.dumps(payload).replace("<", "\\u003c") + ";")
    parts.append("""entries.forEach((entry,id)=>{
      const f=entry.frames;if(!f.length)return;const image=document.getElementById('image-'+id),slider=document.getElementById('slider-'+id),clock=document.getElementById('clock-'+id);
      let playing=false,timer;function show(i){slider.value=i;image.src=f[i].path;clock.textContent='Frame '+(i+1)+'/'+f.length+' | native step '+f[i].fixed_step+' | t='+f[i].fixed_time_seconds.toFixed(6)+'s | playback '+entry.rate+'×';}
      function next(){let i=Number(slider.value);if(!playing)return;if(i===f.length-1){playing=false;return;}timer=setTimeout(()=>{show(i+1);next();},Math.max(1,1000*(f[i+1].fixed_time_seconds-f[i].fixed_time_seconds)/entry.rate));}
      slider.oninput=()=>{playing=false;clearTimeout(timer);show(Number(slider.value));};document.getElementById('play-'+id).onclick=()=>{playing=!playing;clearTimeout(timer);if(playing){if(Number(slider.value)===f.length-1)show(0);next();}};show(0);
    });</script>""")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--publish", action="store_true")
    modes.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    root, value = inventory()
    rendered = page(root, value)
    if args.publish:
        root.mkdir(parents=True, exist_ok=True)
        files.write(root / "media.json", value)
        (root / "index.html").write_text(rendered)
    elif files.read(root / "media.json") != value or (root / "index.html").read_text() != rendered:
        raise ValueError("retained native-frame gallery differs from source evidence")
    print(f"[native-media] {'published' if args.publish else 'validated'} {root/'index.html'}", flush=True)


if __name__ == "__main__":
    main()
