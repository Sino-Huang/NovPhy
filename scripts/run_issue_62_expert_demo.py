"""Run the WebUI against one exact failed issue-62 production lineage."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Mapping

from scripts.capture_issue_53_evidence import _install_level
from scripts.run_issue_62_successor_cohort import (
    DEFAULT_PRODUCTION_PLAN,
    DEFAULT_PRODUCTION_RUNTIME,
    ROOT,
    STAGE_ROOT,
    _load_plan,
)
from scripts.smoke_physics_capture import (
    archive_details,
    free_port,
    start_display,
    terminate,
)
from src.webui.server import AppState, create_handler


EXPERT_DEMO_LINEAGES = (1431, 3176)
DEFAULT_OUTPUT = ROOT / "data/issue-62-expert-demo"


def stage_expert_demo_runtime(
    plan: Mapping[str, Any],
    *,
    production_runtime: Path,
    game: Path,
    lineage_number: int,
) -> dict[str, Any]:
    if not 1 <= lineage_number <= len(plan["lineages"]):
        raise ValueError("expert-demo lineage is outside the frozen plan")
    slot = plan["lineages"][lineage_number - 1]
    source = (
        Path(production_runtime)
        / "attempts"
        / slot["slot_identity"]
        / "attempt-01"
        / "scenario.xml"
    )
    if not source.is_file():
        raise FileNotFoundError(
            f"failed lineage source XML is missing: {source}"
        )
    archive_details(STAGE_ROOT, game)
    _install_level(game, source, slot["slot_identity"])
    return {
        "schema": "issue_62_expert_demo_context_v1",
        "production_plan_identity": plan["identity"],
        "lineage": lineage_number,
        "ordinal": slot["ordinal"],
        "slot_identity": slot["slot_identity"],
        "generation_seed": slot["generation_seed"],
        "exposure_role": slot["exposure_role"],
        "generator_family": slot["generator_family"],
        "behavior_policy": slot["behavior_policy"],
        "source_scenario_xml": str(source.resolve()),
    }


def _start_video(display: str, output: Path) -> subprocess.Popen[bytes]:
    display_input = display if "." in display else f"{display}.0"
    process = subprocess.Popen([
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-f", "x11grab",
        "-draw_mouse", "0",
        "-framerate", "20",
        "-video_size", "1024x768",
        "-i", display_input,
        "-c:v", "libvpx",
        "-deadline", "realtime",
        "-cpu-used", "6",
        "-b:v", "1M",
        str(output),
    ], stdin=subprocess.PIPE, start_new_session=True)
    time.sleep(0.25)
    if process.poll() is not None:
        raise RuntimeError("ffmpeg could not record the private expert-demo display")
    return process


def _stop_video(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    if process.stdin is not None:
        process.stdin.write(b"q\n")
        process.stdin.flush()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lineage", type=int, choices=EXPERT_DEMO_LINEAGES, required=True
    )
    parser.add_argument("--production-plan", type=Path, default=DEFAULT_PRODUCTION_PLAN)
    parser.add_argument(
        "--production-runtime", type=Path, default=DEFAULT_PRODUCTION_RUNTIME
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--speed", type=int, default=1)
    parser.add_argument("--start-display", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = _load_plan(args.production_plan, "production")
    with tempfile.TemporaryDirectory(
        prefix=f"issue62_expert_lineage_{args.lineage}_"
    ) as temporary:
        game = Path(temporary) / "game-runtime"
        context = stage_expert_demo_runtime(
            plan,
            production_runtime=args.production_runtime,
            game=game,
            lineage_number=args.lineage,
        )
        if args.dry_run:
            print(json.dumps({
                **context,
                "game_runtime_staged": True,
                "files_written": False,
            }, indent=2, sort_keys=True), flush=True)
            return 0

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session = Path(args.output).resolve() / f"lineage-{args.lineage:04d}" / timestamp
        session.mkdir(parents=True)
        (session / "context.json").write_text(
            json.dumps(context, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        action_log = session / "actions.jsonl"
        video = session / "expert-demo.webm"
        display_process = None
        recorder = None
        server = None
        app = None
        prior_display = os.environ.get("DISPLAY")
        try:
            if args.start_display:
                display, display_process = start_display(session / "display.log")
                os.environ["DISPLAY"] = display
            elif prior_display is None:
                raise RuntimeError(
                    "no DISPLAY is available; rerun with --start-display"
                )
            if not args.no_video:
                recorder = _start_video(os.environ["DISPLAY"], video)
            ports = []
            while len(ports) < 3:
                candidate = free_port()
                if candidate not in ports:
                    ports.append(candidate)
            agent_port, engine_game_port, physics_port = ports
            app = AppState(
                root=ROOT,
                game_dir_override=game,
                game_port=agent_port,
                physics_port=physics_port,
                engine_game_port=engine_game_port,
                explicit_game_ports=True,
                speed=args.speed,
                manual_action_log=action_log,
                manual_action_context=context,
            )
            server = ThreadingHTTPServer((args.host, args.port), create_handler(app))
            print(f"Issue-62 expert demo: http://{args.host}:{args.port}/", flush=True)
            print(
                f"Exact failed lineage={args.lineage} seed={context['generation_seed']}",
                flush=True,
            )
            print(f"Action log: {action_log}", flush=True)
            if not args.no_video:
                print(f"WebM audit: {video}", flush=True)
            print(
                "Click Start game, drag from the slingshot to demonstrate the shot, "
                "then press Ctrl-C here when finished.",
                flush=True,
            )
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if app is not None:
                app.stop()
            if server is not None:
                server.server_close()
            _stop_video(recorder)
            if display_process is not None:
                print(f"Display stopped: {terminate(display_process)}", flush=True)
            if prior_display is None:
                os.environ.pop("DISPLAY", None)
            else:
                os.environ["DISPLAY"] = prior_display
        print(f"Expert demo saved: {session}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
