#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world_model.training.frontier import FrontierError, UNAVAILABLE_SCOPE, analyze_frontier, canonical_frontier_rows, source_digest

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        source_path = Path(args.input)
        source = source_path.read_bytes()
        rows = canonical_frontier_rows(source, source_path)
        result = analyze_frontier(rows, seed=args.seed)
        digest = source_digest(source); result["source_digest"] = digest
        out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
        (out / "frontier.json").write_bytes(raw)
        (out / "frontier.md").write_text(f"# Temporal Pareto Frontier\n\nSource digest: `{digest}`\n\nScope: {UNAVAILABLE_SCOPE}.\n\nVerdict: **{result['verdict']}**\n\n| Regime | Frontier deltas |\n|---|---|\n" + "".join(f"| {r} | {', '.join(map(str, ds))} |\n" for r, ds in result["frontiers"].items()), encoding="utf-8")
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots();
            for regime, ds in result["frontiers"].items(): ax.plot(ds, range(len(ds)), marker="o", label=regime)
            ax.set(xlabel="delta", ylabel="frontier rank", title=f"Temporal Pareto Frontier\n{UNAVAILABLE_SCOPE}")
            ax.legend()
            metadata = {"Creator": f"NovPhy source_digest={digest}", "Keywords": f"source_digest={digest}; alpha unavailable; physical unavailable"}
            fig.savefig(out / "frontier.svg", metadata=metadata)
            fig.savefig(out / "frontier.pdf", metadata=metadata)
            plt.close(fig)
        except Exception as error: raise FrontierError(f"plotting failed: {error}") from error
        return 0
    except (OSError, json.JSONDecodeError, FrontierError, KeyError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr); return 2
if __name__ == "__main__": raise SystemExit(main())
