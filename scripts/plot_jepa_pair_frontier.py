#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world_model.training.frontier import FrontierError, analyze_frontier

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        source = Path(args.input).read_bytes(); payload = json.loads(source)
        rows = payload.get("states", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list): raise FrontierError("input must contain a states list")
        result = analyze_frontier(rows, seed=args.seed)
        digest = hashlib.sha256(source).hexdigest(); result["source_digest"] = digest
        out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
        (out / "frontier.json").write_bytes(raw)
        (out / "frontier.md").write_text(f"# Temporal Pareto Frontier\n\nSource digest: `{digest}`\n\nVerdict: **{result['verdict']}**\n\n| Regime | Frontier deltas |\n|---|---|\n" + "".join(f"| {r} | {', '.join(map(str, ds))} |\n" for r, ds in result["frontiers"].items()), encoding="utf-8")
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots();
            for regime, ds in result["frontiers"].items(): ax.plot(ds, range(len(ds)), marker="o", label=regime)
            ax.set(xlabel="delta", ylabel="frontier rank"); ax.legend(); fig.savefig(out / "frontier.svg"); fig.savefig(out / "frontier.pdf"); plt.close(fig)
        except Exception as error: raise FrontierError(f"plotting failed: {error}") from error
        return 0
    except (OSError, json.JSONDecodeError, FrontierError, KeyError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr); return 2
if __name__ == "__main__": raise SystemExit(main())
