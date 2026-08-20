#!/usr/bin/env python3
"""Render the public issue-45 review facts as one local PNG board."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping
from xml.etree import ElementTree as ET

from scripts.cohort_v2_scenarios import (
    validate_central_v2_scenario_inventory_draft,
    validate_deterministic_scenario_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / ".claude/project-docs/evidence/issue-45-cohort-v2-lineage"
DEFAULT_OUTPUT = ROOT / ".local-artifacts/issue-45-review"


def _object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"review artifact is not an object: {path}")
    return value


def _short(identity: str) -> str:
    return identity if len(identity) <= 54 else f"{identity[:31]}…{identity[-18:]}"


def _layout_svg(path: Path) -> str:
    root = ET.fromstring(path.read_bytes())
    objects = []
    for node in root.findall(".//*[@x][@y]"):
        try:
            x = float(node.attrib["x"])
            y = float(node.attrib["y"])
        except ValueError:
            continue
        label = node.attrib.get("type", node.tag)
        material = node.attrib.get("material", "")
        objects.append((node.tag, label, material, x, y, node.attrib))
    width, height = 560, 220
    world_min_x, world_max_x = -15.0, 12.0
    world_min_y, world_max_y = -5.0, 7.0

    def point(x: float, y: float) -> tuple[float, float]:
        return (
            (x - world_min_x) / (world_max_x - world_min_x) * width,
            height - (y - world_min_y) / (world_max_y - world_min_y) * height,
        )

    shapes = [f'<rect width="{width}" height="{height}" rx="14" fill="#071018"/>']
    ground_y = point(0, -3.6)[1]
    shapes.append(f'<path d="M0 {ground_y:.1f} H{width}" stroke="#52636e" stroke-width="3"/>')
    for tag, label, material, x, y, attrs in objects:
        px, py = point(x, y)
        if tag == "Slingshot":
            shapes.append(f'<path d="M{px - 8:.1f} {py + 28:.1f} L{px - 4:.1f} {py - 16:.1f} M{px + 8:.1f} {py + 28:.1f} L{px + 4:.1f} {py - 16:.1f}" stroke="#d7a15d" stroke-width="6"/>')
        elif tag == "Pig":
            shapes.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="11" fill="#78d56b" stroke="#d7ffd1" stroke-width="2"/>')
        elif tag == "Platform":
            scale_x = max(1.0, float(attrs.get("scaleX", 1.0)))
            shapes.append(f'<rect x="{px - 18 * scale_x:.1f}" y="{py - 5:.1f}" width="{36 * scale_x:.1f}" height="10" rx="3" fill="#9ba6ae"/>')
        elif tag == "Block":
            color = {"wood": "#bb7a45", "stone": "#84909b", "ice": "#75c9dd"}.get(material, "#c69b62")
            if "Circle" in label:
                shapes.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="10" fill="{color}"/>')
            else:
                shapes.append(f'<rect x="{px - 11:.1f}" y="{py - 11:.1f}" width="22" height="22" rx="3" fill="{color}"/>')
        shapes.append(f'<text x="{px + 13:.1f}" y="{py - 9:.1f}" fill="#9db2bf" font-size="10">{escape(label)}</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Source layout preview">{"".join(shapes)}</svg>'


def _role_card(entry: Mapping[str, Any], *, evidence_root: Path) -> str:
    role = entry["exposure_role"]
    if role == "final_evaluation":
        preview = '<div class="sealed"><div class="lock">🔒</div><strong>Sealed — intentionally not rendered</strong><span>Opaque reference only</span></div>'
    else:
        xml_name = {
            "training": "training.xml",
            "calibration": "calibration.xml",
            "model_selection": "model-selection.xml",
        }[role]
        preview = _layout_svg(evidence_root / "xml" / xml_name)
    return f"""
      <article class="role-card">
        <div class="role-title"><h2>{escape(role)}</h2><span>{escape(entry['inventory_state'])}</span></div>
        <div class="preview">{preview}</div>
        <dl>
          <dt>Manifest</dt><dd title="{escape(entry['scenario_manifest_identity'])}">{escape(_short(entry['scenario_manifest_identity']))}</dd>
          <dt>Template</dt><dd title="{escape(entry['scenario_template_identity'])}">{escape(_short(entry['scenario_template_identity']))}</dd>
          <dt>Lineage</dt><dd title="{escape(entry['scenario_lineage_identity'])}">{escape(_short(entry['scenario_lineage_identity']))}</dd>
        </dl>
      </article>
    """


def _html(draft: Mapping[str, Any], receipt: Mapping[str, Any], evidence_root: Path) -> str:
    approval = f"APPROVE {draft['identity']}"
    cards = "".join(_role_card(entry, evidence_root=evidence_root) for entry in draft["entries"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}} body{{margin:0;width:1600px;height:1200px;overflow:hidden;background:#081018;color:#edf5f8;font:16px/1.35 system-ui,sans-serif}}
main{{height:100%;padding:42px 52px;background:radial-gradient(circle at 8% 0,#163443 0,transparent 30%),#081018}}
header{{display:flex;justify-content:space-between;gap:30px;align-items:flex-start;margin-bottom:24px}} .eyebrow{{color:#65d6b5;text-transform:uppercase;letter-spacing:.13em;font-size:13px}}
h1{{font-size:50px;letter-spacing:-.045em;margin:5px 0 8px}} .subtitle{{color:#91a7b5;max-width:800px;margin:0}}
.draft{{width:550px;border:1px solid #315064;background:#0f1c26;border-radius:14px;padding:16px}} .draft strong{{display:block;color:#9bf3d7;margin-bottom:7px}} code{{word-break:break-all;color:#c7d8e1;font-size:12px}}
.roles{{display:grid;grid-template-columns:1fr 1fr;gap:18px}} .role-card{{border:1px solid #263746;border-radius:16px;background:#101a23;padding:18px;min-height:360px}}
.role-title{{display:flex;align-items:center;justify-content:space-between}} h2{{margin:0;font-size:20px}} .role-title span{{color:#91a7b5;font-size:12px}}
.preview{{height:170px;margin:12px 0;border-radius:14px;overflow:hidden;background:#071018}} .preview svg{{width:100%;height:100%}}
.sealed{{height:100%;display:grid;place-content:center;text-align:center;color:#91a7b5;gap:7px}} .lock{{font-size:44px}} .sealed strong{{color:#edf5f8}}
dl{{display:grid;grid-template-columns:75px 1fr;gap:5px 10px;margin:0;font-size:12px}} dt{{color:#65d6b5}} dd{{margin:0;color:#b8cad3;font-family:ui-monospace,monospace}}
footer{{display:grid;grid-template-columns:1fr 1.4fr;gap:18px;margin-top:18px}} .fact{{border:1px solid #263746;border-radius:14px;background:#0e1821;padding:16px}}
.fact h3{{margin:0 0 8px;font-size:16px}} .pass{{color:#65d6b5;font-weight:700}} .fact p{{margin:5px 0;color:#91a7b5;font-size:13px}} .approval{{background:#13251f;border-color:#397763}}
.approval code{{display:block;color:#dffcf3;font-size:13px;margin-top:8px}} .notice{{position:absolute;right:54px;bottom:20px;color:#6f8592;font-size:11px}}
</style></head><body><main>
<header><div><div class="eyebrow">NovPhy · Issue #45 review</div><h1>Cohort-v2 lineage approval board</h1><p class="subtitle">Public non-final roles, source-bound identities, deterministic Unity reset evidence, and an opaque final-evaluation projection.</p></div>
<div class="draft"><strong>Draft inventory identity</strong><code>{escape(draft['identity'])}</code></div></header>
<section class="roles">{cards}</section>
<footer><section class="fact"><h3>Unity reset reproduction <span class="pass">PASS</span></h3><p>Two independent captures resolve the same normalized initial engine state.</p><code>{escape(receipt['normalized_initial_engine_state_identity'])}</code><p>Capture byte digests remain distinct and independently recorded.</p></section>
<section class="fact approval"><h3>Approval action</h3><p>After checking the roles, sealed projection, and reset evidence, post this exact comment on GitHub issue #45:</p><code>{escape(approval)}</code></section></footer>
<div class="notice">Layout thumbnails are source-layout review aids. Artifact identities and Unity engine receipts remain the evidence authority.</div>
</main></body></html>"""


def build_review_board(evidence_root: Path, output_dir: Path) -> dict[str, str]:
    evidence_root = Path(evidence_root)
    output_dir = Path(output_dir)
    draft = _object(evidence_root / "inventory/draft.json")
    receipt = _object(evidence_root / "receipts/training-unity-reset.json")
    validate_central_v2_scenario_inventory_draft(draft, manifest_root=evidence_root / "manifests")
    validate_deterministic_scenario_receipt(receipt)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "issue-45-review-board.html"
    png_path = output_dir / "issue-45-review-board.png"
    html_path.write_text(_html(draft, receipt, evidence_root), encoding="utf-8")
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        raise RuntimeError("Chrome is required to render the issue-45 review board")
    with tempfile.TemporaryDirectory(prefix="novphy_issue45_chrome_") as profile:
        subprocess.run([
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            f"--user-data-dir={profile}",
            "--window-size=1600,1200",
            f"--screenshot={png_path}",
            html_path.resolve().as_uri(),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"html_path": str(html_path), "png_path": str(png_path), "draft_identity": draft["identity"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the local issue-45 approval review board")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_review_board(args.evidence_root, args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
