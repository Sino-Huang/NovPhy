"""Issue #128 A3: digitise Phy-Q's per-template within-template pass-rate bar chart (Supp. Fig., main.tex l. 2266-2271).

    python scripts/b127_phyq_digitise_within_template.py      # novphy env, CPU, no network

The paper publishes per-template local-generalisation pass rates only as the bar chart
``Fig_within_template_submission_2_avg_pass_rates.PNG`` (read straight from the arXiv v3 source tarball). Method:

1. Legend: the 9 swatches (uniform colour blocks in the legend strip, read row-major = legend order) give each agent's
   exact RGB; bars are drawn in exactly these colours.
2. Panels: the black left/bottom spines locate the 15 panels (row-major = scenario order 1..15). The y=0 and y=1 tick
   marks left of the left spine give the y scale (darkness-weighted row centroids); the x tick marks below the bottom
   spine give the template slot centres x_1..x_n.
3. Bars: the 9 bars of a slot sit at tick + offset + (i - 4) * pitch; offset and pitch are fitted per panel to the
   column centres of exact-colour pixel runs. At each bar's central (fully covered) pixel column the per-row colour
   coverage alpha = <background - pixel, white - colour> / |white - colour|^2 is taken over the bar-free background
   of that row (white, or the bottom spine's anti-aliased grey; the spine is drawn over the bars). The top edge is
   the first fully covered row's upper boundary minus the partial coverage above it; bars with no fully covered row
   (< ~2 px) use the panel's median bar base (top + total coverage of the other bars) minus their total coverage.
   value = (y0_tick - top) / (y0_tick - y1_tick). Total coverage < ABSENT_PX px = 'absent' (value 0);
   value < NEAR_ZERO = 'present_near_zero'. The bar base itself sits under the spine and is not used.
4. Slot -> template mapping: per scenario, every permutation of the scenario's templates (Parameters.py
   ``capability_templates_dict`` key order = scenarios 1..15; the 75-template set = Bambirds' recomputed templates) is
   scored by the squared error between the four heuristic agents' digitised bars and their per-template pass rates
   recomputed from the released playdata (harness-check.json). A slot is 'data_clear' when every permutation within a
   likelihood ratio of CLEAR_LR of the best (Gaussian readout noise, sigma pooled over all best-fit residuals) assigns
   it the same template. The README label numbering (repo xlsx; figure label x_y = README label x.y) resolves the
   remaining slots when it agrees with every data-clear slot and lies inside the data tie set.
5. The heuristic residuals under the final mapping are the measured readout uncertainty (raw). A linear calibration
   truth = a + b * raw fitted on the heuristic bars drawn in the figure gives 'calibrated' values, its error measured
   leave-one-scenario-out. Relational / D-DDQN bars are read with the same procedure; their distance to the nearest
   k/20 is an independent check. Fidelity-target candidates are ranked (not chosen) by readability.
"""
from __future__ import annotations

import ast
import io
import itertools
import json
import math
import re
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b127_phyq_harness_check import MAPPING_XLSX, OUT_DIR, PAPER_TARBALL, ROOT, SCENARIOS, read_xlsx_rows, sha256  # noqa: E402

FIG_MEMBER = "Fig_within_template_submission_2_avg_pass_rates.PNG"
HARNESS_JSON = OUT_DIR / "harness-check.json"
PARAMETERS_PY = ROOT / "world_model/baselines/phyq/vendor/Utils/Parameters.py"
OUT_JSON = OUT_DIR / "within-template-digitised.json"
OUT_MD = OUT_DIR / "within-template-digitised.md"

LEGEND = ["Bambirds", "Eagle's Wing", "Datalab", "Pig Shooter", "D-DDQN-Symbolic", "D-DDQN-Image",
          "Relational-Symbolic", "Relational-Image", "Random Agent"]
HEURISTIC = LEGEND[:4]
RELATIONAL = ["Relational-Symbolic", "Relational-Image"]
SLOTS_PER_PANEL = [5, 5, 6, 5, 5, 6, 5, 4, 4, 4, 5, 6, 5, 2, 8]  # x tick labels printed in the figure

BLACK_MAX = 80        # spine pixels are (near-)black; anti-aliased spine edges are lighter
SPINE_MIN_RUN = 40    # px, vertical spine run length
TICK_DARKNESS = 8     # mean darkness (255 - grey) above which a tick-band row/column belongs to a tick
ABSENT_PX = 0.3       # summed coverage below this = no bar drawn
NEAR_ZERO = 0.02      # value below this = 'present_near_zero'
CLEAR_LR = 100.0      # likelihood ratio separating the best mapping from alternatives
PREFERRED = (0.3, 0.9)


def darkness(rgb):
    return 255.0 - rgb.mean(-1)


def runs(mask):
    """[(start, end_inclusive)] of True runs in a 1-D bool array."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.nonzero(d == 1)[0].tolist(), (np.nonzero(d == -1)[0] - 1).tolist()))


def centroid(weights, start):
    w = np.asarray(weights, float)
    return float(start + (w * np.arange(len(w))).sum() / w.sum())


# ------------------------------------------------------------------------------------------------ inputs
def load_inputs():
    tar_bytes = PAPER_TARBALL.read_bytes()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        fig_bytes = tar.extractfile(FIG_MEMBER).read()
    harness_bytes = HARNESS_JSON.read_bytes()
    params_bytes = PARAMETERS_PY.read_bytes()
    xlsx_bytes = MAPPING_XLSX.read_bytes()
    hashes = {
        f"data/external/phyq/{PAPER_TARBALL.name}": sha256(tar_bytes),
        f"data/external/phyq/{PAPER_TARBALL.name}:{FIG_MEMBER}": sha256(fig_bytes),
        str(HARNESS_JSON.relative_to(ROOT)): sha256(harness_bytes),
        str(PARAMETERS_PY.relative_to(ROOT)): sha256(params_bytes),
        "PHYQ_SRC/tasks/" + MAPPING_XLSX.name: sha256(xlsx_bytes),
    }
    img = Image.open(io.BytesIO(fig_bytes))
    rgba = np.array(img.convert("RGBA"))
    assert (rgba[..., 3] == 255).all(), "figure has transparency"
    tree = ast.parse(params_bytes.decode())
    cap = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "capability_templates_dict" for t in n.targets))
    return rgba[..., :3].astype(float), img.size, json.loads(harness_bytes), list(cap), hashes


def readme_labels():
    """README label 'm.y' -> template id (same xlsx parse as the harness check; scenario taken from the xml id)."""
    scen_ids = [s for s, _ in SCENARIOS]
    out = {}
    for row in read_xlsx_rows(MAPPING_XLSX):
        if len(row) < 2 or not re.fullmatch(r"\d+_\d+_\d+_1", row[1].strip()):
            continue
        a, b, c, _ = row[1].strip().split("_")
        tid = f"{a}_{int(b):02d}_{int(c):02d}"
        y = int(f"{float(row[0]):.1f}".split(".")[1])
        out[f"{scen_ids.index(tid[:4]) + 1:02d}_{y}"] = tid
    return out


# ------------------------------------------------------------------------------------------------ geometry
def legend_colours(rgb, legend_top):
    """Uniform non-white, non-grey-text colour blocks in the legend strip, row-major."""
    strip = rgb[legend_top:].astype(int)
    key = strip[..., 0] * 65536 + strip[..., 1] * 256 + strip[..., 2]
    vals, counts = np.unique(key, return_counts=True)
    swatches = []
    for v, n in zip(vals, counts):
        col = (int(v) // 65536, int(v) // 256 % 256, int(v) % 256)
        if n < 80 or col == (255, 255, 255) or col == (0, 0, 0):
            continue
        ys, xs = np.nonzero(key == v)
        # a swatch is a filled square: its pixels fill their bounding box
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        if n >= 0.9 * h * w and 8 <= h <= 16:
            swatches.append({"rgb": list(col), "bbox_xyxy": [int(xs.min()), int(ys.min() + legend_top),
                                                              int(xs.max()), int(ys.max() + legend_top)]})
    swatches.sort(key=lambda s: (round(s["bbox_xyxy"][1] / 10), s["bbox_xyxy"][0]))
    assert len(swatches) == len(LEGEND), swatches
    return dict(zip(LEGEND, swatches))


def find_panels(rgb):
    black = rgb.max(-1) <= BLACK_MAX
    H, W = black.shape
    spine_cols = [x for x in range(W) if any(e - s + 1 >= SPINE_MIN_RUN for s, e in runs(black[:, x]))]
    col_groups = runs(np.isin(np.arange(W), spine_cols))
    panels = []
    for gx0, gx1 in col_groups:
        for top, bottom in (r for r in runs(black[:, gx0]) if r[1] - r[0] + 1 >= SPINE_MIN_RUN):
            # bottom spine: rows (incl. anti-aliased ones) where most of the panel width is grey-dark (bars are
            # coloured; the grey Eagle's Wing bars cover < 1/9 of the width)
            win = rgb[bottom - 4:bottom + 5, gx1 + 1:gx1 + 201]
            frac = ((darkness(win) > 25) & (win.max(-1) - win.min(-1) < 30)).mean(1)
            spine_rows = [bottom - 4 + i for i in np.nonzero(frac > 0.5)[0].tolist()]
            core = bottom - 4 + int(np.argmax(darkness(win).mean(1)))
            x_right = gx1 + 1 + int(np.argmin(black[core, gx1 + 1:]))
            panels.append({"left_spine_x": [gx0, gx1], "top_y": top,
                           "bottom_spine_rows": [spine_rows[0], spine_rows[-1]], "x_right": x_right})
    tops = [p["top_y"] for p in panels]  # panels of one figure row may differ by a pixel in top_y
    panels.sort(key=lambda p: (min(t for t in tops if abs(t - p["top_y"]) <= 10), p["left_spine_x"][0]))
    assert len(panels) == 15, len(panels)
    return panels


def panel_axes(rgb, p):
    gx0, gx1 = p["left_spine_x"]
    sb0, sb1 = p["bottom_spine_rows"]
    dk = darkness(rgb)
    # y ticks: band of columns just left of the left spine
    band = dk[p["top_y"] - 4:sb1 + 4, gx0 - 5:gx0 - 1].mean(1)
    blobs = [r for r in runs(band > TICK_DARKNESS)]
    assert len(blobs) == 2, blobs
    y1, y0 = (centroid(band[s:e + 1], s + p["top_y"] - 4) for s, e in blobs)
    # bottom spine centre (reported only: y=0 is the tick)
    col = dk[sb0 - 3:sb1 + 3, gx1 + 3]
    spine_c = centroid(col, sb0 - 3)
    # x ticks below the bottom spine
    xb = dk[sb1 + 1:sb1 + 5, gx1 + 1:p["x_right"]].mean(0)
    ticks = [centroid(xb[s:e + 1], s + gx1 + 1) for s, e in runs(xb > 30)]
    # background (bar-free) column right after the left spine; last row above the spine's anti-aliasing
    bg_col = gx1 + 2
    bg = rgb[:, bg_col].mean(-1)
    last_clear = sb0 - 1
    while bg[last_clear] < 254.5:
        last_clear -= 1
    assert (bg[int(y1) - 3:last_clear + 1] >= 254.5).all()
    return {"y0_tick": y0, "y1_tick": y1, "px_per_unit": y0 - y1, "bottom_spine_centre": spine_c,
            "x_ticks": ticks, "bg_col": bg_col, "last_clear_row": last_clear}


def fit_bar_grid(rgb, p, ax, colours):
    """Per-panel offset/pitch of bar centres relative to slot ticks from exact-colour pixel runs."""
    gx1 = p["left_spine_x"][1]
    y_lo, y_hi = int(math.floor(ax["y1_tick"])) - 3, p["bottom_spine_rows"][0]
    region = rgb[y_lo:y_hi, gx1 + 1:p["x_right"]]
    ticks = np.array(ax["x_ticks"])
    slot_pitch = float(np.diff(ticks).mean()) if len(ticks) > 1 else None
    rows, rhs = [], []
    for i, agent in enumerate(LEGEND):
        m = (region == np.array(colours[agent]["rgb"], float)).all(-1).any(0)
        for s, e in runs(m):
            c = (s + e) / 2 + gx1 + 1
            j = int(np.argmin(np.abs(ticks - c)))
            rows.append([1.0, i - 4.0])
            rhs.append(c - ticks[j])
    A, b = np.array(rows), np.array(rhs)
    (off, pitch), *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ np.array([off, pitch])
    return {"offset_px": float(off), "bar_pitch_px": float(pitch), "slot_pitch_px": slot_pitch,
            "n_runs": len(b), "run_centre_resid_max_px": float(np.abs(resid).max())}


def read_bar(rgb, p, ax, grid, j, i, colour):
    xc = ax["x_ticks"][j] + grid["offset_px"] + (i - 4) * grid["bar_pitch_px"]
    x = int(round(xc))
    c = np.array(colour, float)
    span = 255.0 - c
    bg = rgb[:, ax["bg_col"]]  # bar-free column: white, or the spine's anti-aliased grey in the spine rows
    r_ref = ax["last_clear_row"]
    y_start = int(math.floor(ax["y1_tick"])) - 3
    alpha, pure_rows, full_rows = {}, 0, 0
    for y in range(y_start, p["bottom_spine_rows"][1] + 1):
        if bg[y].mean() < 20:  # spine core: bar hidden
            break
        # coverage of the bar colour over this row's background (the spine is drawn over the bars)
        alpha[y] = float(np.clip(((bg[y] - rgb[y, x]) @ span) / (span @ span), 0.0, 1.0))
        if y <= r_ref and alpha[y] > 0.98:
            full_rows += 1
            pure_rows += int(np.abs(rgb[y, x] - c).max() <= 6)
    total = sum(alpha.values())
    full = [y for y, a in alpha.items() if a > 0.98]
    # top edge from the bar's own top: the first fully covered row minus the partial coverage above it. Bars with
    # no fully covered row (< ~2 px) get top = panel bar base - total coverage (base fitted on the other bars).
    top = full[0] - 0.5 - sum(a for y, a in alpha.items() if y < full[0]) if full else None
    return {"status": "absent" if total < ABSENT_PX else "present", "top_y": top,
            "coverage_px": round(total, 3), "alpha_last_clear_row": round(alpha[r_ref], 3),
            "column_x": x, "centre_x": round(xc, 2),
            "purity": round(pure_rows / full_rows, 3) if full_rows else None}


def finish_bars(bars, ax):
    """Bar base per panel (median top + coverage over bars with a fully covered row), then values."""
    tall = [b for b in bars if b["top_y"] is not None and b["status"] == "present"]
    base = float(np.median([b["top_y"] + b["coverage_px"] for b in tall]))
    for b in bars:
        if b["top_y"] is None:
            b["top_y"] = base - b["coverage_px"]
        v = (ax["y0_tick"] - b["top_y"]) / ax["px_per_unit"]
        if b["status"] == "present" and v < NEAR_ZERO:
            b["status"] = "present_near_zero"
        b["value"] = 0.0 if b["status"] == "absent" else round(v, 4)
        b["top_y"] = round(b["top_y"], 3)
    return {"n_bars_with_full_row": len(tall), "bar_base_y_median": round(base, 3),
            "bar_base_minus_y0_tick_px": round(base - ax["y0_tick"], 3),
            "alpha_last_clear_row_min_over_those": min(b["alpha_last_clear_row"] for b in tall)}


# ------------------------------------------------------------------------------------------------ mapping
def permutation_costs(cost):
    n = cost.shape[0]
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int16)
    return perms, cost[np.arange(n), perms].sum(1)


def main():
    rgb, size, harness, cap_keys, hashes = load_inputs()
    scen_ids = [s for s, _ in SCENARIOS]
    assert cap_keys == scen_ids, (cap_keys, scen_ids)
    gt = {a: harness["heuristic"][a]["template_pass_rate"] for a in HEURISTIC}
    templates75 = sorted(gt["Bambirds"])
    per_t = harness["native_task_fidelity"]["per_template"]

    panels = find_panels(rgb)
    legend_top = max(p["bottom_spine_rows"][1] for p in panels) + 40
    colours = legend_colours(rgb, legend_top)

    slots, panel_out = {}, []
    for k, p in enumerate(panels):
        ax = panel_axes(rgb, p)
        assert len(ax["x_ticks"]) == SLOTS_PER_PANEL[k], (k, ax["x_ticks"])
        grid = fit_bar_grid(rgb, p, ax, colours)
        panel_out.append({"scenario": k + 1, "scenario_id": scen_ids[k], "name": SCENARIOS[k][1], **p,
                          **{kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in ax.items()
                             if kk != "x_ticks"},
                          "x_ticks": [round(t, 2) for t in ax["x_ticks"]],
                          **{kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in grid.items()}})
        mine = {}
        for j in range(len(ax["x_ticks"])):
            mine[f"{k + 1:02d}_{j + 1}"] = {
                "scenario": k + 1, "bars": {a: read_bar(rgb, p, ax, grid, j, i, colours[a]["rgb"])
                                            for i, a in enumerate(LEGEND)}}
        panel_out[-1].update(finish_bars([b for s in mine.values() for b in s["bars"].values()], ax))
        slots.update(mine)

    # slot -> template mapping per scenario
    by_scen = {}
    for k, sid in enumerate(scen_ids):
        sl = [s for s in slots if slots[s]["scenario"] == k + 1]
        tl = [t for t in templates75 if t.startswith(sid + "_")]
        assert len(sl) == len(tl), (sid, sl, tl)
        dig = np.array([[slots[s]["bars"][a]["value"] for a in HEURISTIC] for s in sl])
        tru = np.array([[gt[a][t] for a in HEURISTIC] for t in tl])
        cost = ((dig[:, None, :] - tru[None, :, :]) ** 2).sum(-1)
        perms, tot = permutation_costs(cost)
        order = np.lexsort((np.arange(len(tot)), tot))
        by_scen[sid] = (sl, tl, perms, tot, order)
    n_bars = 4 * len(templates75)
    sse_best = sum(float(v[3][v[4][0]]) for v in by_scen.values())
    sigma = math.sqrt(sse_best / n_bars)
    thresh = 2 * sigma ** 2 * math.log(CLEAR_LR)

    labels = readme_labels()
    mapping, slot_to_t = {}, {}
    for k, sid in enumerate(scen_ids):
        sl, tl, perms, tot, order = by_scen[sid]
        best = perms[order[0]]
        near = perms[tot <= tot[order[0]] + thresh]
        clear = {s: bool((near[:, jj] == best[jj]).all()) for jj, s in enumerate(sl)}
        second = float(tot[order[1]]) if len(order) > 1 else None
        m_best = {s: tl[best[jj]] for jj, s in enumerate(sl)}
        m_readme = {s: labels[s] for s in sl}
        readme_perm = np.array([tl.index(m_readme[s]) for s in sl])
        readme_within = bool((near == readme_perm).all(1).any())
        # Final: the README label numbering where it agrees with every clear slot and is within the data tie set
        # (it then only resolves slots the heuristic bars cannot tell apart); else the best permutation.
        use_readme = readme_within and all(m_readme[s] == m_best[s] for s in sl if clear[s])
        m = m_readme if use_readme else m_best
        slot_to_t.update(m)
        mapping[sid] = {
            "scenario": k + 1, "name": SCENARIOS[k][1], "slot_to_template": m,
            "basis": {s: ("data_clear" if clear[s] else
                          "readme_label_within_data_tie" if use_readme else "data_best_unclear") for s in sl},
            "sorted_order": [m[s] for s in sl] == tl,
            "best_slot_to_template": m_best,
            "best_sse": round(float(tot[order[0]]), 5),
            "best_rmse_per_bar": round(math.sqrt(float(tot[order[0]]) / (4 * len(sl))), 4),
            "second_best_sse": None if second is None else round(second, 5),
            "second_best_slot_to_template": None if second is None else
            {s: tl[perms[order[1]][jj]] for jj, s in enumerate(sl)},
            "gap_sse": None if second is None else round(second - float(tot[order[0]]), 5),
            "log_likelihood_ratio_best_vs_second": None if second is None else
            round((second - float(tot[order[0]])) / (2 * sigma ** 2), 2),
            "n_permutations_within_threshold": int(len(near)),
            "unique": bool(len(near) == 1),
            "slot_clear": clear,
            "readme_label_slot_to_template": m_readme,
            "readme_label_sse": round(float(tot[(perms == readme_perm).all(1)][0]), 5),
            "readme_label_within_threshold": readme_within,
            "readme_label_agrees_with_best": {s: m_readme[s] == m_best[s] for s in sl},
        }

    def stats(e):
        e = np.asarray(e)
        ae = np.abs(e)
        return {"n": int(len(e)), "mean_abs": round(float(ae.mean()), 4), "max_abs": round(float(ae.max()), 4),
                "p95_abs": round(float(np.percentile(ae, 95)), 4), "mean_signed": round(float(e.mean()), 4),
                "rmse": round(float(np.sqrt((e ** 2).mean())), 4)}

    # readout error over heuristic bars (final mapping)
    rows = [{"agent": a, "slot": s, "template": t, "scenario": t[:4], "digitised": slots[s]["bars"][a]["value"],
             "truth": gt[a][t], "present": slots[s]["bars"][a]["status"] != "absent",
             "clear": mapping[t[:4]]["slot_clear"][s]} for s, t in slot_to_t.items() for a in HEURISTIC]
    d_all = np.array([r["digitised"] for r in rows])
    t_all = np.array([r["truth"] for r in rows])
    pres = np.array([r["present"] for r in rows])
    clear_m = np.array([r["clear"] for r in rows])
    big = t_all >= 0.1
    err = d_all - t_all
    # linear calibration truth = a + b * digitised on bars drawn in the figure; error by leave-one-scenario-out
    slope, intercept = np.polyfit(d_all[pres], t_all[pres], 1)
    cal_loso = d_all.copy()
    scen_of = np.array([r["scenario"] for r in rows])
    for sid in scen_ids:
        tr, te = pres & (scen_of != sid), pres & (scen_of == sid)
        b_, a_ = np.polyfit(d_all[tr], t_all[tr], 1)
        cal_loso[te] = a_ + b_ * d_all[te]
    err_cal = cal_loso - t_all
    worst = sorted(rows, key=lambda r: -abs(r["digitised"] - r["truth"]))[:10]
    readout = {
        "all_heuristic_bars": stats(err),
        "per_agent": {a: stats(err[[r["agent"] == a for r in rows]]) for a in HEURISTIC},
        "bars_present_in_figure": stats(err[pres]),
        "bars_truth_ge_0.1": stats(err[big]),
        "bars_absent_in_figure": {"n": int((~pres).sum()), "max_true_value": round(float(t_all[~pres].max()), 4)},
        "slots_with_clear_mapping": stats(err[clear_m]),
        "sigma_pooled_for_mapping": round(sigma, 4),
        "linear_calibration": {
            "fit_on": "heuristic bars drawn in the figure (status != absent), all scenarios",
            "slope": round(float(slope), 5), "intercept": round(float(intercept), 5),
            "loso_error_all_bars": stats(err_cal), "loso_error_bars_present": stats(err_cal[pres]),
            "loso_error_bars_truth_ge_0.1": stats(err_cal[big])},
        "largest_errors": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                            if k in ("agent", "slot", "template", "digitised", "truth")} for r in worst],
    }
    unc_raw = readout["bars_truth_ge_0.1"]["p95_abs"]
    unc_cal = readout["linear_calibration"]["loso_error_bars_truth_ge_0.1"]["p95_abs"]

    def cal(v):
        return round(float(intercept + slope * v), 4) if v > 0 else 0.0

    # learning agents: pass rates over 20 test tasks would sit on the k/20 grid (independent accuracy check)
    grid_check = {}
    for a in LEGEND[4:8]:
        v = np.array([slots[s]["bars"][a]["value"] for s in slot_to_t if slots[s]["bars"][a]["status"] == "present"])
        cv = intercept + slope * v
        grid_check[a] = {"n_bars_present": int(len(v)),
                         "raw_minus_nearest_k_over_20": stats(v - np.round(v * 20) / 20),
                         "calibrated_minus_nearest_k_over_20": stats(cv - np.round(cv * 20) / 20)}

    # per-template readout of learning agents
    templates = {}
    for s, t in sorted(slot_to_t.items(), key=lambda kv: kv[1]):
        b = slots[s]["bars"]
        pt = per_t[t]
        templates[t] = {"slot": s, "mapping_basis": mapping[t[:4]]["basis"][s],
                        **{a: b[a]["value"] for a in LEGEND[4:]},
                        "calibrated": {a: cal(b[a]["value"]) for a in LEGEND[4:]},
                        "status": {a: b[a]["status"] for a in LEGEND[4:]},
                        "purity": {a: b[a]["purity"] for a in RELATIONAL},
                        "readout_uncertainty_p95_abs": {"raw": unc_raw, "calibrated": unc_cal},
                        "birds_per_level_min_max": pt["birds_per_level_min_max"],
                        "shots_upper_code": pt["code"]["shots_upper"]}

    # candidate ranking (record only)
    def key(t):
        r = templates[t]
        v = [r[a] for a in RELATIONAL]
        present2 = all(r["status"][a] == "present" for a in RELATIONAL)
        pure = all((r["purity"][a] or 0) >= 0.9 for a in RELATIONAL)
        inrange = all(PREFERRED[0] <= x <= PREFERRED[1] for x in v)
        single = r["birds_per_level_min_max"] == [1, 1]
        margin = min(min(x, 1 - x) for x in v)
        return (not present2, r["mapping_basis"] != "data_clear", not pure, not inrange, not single, -margin, t)

    ranking = sorted(templates, key=key)
    cand = []
    for rank, t in enumerate(ranking, 1):
        r = templates[t]
        kk = key(t)
        cand.append({"rank": rank, "template": t, "slot": r["slot"],
                     "Relational-Symbolic": r["Relational-Symbolic"], "Relational-Image": r["Relational-Image"],
                     "calibrated": {a: r["calibrated"][a] for a in RELATIONAL},
                     "D-DDQN-Symbolic": r["D-DDQN-Symbolic"], "D-DDQN-Image": r["D-DDQN-Image"],
                     "both_present": not kk[0], "slot_mapping_data_clear": not kk[1], "bars_pure": not kk[2],
                     "both_in_0.3_0.9": not kk[3], "single_bird": not kk[4], "margin_from_0_or_1": round(-kk[5], 4),
                     "readout_uncertainty_p95_abs": r["readout_uncertainty_p95_abs"],
                     "birds_per_level_min_max": r["birds_per_level_min_max"],
                     "shots_upper_code": r["shots_upper_code"]})

    out = {
        "schema": "novphy.issue128.a3.phyq_within_template_digitised.v1",
        "source": {"paper": "arXiv 2108.13696v3 (Xue et al., Nature MI 2023), Supplementary 'Local Generalization "
                            "Results' figure (main.tex lines 2266-2271)", "figure_member": FIG_MEMBER,
                   "figure_size_wh": list(size)},
        "inputs_sha256": hashes,
        "method": __doc__.split("\n\n", 2)[2].strip(),
        "constants": {"BLACK_MAX": BLACK_MAX, "SPINE_MIN_RUN": SPINE_MIN_RUN, "TICK_DARKNESS": TICK_DARKNESS,
                      "ABSENT_PX": ABSENT_PX, "NEAR_ZERO": NEAR_ZERO, "CLEAR_LR": CLEAR_LR,
                      "PREFERRED_RANGE": list(PREFERRED)},
        "ground_truth": "harness-check.json heuristic.<agent>.template_pass_rate (mean over each template's 20 tasks "
                        "of the per-task mean over attempts, recomputed from playdata heuristic_agents.zip)",
        "legend": colours,
        "panels": panel_out,
        "slot_to_template_mapping": mapping,
        "mapping_summary": {
            "n_scenarios_unique": sum(m["unique"] for m in mapping.values()),
            "n_slots_clear": sum(sum(m["slot_clear"].values()) for m in mapping.values()),
            "n_slots": len(slot_to_t),
            "n_scenarios_sorted_order": sum(m["sorted_order"] for m in mapping.values()),
            "n_slots_best_equals_readme_label": sum(sum(m["readme_label_agrees_with_best"].values())
                                                    for m in mapping.values()),
            "n_clear_slots_equal_to_readme_label": sum(
                sum(m["readme_label_agrees_with_best"][s] for s in m["slot_clear"] if m["slot_clear"][s])
                for m in mapping.values()),
            "n_scenarios_readme_label_within_threshold": sum(m["readme_label_within_threshold"]
                                                             for m in mapping.values()),
            "n_slots_by_basis": {b: sum(list(m["basis"].values()).count(b) for m in mapping.values())
                                 for b in ("data_clear", "readme_label_within_data_tie", "data_best_unclear")},
            "clear_threshold_sse": round(thresh, 5),
        },
        "readout_error": readout,
        "learning_agent_k_over_20_check": {
            "note": "[INFERENCE] learning agents are tested once on each template's 20 test tasks, so a single-run "
                    "pass rate is k/20; distance of present bars to the nearest k/20 is an independent accuracy check",
            **grid_check},
        "slots": slots,
        "templates": templates,
        "fidelity_target_candidates": {
            "ranking_key": "both relational bars 'present' > slot mapping data-clear > both bar columns pure (>= 0.9 "
                           "of fully covered rows exact colour) > both values in [0.3, 0.9] > single-bird levels > "
                           "larger min(v, 1 - v) over the two variants > template id",
            "ranking": cand},
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    OUT_MD.write_text(render_md(out))
    print(OUT_MD.read_text())


def render_md(r):
    ro = r["readout_error"]
    ms = r["mapping_summary"]
    lc = ro["linear_calibration"]

    def st(name, s):
        return (f"| {name} | {s['n']} | {s['mean_abs']} | {s['p95_abs']} | {s['max_abs']} | {s['mean_signed']} |")

    lines = [
        "# Phy-Q within-template per-template pass rates, digitised",
        "",
        "Generated by `scripts/b127_phyq_digitise_within_template.py` from the arXiv 2108.13696v3 source tarball "
        f"(`{FIG_MEMBER}`). Full detail in `within-template-digitised.json`.",
        "",
        "## Readout error (heuristic bars vs playdata `template_pass_rate`, final mapping)",
        "",
        "| set | n | mean abs | p95 abs | max abs | mean signed |",
        "|---|---|---|---|---|---|",
        st("all heuristic bars (raw)", ro["all_heuristic_bars"]),
        st("drawn in figure (raw)", ro["bars_present_in_figure"]),
        st("truth >= 0.1 (raw)", ro["bars_truth_ge_0.1"]),
        st("on data-clear slots (raw)", ro["slots_with_clear_mapping"]),
        *[st(f"{a} (raw)", s) for a, s in ro["per_agent"].items()],
        st("all bars, calibrated (LOSO)", lc["loso_error_all_bars"]),
        st("truth >= 0.1, calibrated (LOSO)", lc["loso_error_bars_truth_ge_0.1"]),
        "",
        f"Calibration truth = {lc['intercept']} + {lc['slope']} x raw (fit on drawn heuristic bars; errors "
        f"leave-one-scenario-out). {ro['bars_absent_in_figure']['n']} heuristic bars are not drawn; their true "
        f"values are <= {ro['bars_absent_in_figure']['max_true_value']}. [INFERENCE] The raw under-read (~0.3 px) "
        "matches the light outline drawn around every bar (visible when zoomed), which eats into the coloured top edge.",
        "",
        "Learning-agent bars vs nearest k/20 (single run on 20 test tasks):",
        "",
        "| agent | n | raw mean signed | raw p95 abs | calibrated mean signed | calibrated p95 abs |",
        "|---|---|---|---|---|---|",
    ]
    for a, g in r["learning_agent_k_over_20_check"].items():
        if a == "note":
            continue
        rw, cl = g["raw_minus_nearest_k_over_20"], g["calibrated_minus_nearest_k_over_20"]
        lines.append(f"| {a} | {g['n_bars_present']} | {rw['mean_signed']} | {rw['p95_abs']} | {cl['mean_signed']} | "
                     f"{cl['p95_abs']} |")
    lines += [
        "",
        "## Slot -> template mapping",
        "",
        f"{ms['n_scenarios_unique']}/15 scenarios have a unique best permutation; {ms['n_slots_clear']}/"
        f"{ms['n_slots']} slots are data-clear (every permutation within likelihood ratio "
        f"{r['constants']['CLEAR_LR']:g} of the best agrees); {ms['n_scenarios_sorted_order']}/15 scenarios are in "
        f"sorted template order. The README label numbering equals the best permutation on "
        f"{ms['n_slots_best_equals_readme_label']}/{ms['n_slots']} slots ({ms['n_clear_slots_equal_to_readme_label']}/"
        f"{ms['n_slots_clear']} data-clear) and lies within the tie set in "
        f"{ms['n_scenarios_readme_label_within_threshold']}/15 scenarios. Final basis: {ms['n_slots_by_basis']}.",
        "",
        "| scenario | slots -> templates (final) | best SSE | 2nd SSE | LLR best vs 2nd | data-clear slots |",
        "|---|---|---|---|---|---|",
    ]
    for m in r["slot_to_template_mapping"].values():
        s2t = ", ".join(f"{s}={t}" for s, t in m["slot_to_template"].items())
        nc = sum(m["slot_clear"].values())
        lines.append(f"| {m['scenario']}. {m['name']} | {s2t} | {m['best_sse']} | {m['second_best_sse']} | "
                     f"{m['log_likelihood_ratio_best_vs_second']} | {nc}/{len(m['slot_clear'])} |")
    lines += [
        "",
        "## Fidelity-target candidates (ranked, not chosen)",
        "",
        r["fidelity_target_candidates"]["ranking_key"] + ". +/- = p95 |error| over heuristic bars with truth >= 0.1 "
        "(raw / calibrated leave-one-scenario-out).",
        "",
        "| rank | template | slot | R-Sym raw / cal | R-Img raw / cal | +/- p95 raw / cal | D-DDQN-Sym / Img | "
        "birds/level | shots/run (code) | failed criteria |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in r["fidelity_target_candidates"]["ranking"][:10]:
        flags = ",".join(k for k in ("both_present", "slot_mapping_data_clear", "bars_pure", "both_in_0.3_0.9",
                                     "single_bird") if not c[k]) or "-"
        u = c["readout_uncertainty_p95_abs"]
        lines.append(f"| {c['rank']} | {c['template']} | {c['slot']} | {c['Relational-Symbolic']:.3f} / "
                     f"{c['calibrated']['Relational-Symbolic']:.3f} | {c['Relational-Image']:.3f} / "
                     f"{c['calibrated']['Relational-Image']:.3f} | {u['raw']} / {u['calibrated']} | "
                     f"{c['D-DDQN-Symbolic']:.3f} / {c['D-DDQN-Image']:.3f} | {c['birds_per_level_min_max']} | "
                     f"{c['shots_upper_code']:g} | {flags} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
