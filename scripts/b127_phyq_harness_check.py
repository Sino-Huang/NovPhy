"""Issue #128 A3 harness check: recompute the Phy-Q paper's published numbers from the released playdata.

    python scripts/b127_phyq_harness_check.py      # novphy env, CPU, no network

Reads (never extracts into the repo):
- the upstream clone ``PHYQ_SRC`` (phy-q/benchmark @ 514491c): ``playdata/`` (agent and human results, the nested
  ``baseline_agent_data/heuristic_agents.zip``), ``tasks/Mapping of task names ... .xlsx`` (README template
  numbering -> XML template ids), ``sciencebirdslevels/generated_levels/fifth_generation.zip`` (the 75 x 100 task
  XMLs the learning agents train on; used for bird counts) and the training code under ``sciencebirdsagents/``;
- the arXiv 2108.13696v3 LaTeX source tarball ``data/external/phyq/phyq_2108.13696_src.tar.gz`` (see the
  ``MANIFEST.json`` next to it; download timestamps live only there).

Writes ``.local-artifacts/issue-128-direct-predictors-v1/a3/harness-check.{json,md}``. Nothing time-dependent is
written, so reruns are byte-identical.

Published numbers come from three places: the paper text/tables (each value's quoted sentence is asserted to occur
in ``main.tex``), the paper figures (Fig. 4b and Fig. 5 are images; their values were read off the PNGs in the
tarball and are hard-coded below with the PNG's sha256), and the repo's ``playdata`` CSVs (full-precision numbers
released with the paper; ``broad_generalization_all_agents.csv`` is the data behind Fig. 4a).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tarfile
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHYQ_SRC = Path(os.environ.get("PHYQ_SRC", "/mnt/array/sukaih/novphy-baselines-src/benchmark@514491c"))
PLAYDATA = PHYQ_SRC / "playdata"
AGENTS_SRC = PHYQ_SRC / "sciencebirdsagents"
LEVELS_ZIP = PHYQ_SRC / "sciencebirdslevels/generated_levels/fifth_generation.zip"
MAPPING_XLSX = PHYQ_SRC / "tasks/Mapping of task names in README to XML template file names.xlsx"
PAPER_TARBALL = ROOT / "data/external/phyq/phyq_2108.13696_src.tar.gz"
PAPER_MANIFEST = ROOT / "data/external/phyq/MANIFEST.json"
OUT_DIR = ROOT / ".local-artifacts/issue-128-direct-predictors-v1/a3"

CSV_TOL = 1e-9  # playdata CSV values are full-precision floats
SECONDS_PER_EXECUTION_EXAMPLES = (20, 60)

# The 15 scenarios: data capability id -> (paper index, name). Paper index m = position in this list.
SCENARIOS = [
    ("1_01", "Single force"), ("1_02", "Multiple forces"), ("2_01", "Rolling"), ("2_02", "Falling"),
    ("2_03", "Sliding"), ("2_04", "Bouncing"), ("3_01", "Relative weight"), ("3_02", "Relative height"),
    ("3_03", "Relative width"), ("3_04", "Shape difference"), ("3_05", "Non-greedy actions"),
    ("3_06", "Structural analysis"), ("3_07", "Clearing paths"), ("3_08", "Adequate timing"), ("3_09", "Manoeuvring"),
]
SCEN_IDS = [s for s, _ in SCENARIOS]
PHYQ_SCENARIOS = SCEN_IDS[2:]  # the paper excludes scenarios 1 and 2

# Canonical agent names; CSV names in broad_generalization_all_agents.csv; raw heuristic files.
AGENTS = ["Human", "Bambirds", "Eagle's Wing", "Datalab", "Pig Shooter", "D-DDQN-Symbolic", "D-DDQN-Image",
          "Relational-Symbolic", "Relational-Image", "Random"]
CSV_NAME = {"Human": "Human", "Bambirds": "Bambirds", "Eagle's Wing": "Eagle\u2019s Wing", "Datalab": "Datalab",
            "Pig Shooter": "Pig Shooter", "D-DDQN-Symbolic": "D-DDQN-Symbolic", "D-DDQN-Image": "D-DDQN-Image",
            "Relational-Symbolic": "Relational-Symbolic", "Relational-Image": "Relational-Image",
            "Random": "Random Agent"}
HEURISTIC_RAW = {"Bambirds": "Bambirds", "Eagle's Wing": "EagleWings", "Datalab": "DataLab", "Pig Shooter": "Naive"}
LEARNING = ["D-DDQN-Symbolic", "D-DDQN-Image", "Relational-Symbolic", "Relational-Image"]

# ---- published numbers from the paper figures (images; values read off the PNGs in the v3 source tarball) ----
FIG4B_PNG = "Fig4_local_broad_gen.PNG"
FIG4B_LOCAL = {"Bambirds": "0.1536", "Eagle's Wing": "0.1367", "Datalab": "0.1481", "Pig Shooter": "0.1050",
               "D-DDQN-Symbolic": "0.3350", "D-DDQN-Image": "0.2487", "Relational-Symbolic": "0.3220",
               "Relational-Image": "0.2393", "Random": "0.0338"}
FIG4B_BROAD = {"Human": "0.8288", "Bambirds": "0.1776", "Eagle's Wing": "0.1999", "Datalab": "0.1880",
               "Pig Shooter": "0.1282", "D-DDQN-Symbolic": "0.1209", "D-DDQN-Image": "0.0997",
               "Relational-Symbolic": "0.1411", "Relational-Image": "0.0897", "Random": "0.0427"}
FIG5_PNG = "PhyQ_distribution.png"
# Fig. 5 prints only integers under agent icons; the icon -> agent assignment is a visual reading of the legend.
FIG5_PHYQ = {"Bambirds": "14", "Eagle's Wing": "12", "Datalab": "7", "Pig Shooter": "-2", "D-DDQN-Symbolic": "6",
             "D-DDQN-Image": "9", "Relational-Symbolic": "8", "Relational-Image": "6"}
FIG_LOCAL_PER_TEMPLATE_PNG = "Fig_within_template_submission_2_avg_pass_rates.PNG"
FIG_HUMAN_PNG = "Fig_human_charts.PNG"

# ---- published numbers quoted in the paper text/tables; each quote must occur verbatim in main.tex ----
TEXT = {
    "table1_phyq": ("Phy-Q score & 14 & 12 & 7 & -2",
                    {"Bambirds": "14", "Eagle's Wing": "12", "Datalab": "7", "Pig Shooter": "-2"}),
    "scaling": ("we set the scaling factor to 100/$|Z_{random}|$ which is 13.58", "13.58"),
    "broad_excl12": ("Eagle's Wing: 0.1142, Bambirds: 0.1022", {"Eagle's Wing": "0.1142", "Bambirds": "0.1022"}),
    "local_heur": ("averaging $15\\%$, $14\\%$, and $15\\%$ respectively",
                   {"Bambirds": "0.15", "Eagle's Wing": "0.14", "Datalab": "0.15"}),
    "broad_learn": ("The D-DDQN-Symbolic and the D-DDQN-Image agents have an average pass rate of 12\\% and 10\\% "
                    "respectively while Relational-Symbolic and Relational-Image have 14\\% and 9\\% respectively",
                    {"D-DDQN-Symbolic": "0.12", "D-DDQN-Image": "0.10", "Relational-Symbolic": "0.14",
                     "Relational-Image": "0.09"}),
    "human_attempts_time": ("On average, they used 1.86 attempts per task and took 23.73 seconds to think per task",
                            {"attempts": "1.86", "think_seconds": "23.73"}),
    "human_n": ("Error bars represent the 95\\% confidence interval calculated using 20 human participants", "20"),
    "aibirds": ("Competition performance & 0.2727 & 0.2000 & 0.2000 & 0.0000",
                {"Bambirds": "0.2727", "Eagle's Wing": "0.2000", "Datalab": "0.2000", "Pig Shooter": "0.0000"}),
}
QUOTES = {
    "phyq_definition": ["We calculate Phy-Q of an agent using the results of our \\textit{broad generalization} "
                        "evaluation",
                        "When calculating the Phy-Q we exclude the first two scenarios: single force and multiple "
                        "forces",
                        "\\textit{$Z_{agent}$} = $\\frac{1}{|SCENARIO - \\{scenario_1,scenario_2\\}|} "
                        "\\sum_{m=3}^{|SCENARIO|} \\frac{P_{agent,m} - {P_{human,m}}}{\\sigma_{human,m}} $",
                        "\\textit{Phy-Q score$_{agent}$} = $100 + Z_{agent} \\frac{100}{|Z_{random}|}$",
                        "where $P_{n,m}$ is the average pass rate of subject \\textit{n} in the $m^{th}$ scenario. "
                        "$\\sigma_{human,m}$ is the standard deviation of human pass rate in scenario $m$"],
    "human_average_pass_rate": ["The average pass rate is calculated as 100\\% if the player passes at the first "
                                "attempt and if the player passes at the fifth attempt, the pass rate is 20\\%."],
    "heuristic_pass_rate": ["Due to the randomness in the heuristic agents, we allow them to have five attempts per "
                            "task and calculate the task pass rate by averaging the result over these five attempts",
                            "the agents were tested on the same $20\\%$ of the test tasks from each task template "
                            "(1500 tasks in total) used for the D-DDQN evaluation. We report the \\textit{local "
                            "generalization} performance by averaging the pass rates of all templates",
                            "For the \\textit{broad generalization} setting, the same testing templates used for the "
                            "D-DDQN evaluation were used and the within scenario pass rate is calculated by averaging "
                            "over all the tested templates within the scenario."],
    "random_agent": ["We run the random agent 50 times per task and report the average pass rate of these 50 "
                     "attempts."],
    "dqn_local_setup": ["For the \\textit{local generalization} evaluation, we run 10 sampling agents that use the same "
                        "DQN model to collect experiences. Each sampling agent runs on the randomly selected task for "
                        "10 episodes. After the set of experiences is collected, the DQN model is trained for 10 "
                        "epochs with a batch size of 32. We train DQN until it either converges or reaches $N$ update "
                        "steps, where $N$ is the number of training tasks per template divided by 5.",
                        "We train our agent on 80$\\%$ of the tasks of the task template and evaluate on the rest of "
                        "the tasks of the same template.",
                        "At the testing time, the agent runs on each of the testing tasks only once and selects the "
                        "action that has the highest Q-value for a given state."],
    "local_split": ["For learning agents, we take the first 80 tasks from each task template as the training set and "
                    "the last 20 tasks to evaluate the local generalization ability of the agent."],
    "local_target_text": ["While both the symbolic learning agents and both the image learning agents on average pass "
                          "approximately $33\\%$ and $24\\%$ of the test levels respectively"],
    "compute": ["Four instances of vCPUs with 2.5 GHz processors and 16 GB memory were used. NVIDIA T4 GPUs with 16 "
                "GB memory were used in those instances."],
}
CODE_FILES = ["sciencebirdsagents/TrainLearningAgent.py", "sciencebirdsagents/TrainLearningAgent.sh",
              "sciencebirdsagents/Utils/Parameters.py", "sciencebirdsagents/Utils/Config.py",
              "sciencebirdsagents/Utils/LevelSelection.py", "sciencebirdsagents/Utils/GenerateTemplateName.py",
              "sciencebirdsagents/LearningAgents/LearningAgentThread.py",
              "sciencebirdsagents/LearningAgents/DQNDiscreteAgent.py",
              "sciencebirdsagents/LearningAgents/RLNetwork/DQNBase.py",
              "sciencebirdsagents/SBEnvironment/SBEnvironmentWrapper.py", "README.md"]


def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def tol_for(published: str) -> float:
    """Half a unit in the last published digit (plus float slack)."""
    decimals = len(published.split(".")[1]) if "." in published else 0
    return 0.5 * 10.0 ** -decimals + 1e-12


def fl(x):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


class Comparisons:
    def __init__(self):
        self.rows = []

    def add(self, quantity, agent, scenario, recomputed, published, source, *, tol=None, definition="primary",
            note=None):
        """published: str (paper digits; tolerance from its precision) or float (CSV; tol=CSV_TOL) or None."""
        recomputed = fl(recomputed)
        if isinstance(published, str):
            pub_val, tol = float(published), tol_for(published) if tol is None else tol
        else:
            pub_val, tol = fl(published), CSV_TOL if tol is None else tol
        if pub_val is None:
            status = "not_published"
        elif recomputed is None:
            status = "not_recomputable"
        else:
            status = "match" if abs(recomputed - pub_val) <= tol else "mismatch"
        diff = None if pub_val is None or recomputed is None else abs(recomputed - pub_val)
        row = {"quantity": quantity, "agent": agent, "scenario": scenario, "recomputed": recomputed,
               "published": pub_val, "published_text": published if isinstance(published, str) else None,
               "abs_diff": diff, "tolerance": tol if pub_val is not None else None,
               "match": None if status in ("not_published", "not_recomputable") else status == "match",
               "status": status, "definition": definition, "published_source": source}
        if note:
            row["note"] = note
        self.rows.append(row)

    def summary(self, rows):
        out = {k: 0 for k in ("matches", "mismatches", "not_published", "not_recomputable")}
        key = {"match": "matches", "mismatch": "mismatches", "not_published": "not_published",
               "not_recomputable": "not_recomputable"}
        for r in rows:
            out[key[r["status"]]] += 1
        return out


# ------------------------------------------------------------------------------------------------ inputs
def load_inputs():
    hashes = {}
    files = {}
    for name in ["broad_generalization_all_agents.csv", "broad_generalisation_human.csv",
                 "baseline_agent_data.zip", "human_player_data.zip"]:
        data = (PLAYDATA / name).read_bytes()
        files[name] = data
        hashes[f"playdata/{name}"] = sha256(data)
    outer = zipfile.ZipFile(io.BytesIO(files["baseline_agent_data.zip"]))
    nested_bytes = outer.read("baseline_agent_data/heuristic_agents.zip")
    hashes["playdata/baseline_agent_data.zip:baseline_agent_data/heuristic_agents.zip"] = sha256(nested_bytes)
    nested = zipfile.ZipFile(io.BytesIO(nested_bytes))
    for m in sorted(nested.namelist()):
        if not m.endswith("/"):
            hashes[f"playdata/baseline_agent_data.zip:baseline_agent_data/heuristic_agents.zip:{m}"] = \
                sha256(nested.read(m))
    human_zip = zipfile.ZipFile(io.BytesIO(files["human_player_data.zip"]))
    for m in sorted(human_zip.namelist()):
        if not m.endswith("/"):
            hashes[f"playdata/human_player_data.zip:{m}"] = sha256(human_zip.read(m))
    for rel, path in [("tasks/" + MAPPING_XLSX.name, MAPPING_XLSX),
                      ("sciencebirdslevels/generated_levels/fifth_generation.zip", LEVELS_ZIP)]:
        hashes[rel] = sha256(path.read_bytes())
    for rel in CODE_FILES:
        hashes[rel] = sha256((PHYQ_SRC / rel).read_bytes())
    if not PAPER_TARBALL.exists():
        raise SystemExit(f"missing {PAPER_TARBALL}; fetch https://arxiv.org/e-print/2108.13696v3 (see MANIFEST.json)")
    tar_bytes = PAPER_TARBALL.read_bytes()
    hashes["data/external/phyq/" + PAPER_TARBALL.name] = sha256(tar_bytes)
    tar = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz")
    paper = {}
    for member in ["main.tex", FIG4B_PNG, FIG5_PNG, FIG_LOCAL_PER_TEMPLATE_PNG, FIG_HUMAN_PNG]:
        data = tar.extractfile(member).read()
        paper[member] = data
        hashes[f"data/external/phyq/{PAPER_TARBALL.name}:{member}"] = sha256(data)
    return files, nested, human_zip, paper, hashes


def tex_locate(tex_lines, needle):
    """1-based main.tex line of a quote (quotes never span lines here)."""
    for i, line in enumerate(tex_lines, 1):
        if needle in line:
            return i
    raise SystemExit(f"quote not found in main.tex: {needle[:80]!r}")


def section_of(tex_lines, lineno):
    for i in range(lineno - 1, -1, -1):
        m = re.match(r"\\(sub)*section\*?\{([^}]*)\}", tex_lines[i].strip())
        if m:
            return m.group(2)
    return None


# ------------------------------------------------------------------------------------------------ human
def human_recompute(human_zip):
    frames = []
    for m in sorted(human_zip.namelist()):
        if m.endswith(".csv"):
            d = pd.read_csv(io.BytesIO(human_zip.read(m)))
            d["player"] = Path(m).stem
            frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["scen"] = d.scenario.map(lambda i: SCEN_IDS[i - 1])
    # README 'Play Data' item 5: average_rate = (6 - attempts) / 5, 0 if more than 5 attempts.
    rate_formula = d.attempts.map(lambda a: (6 - a) / 5 if a <= 5 else 0.0)
    per_player = d.groupby(["scen", "player"]).average_rate.mean()
    breakdown = d.time_breakdown.map(lambda s: {int(k): int(v) for k, v in re.findall(r"(\d+):\s*(\d+)", s)})
    att = d.attempts
    return {
        "n_players": int(d.player.nunique()),
        "players": sorted(d.player.unique().tolist()),
        "n_task_plays": len(d),
        "tasks_per_scenario_per_player": {s: int(n) for s, n in
                                          (d.groupby("scen").size() / d.player.nunique()).items()},
        "average_rate_formula_violations": int((abs(rate_formula - d.average_rate) > 1e-9).sum()),
        "pass_rate": d.groupby("scen").average_rate.mean().to_dict(),
        "sd_players_ddof0": per_player.groupby("scen").std(ddof=0).to_dict(),
        "sd_players_ddof1": per_player.groupby("scen").std(ddof=1).to_dict(),
        "sd_task_plays_ddof1": d.groupby("scen").average_rate.std(ddof=1).to_dict(),
        "five_attempt_pass_rate": (d.average_rate > 0).groupby(d.scen).mean().to_dict(),
        "attempts_sentinel_100_rows": int((att == 100).sum()),
        "attempts_candidates": {
            "mean_attempts_excluding_sentinel_100": float(att[att < 100].mean()),
            "mean_attempts_all_rows": float(att.mean()),
            "mean_attempts_capped_at_5": float(att.clip(upper=5).mean()),
            "mean_recorded_attempts_len_time_breakdown": float(breakdown.map(len).mean()),
        },
        "think_time_candidates": {
            "mean_total_time_column (first 5 attempts)": float(d.total_time.mean()),
            "mean_sum_time_breakdown_all_attempts": float(breakdown.map(lambda b: sum(b.values())).mean()),
            "mean_over_scenarios_of_mean_sum_time_breakdown": float(
                breakdown.map(lambda b: sum(b.values())).groupby(d.scen).mean().mean()),
        },
    }


# ------------------------------------------------------------------------------------------------ heuristics
def heuristic_raw(nested, raw_name):
    df = pd.read_csv(io.BytesIO(nested.read(raw_name + ".csv")))
    df.columns = [c.strip() for c in df.columns]
    df["lvl"] = df.levelName.str.extract(r"/(\d_\d\d_\d\d_\d+)\.xml")[0]
    df["tmpl"] = df.lvl.str[:7]
    df["scen"] = df.lvl.str[:4]
    df["task"] = df.lvl.str[8:].astype(int)
    df["passed"] = (df.LevelStatus == "Pass").astype(float)
    return df


def heuristic_recompute(nested, templates75, split):
    out = {}
    for agent, raw_name in HEURISTIC_RAW.items():
        df = heuristic_raw(nested, raw_name)
        level_rate = df.groupby(["tmpl", "lvl"]).passed.mean()
        tmpl_rate = level_rate.groupby("tmpl").mean()
        # Attempt number = position in file order: the 10-play Bambirds tasks are two concatenated 5-play runs that
        # reuse EvaluationIndex values, so sorting by EvaluationIndex would interleave them.
        first_pass = df.groupby("lvl", sort=True).passed.apply(
            lambda s: int(np.argmax(s.values)) + 1 if s.values.max() > 0 else np.nan)
        lvl_any = df.groupby(["tmpl", "lvl"]).passed.max()
        tmpl_any = lvl_any.groupby("tmpl").mean()
        tmpl_mean_att = first_pass.groupby(first_pass.index.str[:7]).mean()
        in75 = tmpl_rate[tmpl_rate.index.isin(templates75)]
        test_rate = {s: float(np.mean([tmpl_rate[t] for t in split[s]["test"]])) for s in SCEN_IDS}
        # Alternative: pool the tasks (each task weighs 1) instead of averaging template means.
        lv75 = level_rate[level_rate.index.get_level_values("tmpl").isin(templates75)]
        lv_scen = lv75.index.get_level_values("tmpl").str[:4]
        test_t = {t: s for s in SCEN_IDS for t in split[s]["test"]}
        lv_test = lv75[lv75.index.get_level_values("tmpl").isin(list(test_t))]
        pooled_test = lv_test.groupby(lv_test.index.get_level_values("tmpl").map(test_t)).mean().to_dict()
        tasks_in_last20 = df.drop_duplicates("lvl").task.between(81, 100).mean()
        out[agent] = {
            "raw_file": raw_name + ".csv",
            "n_rows": len(df),
            "n_templates_played": int(df.tmpl.nunique()),
            "templates_outside_the_75": sorted(set(df.tmpl) - set(templates75)),
            "n_tasks_per_template_in_75": {k: int(v) for k, v in
                                           df[df.tmpl.isin(templates75)].groupby("tmpl").lvl.nunique()
                                           .value_counts().sort_index().items()},
            "plays_per_task": {int(k): int(v) for k, v in df.groupby("lvl").size().value_counts().sort_index()
                               .items()},
            "fraction_of_tasks_in_last20_(81-100)": float(tasks_in_last20),
            "template_pass_rate": tmpl_rate.to_dict(),
            "template_pass_rate_any_of_attempts": tmpl_any.to_dict(),
            "template_mean_attempts_to_first_pass": {k: fl(v) for k, v in tmpl_mean_att.items()},
            "scenario_pass_rate_all_templates": in75.groupby(in75.index.str[:4]).mean().to_dict(),
            "scenario_pass_rate_test_templates": test_rate,
            "local_gen_mean_over_75_templates": float(in75.mean()),
            "broad_gen_mean_over_15_scenarios": float(np.mean([test_rate[s] for s in SCEN_IDS])),
            "broad_gen_mean_over_scenarios_3_15": float(np.mean([test_rate[s] for s in PHYQ_SCENARIOS])),
            "pooled_tasks": {
                "scenario_pass_rate_all_templates": lv75.groupby(lv_scen).mean().to_dict(),
                "scenario_pass_rate_test_templates": pooled_test,
                "local_gen_mean_over_tasks_of_75_templates": float(lv75.mean()),
                "broad_gen_mean_over_15_scenarios": float(np.mean([pooled_test[s] for s in SCEN_IDS])),
                "broad_gen_mean_over_scenarios_3_15": float(np.mean([pooled_test[s] for s in PHYQ_SCENARIOS])),
            },
        }
    return out


# ------------------------------------------------------------------------------------------------ splits
def read_xlsx_rows(path):
    z = zipfile.ZipFile(path)
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    shared = [si.findtext(".//m:t", default="", namespaces=ns) if si.find("m:t", ns) is not None else
              "".join(t.text or "" for t in si.iter("{%s}t" % ns["m"]))
              for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.iter("{%s}row" % ns["m"]):
        vals = []
        for c in row.findall("m:c", ns):
            v = c.findtext("m:v", default="", namespaces=ns)
            vals.append(shared[int(v)] if c.get("t") == "s" else v)
        rows.append(vals)
    return rows


def splits(tex_lines, templates75):
    """Paper Table (Supplementary E, README numbering) -> XML ids via the repo's xlsx; vs code and summary CSVs."""
    label_to_id, anomalies = {}, []
    for row in read_xlsx_rows(MAPPING_XLSX):
        if len(row) < 2 or not re.fullmatch(r"\d+_\d+_\d+_1", row[1].strip()):
            continue
        label, xml = f"{float(row[0]):.1f}", row[1].strip()  # labels are numeric cells (e.g. 1.1000000000000001)
        a, b, c, _ = xml.split("_")
        tid = f"{a}_{int(b):02d}_{int(c):02d}"
        m = SCEN_IDS.index(tid[:4]) + 1
        scen_label, tmpl_label = label.split(".")
        if int(scen_label) != m:
            anomalies.append(f"xlsx row '{label} -> {xml}': README scenario number {scen_label} should be {m}")
        label_to_id[f"{m}.{int(tmpl_label)}"] = tid
    start = tex_locate(tex_lines, "Scenario & Training templates & Testing templates")
    table = {}
    for line in tex_lines[start:start + 40]:
        m = re.match(r"\s*(\d+)\. [^&]+&([^&]+)&([^\\]+)\\\\", line)
        if m:
            idx = int(m.group(1))
            table[SCEN_IDS[idx - 1]] = {k: [label_to_id[x.strip()] for x in g.split(",")]
                                        for k, g in (("train", m.group(2)), ("test", m.group(3)))}
        if "\\bottomrule" in line:
            break
    assert len(table) == 15, table
    unmapped = sorted(set(templates75) - set(label_to_id.values()))
    return table, start + 1, label_to_id, anomalies, unmapped


def code_benchmark_split():
    src = (AGENTS_SRC / "TrainLearningAgent.py").read_text()
    block = src[src.index("benchmark_capability_templates_dict = {"):src.index("\ncapability_templates_dict = {")]
    out = {}
    for m in re.finditer(r"'(\d_\d\d)': \{'train': \[([^\]]*)\],\s*'test': \[([^\]]*)\]\}", block):
        out[m.group(1)] = {"train": re.findall(r"'([\d_]+)'", m.group(2)), "test": re.findall(r"'([\d_]+)'", m.group(3))}
    assert len(out) == 15, out
    return out


# ------------------------------------------------------------------------------------------------ Phy-Q
def phyq(P, P_human, sd):
    z = {a: float(np.mean([(P[a][s] - P_human[s]) / sd[s] for s in PHYQ_SCENARIOS])) for a in P}
    scale = 100.0 / abs(z["Random"])
    return {"Z": z, "scaling_factor_100_over_abs_Z_random": scale,
            "phyq": {a: 100.0 + z[a] * scale for a in P}}


# ------------------------------------------------------------------------------------------------ fidelity
def code_constants():
    p = (AGENTS_SRC / "Utils/Parameters.py").read_text()

    def get(key):
        m = re.search(rf"'{key}':\s*([^,#\n]+)", p)
        return m.group(1).strip()

    keys = ["num_update_steps", "num_level_per_agent", "num_worker", "agent", "training_attempts_per_level",
            "memory_size", "train_time_per_ep", "train_time_rise", "train_batch", "gamma", "eps_start", "eps_test",
            "lr", "network", "reward_type", "simulation_speed", "h", "w", "output", "device"]
    vals = {k: get(k) for k in keys}
    t = (AGENTS_SRC / "TrainLearningAgent.py").read_text()
    for needle in ["c.num_update_steps = len(c.train_level_list) // 5 if len(c.train_level_list) >= 5 else 50",
                   "level_per_agent=c.num_level_per_agent if step != 0 else None",
                   "c.training_attempts_per_level if step != 0 else 10).select, EPS_START=c.eps_start",
                   "level_selection_function=LevelSelectionSchema.RepeatPlay(1).select,\n                        "
                   "EPS_START=0",
                   "c.train_time_per_ep = max(1, c.train_time_per_ep - int(step * c.train_time_rise))",
                   "if np.average(winning_rate) > 0.99:",
                   "seed_value = 5123690",
                   "network = c.network(h=c.h, w=c.w, outputs=c.output"]:
        assert needle in t, needle
    cfg = (AGENTS_SRC / "Utils/Config.py").read_text()
    assert "train_percent = 0.8" in cfg
    sh = (AGENTS_SRC / "TrainLearningAgent.sh").read_text()
    assert "for val in ${templates}; do" in sh
    return vals


def bird_counts():
    z = zipfile.ZipFile(LEVELS_ZIP)
    birds = {}
    for n in z.namelist():
        m = re.search(r"/(\d_\d\d_\d\d)_(\d+)\.xml$", n)
        if m:
            data = z.read(n)  # declared utf-16 but stored utf-8 unless a BOM says otherwise
            text = data.decode("utf-16") if data[:2] in (b"\xff\xfe", b"\xfe\xff") else data.decode("utf-8-sig")
            birds.setdefault(m.group(1), {})[int(m.group(2))] = len(re.findall(r"<Bird\s", text))
    return birds


def fidelity(birds, consts, tex_lines):
    n_workers = int(consts["num_worker"])
    lpa = int(consts["num_level_per_agent"])
    attempts = int(consts["training_attempts_per_level"])
    t0, rise = int(consts["train_time_per_ep"]), int(consts["train_time_rise"])
    per_template = {}
    for tid in sorted(birds):
        lv = birds[tid]
        n_levels = len(lv)
        ordered = [lv[k] for k in sorted(lv)]
        n_train = int(n_levels * 0.8)
        train, test = ordered[:n_train], ordered[n_train:]
        n_steps = n_train // 5 if n_train >= 5 else 50
        mean_train = float(np.mean(train))
        code_eps0 = n_train * 10
        code_eps_later = (n_steps - 1) * n_workers * lpa * attempts
        paper_eps = n_steps * n_workers * 10
        per_template[tid] = {
            "n_levels": n_levels, "n_train": n_train, "n_test": len(test), "num_update_steps": n_steps,
            "birds_train_sum": int(sum(train)), "birds_test_sum": int(sum(test)),
            "birds_per_level_min_max": [int(min(ordered)), int(max(ordered))],
            "code": {
                "train_episodes": code_eps0 + code_eps_later, "test_episodes": len(test),
                "shots_lower": code_eps0 + code_eps_later + len(test),
                "shots_upper": 10 * int(sum(train)) + code_eps_later * mean_train + int(sum(test)),
            },
            "paper_text": {
                "train_episodes": paper_eps, "test_episodes": len(test),
                "shots_lower": paper_eps + len(test),
                "shots_upper": paper_eps * mean_train + int(sum(test)),
            },
        }
    epochs, cur = [], t0
    for step in range(16):
        cur = max(1, cur - int(step * rise))
        epochs.append(cur)

    def total(ids, variant):
        lo = sum(per_template[t][variant]["shots_lower"] for t in ids)
        hi = sum(per_template[t][variant]["shots_upper"] for t in ids)
        eps = sum(per_template[t][variant]["train_episodes"] + per_template[t][variant]["test_episodes"]
                  for t in ids)
        return {"templates": len(ids), "episodes": eps, "executions_lower": lo, "executions_upper": hi,
                **{f"worker_hours_at_{s}s": [lo * s / 3600, hi * s / 3600] for s in SECONDS_PER_EXECUTION_EXAMPLES}}

    all75 = sorted(per_template)
    one_per_scen = [sorted(t for t in all75 if t.startswith(s))[0] for s in SCEN_IDS]
    single = "3_02_01"  # first template of relative height (paper template 8.3), any template works the same way
    sets = {}
    for variant in ("code", "paper_text"):
        sets[variant] = {"one_template_" + single: total([single], variant),
                         "one_template_per_scenario": total(one_per_scen, variant),
                         "all_75_templates": total(all75, variant)}
    shots_per_t = {v: [per_template[t][v]["shots_upper"] for t in all75] for v in ("code", "paper_text")}
    p_rel = 0.3220
    return {
        "what_a_run_is": {
            "entry_point": "sciencebirdsagents/TrainLearningAgent.py --template <L_CC_TT> --mode within_template "
                           "--game_version Linux --level_path fifth_generation (one process per template; "
                           "TrainLearningAgent.sh loops over Utils/GenerateTemplateName.py output)",
            "networks": "Utils/Parameters.py 'network': DQNRelationalSymbolic (default) or DQNRelationalImage; agent "
                        "class DQNDiscreteAgent (180 discrete release angles, max stretch)",
            "templates": "75 templates (fifth_generation.zip: 75 dirs x 100 task XMLs); a within-template run "
                         "trains and tests on ONE template",
            "train_test_split": "Utils/Config.py update_level_index: tasks sorted by index, train = first "
                                "int(0.8*100) = 80 (1..80), test = last 20 (81..100); paper Supp. D: 'first 80 ... "
                                "last 20'",
            "num_update_steps": "Parameters.py sets 20 (40 cross-template) but TrainLearningAgent.py line 219 "
                                "overrides it to len(train)//5 = 16 for level_path != 'MattewLevel'; paper: N = "
                                "training tasks per template / 5 = 16",
            "num_worker": consts["num_worker"],
            "num_level_per_agent": consts["num_level_per_agent"] + " (source comment '#10')",
            "training_attempts_per_level": consts["training_attempts_per_level"] + " (source comment '#10')",
            "step_0_collection": "level_per_agent=None and RepeatPlay(10) at step 0: every one of the 80 training "
                                 "tasks is played 10 times (800 episodes), split over the 10 workers (8 tasks each)",
            "steps_1_to_15_collection": "each of the 10 workers plays 1 task sampled from its 8 (random.sample) "
                                        "once: 10 episodes per step",
            "early_stop": "collection loop breaks if the mean winning rate of the step's episodes > 0.99",
            "epochs_per_update_step": epochs,
            "epochs_note": "train_time_per_ep starts at 32 and is reduced cumulatively by step*train_time_rise "
                           "(TrainLearningAgent.py line 224); the paper text says 10 epochs",
            "train_batch": consts["train_batch"], "lr": consts["lr"], "gamma": consts["gamma"],
            "memory": f"PrioritizedReplayMemory({consts['memory_size']}) plus on-disk MemoryDatasetMemmap",
            "eps_start": consts["eps_start"], "eps_floor": consts["eps_test"],
            "reward_type": consts["reward_type"] + " (1 if the task is passed, else 0)",
            "simulation_speed": consts["simulation_speed"] + " (paper text: framework allows up to 50x)",
            "seed": "5123690 (python, numpy, torch); torch deterministic algorithms disabled; one run per template",
            "evaluation": "after training, each of the 20 test tasks is played once (RepeatPlay(1)) greedily "
                          "(EPS_START=0); per-template metric = fraction of the 20 test tasks passed",
            "episode": "one task play: shots until WON/LOST (SBEnvironmentWrapper.step: one shot per step; is_done "
                       "on WON/LOST); shots per episode is between 1 and the task's bird count",
        },
        "release_code_defects_observed_by_reading": [
            "Utils/Parameters.py sets 'agent': 'ppo' (a str) while TrainLearningAgent.py uses c.agent.__name__ and "
            "calls c.agent(...): the released defaults must be edited to DQNDiscreteAgent to run the DQN "
            "baselines [observed in source, not executed]",
            "Utils/Parameters.py 'device' is cuda:0/cuda:1 unconditionally; a CPU-only run needs it changed [observed "
            "in source]",
            "TrainLearningAgent.sh iterates 'for val in ${templates}' (bash array without [@]): only the first "
            "template (1_01_01) runs [observed in source]",
            "TrainLearningAgent.py benchmark split lists '1_01_09' as a 2_01 training template (paper Table S: 3.3 "
            "= 2_01_09) [observed in source]",
        ],
        "published_targets": {
            "per_template": {
                "value": None,
                "location": f"Supplementary D, Fig. 'Passing rates of the agents for the within template "
                            f"evaluation' ({FIG_LOCAL_PER_TEMPLATE_PNG}, main.tex line "
                            f"{tex_locate(tex_lines, FIG_LOCAL_PER_TEMPLATE_PNG)})",
                "note": "bar chart only (75 templates x 9 agents): no numeric values, no SD, no error bars; one run "
                        "per template. Any per-template comparison is a visual bar reading (+-~0.05).",
            },
            "per_scenario_within_template": {"value": None, "note": "not reported (the Fig. S bars are grouped by "
                                                                    "scenario but no scenario means are given)"},
            "all_templates_mean": {
                "Relational-Symbolic": FIG4B_LOCAL["Relational-Symbolic"],
                "Relational-Image": FIG4B_LOCAL["Relational-Image"],
                "location": f"Fig. 4b row 'Local gen.' ({FIG4B_PNG}); text: 'symbolic ... and image learning "
                            f"agents on average pass approximately 33% and 24%' (main.tex line "
                            f"{tex_locate(tex_lines, QUOTES['local_target_text'][0])})",
                "definition": "mean over the 75 templates of the per-template pass rate on the 20 test tasks",
                "sd": "not reported (single run; no seeds or SD anywhere in the paper)",
            },
        },
        "sampling_noise_[INFERENCE]": {
            "per_template_binomial_se_at_p0.322_n20": math.sqrt(p_rel * (1 - p_rel) / 20),
            "all_75_mean_binomial_se_n1500": math.sqrt(p_rel * (1 - p_rel) / 1500),
            "note": "test-set sampling noise only; training-run (seed) variance is unreported and comes on top",
        },
        "execution_counts": {
            "definition": "execution = one engine shot (SBEnvironmentWrapper.step -> ar.shoot); lower bound = 1 "
                          "shot per episode; upper bound = every episode uses all of the task's birds (bird counts "
                          "from fifth_generation.zip; steps 1-15 use the mean bird count of the template's training "
                          "tasks [INFERENCE: uniform random.sample]). Early stopping, stuck-level reloads (>15 "
                          "shots) and level loads are not counted.",
            "variants": {
                "code": "released code as written (step 0: 80 x 10 episodes; steps 1-15: 10 x 1 x 1)",
                "paper_text": "[INFERENCE] the paper's description taken literally: 16 steps x 10 agents x 10 "
                              "episodes",
            },
            "per_template_summary": {v: {"shots_upper_min": min(shots_per_t[v]), "shots_upper_max":
                                         max(shots_per_t[v]), "shots_upper_mean": float(np.mean(shots_per_t[v]))}
                                     for v in shots_per_t},
            "one_template_per_scenario_ids": one_per_scen,
            "sets": sets,
            "smallest_credible_set": "all_75_templates: the only numeric published within-template target is the "
                                     "75-template mean (Fig. 4b). A per-template or one-per-scenario run can only "
                                     "be compared with bar heights read off Fig. S.",
        },
        "seconds_per_execution": None,
        "worker_hours_formula": "worker_hours = executions * seconds_per_execution / 3600 (game-instance hours; "
                                "the 10 workers run in parallel, so wall-clock ~ worker_hours / 10)",
        "per_template": per_template,
    }


# ------------------------------------------------------------------------------------------------ main
def main():
    files, nested, human_zip, paper, hashes = load_inputs()
    tex = paper["main.tex"].decode("utf-8")
    tex_lines = tex.split("\n")
    loc = {}
    for key, (quote, _) in TEXT.items():
        loc[key] = tex_locate(tex_lines, quote)
    quotes = {k: [{"main_tex_line": tex_locate(tex_lines, q), "section": section_of(tex_lines, tex_locate(
        tex_lines, q)), "text": q} for q in qs] for k, qs in QUOTES.items()}
    for png in (FIG4B_PNG, FIG5_PNG):
        tex_locate(tex_lines, png)

    agents_csv = pd.read_csv(io.BytesIO(files["broad_generalization_all_agents.csv"]))
    human_csv = pd.read_csv(io.BytesIO(files["broad_generalisation_human.csv"]))
    agents_csv["scen"] = agents_csv.category.str[:4]
    csv_P = {a: agents_csv[agents_csv.name == CSV_NAME[a]].set_index("scen").pass_rate.to_dict() for a in AGENTS}
    human_csv["scen"] = human_csv.category.map(lambda c: f"{c[0]}_{int(c[2:]):02d}")
    csv_sd = human_csv.set_index("scen").sd_human_pass_rate.to_dict()
    csv_human_from_fail = (1 - human_csv.set_index("scen").fail_rate).to_dict()

    birds = bird_counts()
    templates75 = sorted(birds)
    split_paper, split_line, label_to_id, xlsx_anomalies, unmapped = splits(tex_lines, templates75)
    split_code = code_benchmark_split()
    hum = human_recompute(human_zip)
    heur = heuristic_recompute(nested, templates75, split_paper)

    C = Comparisons()
    fig4a = "playdata/broad_generalization_all_agents.csv (released data behind Fig. 4a)"
    # Human per-scenario pass rate and SD.
    for s in SCEN_IDS:
        C.add("broad_gen_pass_rate", "Human", s, hum["pass_rate"][s], csv_P["Human"][s], fig4a)
        C.add("broad_gen_pass_rate", "Human", s, hum["pass_rate"][s], csv_human_from_fail[s],
              "playdata/broad_generalisation_human.csv (1 - fail_rate)")
        C.add("human_pass_rate_sd", "Human", s, hum["sd_players_ddof0"][s], csv_sd[s],
              "playdata/broad_generalisation_human.csv sd_human_pass_rate",
              definition="primary: population SD (ddof=0) over players of each player's scenario mean average_rate")
        C.add("human_pass_rate_sd", "Human", s, hum["sd_players_ddof1"][s], csv_sd[s],
              "playdata/broad_generalisation_human.csv sd_human_pass_rate",
              definition="alternative: sample SD (ddof=1) over players")
        C.add("human_five_attempt_pass_rate", "Human", s, hum["five_attempt_pass_rate"][s], None,
              f"Supp. F {FIG_HUMAN_PNG} (bar chart only)")
    C.add("broad_gen_mean_15_scenarios", "Human", None, np.mean([hum["pass_rate"][s] for s in SCEN_IDS]),
          FIG4B_BROAD["Human"], f"Fig. 4b 'Broad gen.' ({FIG4B_PNG})")
    C.add("n_human_players", "Human", None, hum["n_players"], TEXT["human_n"][1],
          f"main.tex line {loc['human_n']} (Supp. F caption)", tol=0.0,
          note="human_player_data.zip holds 19 players (001, 003-020; 002 absent); its per-scenario means equal "
               "the released CSVs, so the released human numbers come from these 19")
    att_pub = TEXT["human_attempts_time"][1]
    C.add("human_mean_attempts_per_task", "Human", None,
          hum["attempts_candidates"]["mean_attempts_excluding_sentinel_100"], att_pub["attempts"],
          f"main.tex line {loc['human_attempts_time']}",
          note="other candidate definitions in human.attempts_candidates; none equals 1.86")
    C.add("human_mean_think_seconds_per_task", "Human", None,
          hum["think_time_candidates"]["mean_total_time_column (first 5 attempts)"], att_pub["think_seconds"],
          f"main.tex line {loc['human_attempts_time']}",
          note="other candidate definitions in human.think_time_candidates; none equals 23.73")

    # Heuristic agents.
    for agent, raw_name in HEURISTIC_RAW.items():
        h = heur[agent]
        summ_avg = pd.read_csv(io.BytesIO(nested.read(
            f"all_template_summary/based_on_avg_passing_rate/{raw_name}_summarized_h_c_t.csv"))) \
            .set_index("category").pass_rate
        summ_avg_c = pd.read_csv(io.BytesIO(nested.read(
            f"all_template_summary/based_on_avg_passing_rate/{raw_name}_summarized_h_c.csv"))) \
            .set_index("category").pass_rate
        summ_max = pd.read_csv(io.BytesIO(nested.read(
            f"all_template_summary/based_on_the_max_attempts/{raw_name}_summarized_h_c_t.csv"))) \
            .set_index("category")
        test_c = pd.read_csv(io.BytesIO(nested.read(
            f"test_template_summary/based_on_average_pass_rate/{raw_name}_summarized_h_c.csv"))) \
            .set_index("category").pass_rate
        src_zip = "heuristic_agents.zip:"
        for t in templates75:
            C.add("template_pass_rate", agent, t, h["template_pass_rate"].get(t), summ_avg.get(t),
                  src_zip + f"all_template_summary/based_on_avg_passing_rate/{raw_name}_summarized_h_c_t.csv")
            C.add("template_pass_rate_any_attempt", agent, t, h["template_pass_rate_any_of_attempts"].get(t),
                  summ_max.pass_rate.get(t),
                  src_zip + f"all_template_summary/based_on_the_max_attempts/{raw_name}_summarized_h_c_t.csv")
            if h["template_mean_attempts_to_first_pass"].get(t) is not None or fl(summ_max.mean_attempts.get(t)):
                C.add("template_mean_attempts_to_first_pass", agent, t,  # undefined on both sides if never passed
                      h["template_mean_attempts_to_first_pass"].get(t), summ_max.mean_attempts.get(t),
                      src_zip + f"all_template_summary/based_on_the_max_attempts/{raw_name}_summarized_h_c_t.csv")
        for s in SCEN_IDS:
            C.add("scenario_pass_rate_all_templates", agent, s, h["scenario_pass_rate_all_templates"][s],
                  summ_avg_c.get(s),
                  src_zip + f"all_template_summary/based_on_avg_passing_rate/{raw_name}_summarized_h_c.csv")
            C.add("broad_gen_pass_rate", agent, s, h["scenario_pass_rate_test_templates"][s], test_c.get(s),
                  src_zip + f"test_template_summary/based_on_average_pass_rate/{raw_name}_summarized_h_c.csv")
            C.add("broad_gen_pass_rate", agent, s, h["scenario_pass_rate_test_templates"][s], csv_P[agent][s], fig4a)
        C.add("local_gen_mean_75_templates", agent, None, h["local_gen_mean_over_75_templates"],
              FIG4B_LOCAL[agent], f"Fig. 4b 'Local gen.' ({FIG4B_PNG})")
        C.add("broad_gen_mean_15_scenarios", agent, None, h["broad_gen_mean_over_15_scenarios"],
              FIG4B_BROAD[agent], f"Fig. 4b 'Broad gen.' ({FIG4B_PNG})")
        if agent in TEXT["local_heur"][1]:
            C.add("local_gen_mean_75_templates", agent, None, h["local_gen_mean_over_75_templates"],
                  TEXT["local_heur"][1][agent], f"main.tex line {loc['local_heur']} (percent, 2 d.p. as fraction)")
        if agent in TEXT["broad_excl12"][1]:
            C.add("broad_gen_mean_scenarios_3_15", agent, None, h["broad_gen_mean_over_scenarios_3_15"],
                  TEXT["broad_excl12"][1][agent], f"main.tex line {loc['broad_excl12']}")
        C.add("aibirds_competition_pass_rate", agent, None, None, TEXT["aibirds"][1][agent],
              f"main.tex line {loc['aibirds']} (Table 1; no competition playdata released)")
        # Alternative: pooled over tasks (differs from the template-mean only where templates have < 20 tasks).
        pool = h["pooled_tasks"]
        alt = "alternative: pooled over tasks (each task weighs 1); the paper states averaging over templates"
        for s in SCEN_IDS:
            C.add("scenario_pass_rate_all_templates", agent, s, pool["scenario_pass_rate_all_templates"][s],
                  summ_avg_c.get(s),
                  src_zip + f"all_template_summary/based_on_avg_passing_rate/{raw_name}_summarized_h_c.csv",
                  definition=alt)
            C.add("broad_gen_pass_rate", agent, s, pool["scenario_pass_rate_test_templates"][s], csv_P[agent][s],
                  fig4a, definition=alt)
        C.add("local_gen_mean_75_templates", agent, None, pool["local_gen_mean_over_tasks_of_75_templates"],
              FIG4B_LOCAL[agent], f"Fig. 4b 'Local gen.' ({FIG4B_PNG})", definition=alt)
        C.add("broad_gen_mean_15_scenarios", agent, None, pool["broad_gen_mean_over_15_scenarios"],
              FIG4B_BROAD[agent], f"Fig. 4b 'Broad gen.' ({FIG4B_PNG})", definition=alt)

    # Learning agents and random: only the released per-scenario CSV exists.
    for agent in LEARNING + ["Random"]:
        for s in SCEN_IDS:
            C.add("broad_gen_pass_rate", agent, s, None, csv_P[agent][s], fig4a,
                  note="no raw learning-agent / random-agent episodes are released")
        C.add("broad_gen_mean_15_scenarios", agent, None, np.mean([csv_P[agent][s] for s in SCEN_IDS]),
              FIG4B_BROAD[agent], f"Fig. 4b 'Broad gen.' ({FIG4B_PNG})",
              note="recomputed = mean of the released per-scenario CSV values")
        if agent in TEXT["broad_learn"][1]:
            C.add("broad_gen_mean_15_scenarios", agent, None, np.mean([csv_P[agent][s] for s in SCEN_IDS]),
                  TEXT["broad_learn"][1][agent], f"main.tex line {loc['broad_learn']} (percent)")
        C.add("local_gen_mean_75_templates", agent, None, None, FIG4B_LOCAL[agent],
              f"Fig. 4b 'Local gen.' ({FIG4B_PNG})", note="no per-template learning/random results are released")

    # Phy-Q.
    P_csv = {a: csv_P[a] for a in AGENTS if a != "Human"}
    P_raw = {a: (heur[a]["scenario_pass_rate_test_templates"] if a in HEURISTIC_RAW else csv_P[a])
             for a in AGENTS if a != "Human"}
    variants = {
        "playdata_csv_inputs": phyq(P_csv, csv_P["Human"], csv_sd),
        "recomputed_sd_ddof0": phyq(P_raw, hum["pass_rate"], hum["sd_players_ddof0"]),
        "recomputed_sd_ddof1": phyq(P_raw, hum["pass_rate"], hum["sd_players_ddof1"]),
    }
    variant_def = {
        "playdata_csv_inputs": "alternative: P from broad_generalization_all_agents.csv, sigma from "
                               "broad_generalisation_human.csv",
        "recomputed_sd_ddof0": "primary: P_human and sigma_human (population SD over players) from "
                               "human_player_data.zip, heuristic P from heuristic_agents.zip raw episodes, "
                               "learning/random P from the released CSV",
        "recomputed_sd_ddof1": "alternative: as primary but sigma_human with ddof=1",
    }
    for vname, v in variants.items():
        d = variant_def[vname]
        C.add("phyq_scaling_factor", None, None, v["scaling_factor_100_over_abs_Z_random"], TEXT["scaling"][1],
              f"main.tex line {loc['scaling']}", definition=d)
        for agent in AGENTS[1:-1]:
            C.add("phyq_score", agent, None, v["phyq"][agent], FIG5_PHYQ[agent],
                  f"Fig. 5 ({FIG5_PNG}; icon->agent read visually)", definition=d)
            if agent in TEXT["table1_phyq"][1]:
                C.add("phyq_score", agent, None, v["phyq"][agent], TEXT["table1_phyq"][1][agent],
                      f"main.tex line {loc['table1_phyq']} (Table 1)", definition=d)

    rows = C.rows
    primary = [r for r in rows if not r["definition"].startswith("alternative")]
    alt = [r for r in rows if r["definition"].startswith("alternative")]
    mism = [r for r in primary if r["status"] == "mismatch"]

    split_check = {
        "paper_table_location": f"Supplementary E, Table 'The train-test split of the task templates for the "
                                f"within scenario evaluation' (main.tex line {split_line})",
        "paper_table_as_xml_ids": split_paper,
        "readme_label_to_xml_id_source": "tasks/" + MAPPING_XLSX.name,
        "xlsx_anomalies": xlsx_anomalies,
        "templates_without_readme_label": unmapped,
        "code_benchmark_split_differences": {
            s: {"paper": split_paper[s], "code": split_code[s]} for s in SCEN_IDS
            if sorted(split_paper[s]["train"]) != sorted(split_code[s]["train"])
            or sorted(split_paper[s]["test"]) != sorted(split_code[s]["test"])},
        "test_template_summary_matches_paper_test_templates": all(
            sorted(pd.read_csv(io.BytesIO(nested.read(
                f"test_template_summary/based_on_average_pass_rate/{raw}_summarized_h_c_t.csv"))).category)
            == sorted(t for s in SCEN_IDS for t in split_paper[s]["test"]) for raw in HEURISTIC_RAW.values()),
        "n_templates_fifth_generation": len(templates75),
        "tasks_per_template": sorted({len(v) for v in birds.values()}),
    }

    findings = [
        "Phy-Q (primary definition): " + ", ".join(
            f"{a} {variants['recomputed_sd_ddof0']['phyq'][a]:.2f}" for a in AGENTS[1:-1])
        + f"; scaling factor {variants['recomputed_sd_ddof0']['scaling_factor_100_over_abs_Z_random']:.4f}.",
        "sigma_human: the released sd_human_pass_rate column equals the population SD (ddof=0) over the 19 players "
        "of each player's mean average_rate in the scenario; the paper only says 'standard deviation of human pass "
        "rate'.",
        f"Human data: {hum['n_players']} players released vs 20 stated in the paper; {hum['attempts_sentinel_100_rows']} "
        "task rows carry attempts = 100 (sentinel; average_rate 0).",
        "Heuristic per-template pass rate = mean over a task's plays of (LevelStatus == 'Pass'), averaged over the "
        "template's tasks; the scenario broad-gen value averages the paper's test templates (Supp. E).",
        f"Bambirds was played on {heur['Bambirds']['fraction_of_tasks_in_last20_(81-100)']:.3f} of its tasks within "
        "81-100 (the paper says heuristics used the learning agents' test 20%, i.e. tasks 81-100); 223 Bambirds "
        "tasks have 10 plays instead of 5.",
        "Pig Shooter = raw file Naive.csv (its test-template scenario means equal the 'Pig Shooter' CSV rows "
        "except 2_03, see below).",
        "Released heuristic scenario aggregates (the *_summarized_h_c.csv files and the Pig Shooter 2_03 cell of "
        "broad_generalization_all_agents.csv) pool tasks across templates; the paper says the scenario value "
        "averages the tested templates. The two differ only where a template has < 20 tasks (Naive, EagleWings, "
        "DataLab): Pig Shooter 2_03 broad gen "
        f"{heur['Pig Shooter']['scenario_pass_rate_test_templates']['2_03']:.4f} (template mean) vs "
        f"{csv_P['Pig Shooter']['2_03']:.4f} released (= pooled "
        f"{heur['Pig Shooter']['pooled_tasks']['scenario_pass_rate_test_templates']['2_03']:.4f}).",
        f"Pig Shooter local gen {heur['Pig Shooter']['local_gen_mean_over_75_templates']:.5f} (template mean) / "
        f"{heur['Pig Shooter']['pooled_tasks']['local_gen_mean_over_tasks_of_75_templates']:.5f} (pooled) vs "
        f"Fig. 4b {FIG4B_LOCAL['Pig Shooter']}: neither definition reproduces it, while the template mean "
        "reproduces Bambirds, Eagle's Wing and Datalab to 4 d.p.",
        "Human 1.86 attempts / 23.73 s per task (paper text): no candidate definition over the 19 released players "
        "reproduces them (human.attempts_candidates, human.think_time_candidates); [INFERENCE] they include the "
        "unreleased 20th player.",
        "Phy-Q with sigma_human ddof=1 gives scaling factor "
        f"{variants['recomputed_sd_ddof1']['scaling_factor_100_over_abs_Z_random']:.3f} (paper 13.58), so the "
        "published Phy-Q used the population SD.",
    ]

    result = {
        "schema": "issue_128_a3_phyq_harness_check_v1",
        "paper": {
            "title": "Phy-Q as a measure for physical reasoning intelligence",
            "authors": "Xue, Pinto, Gamage, Nikonova, Zhang, Renz",
            "venue": "Nature Machine Intelligence 2023; arXiv 2108.13696v3",
            "source": "data/external/phyq/phyq_2108.13696_src.tar.gz (see data/external/phyq/MANIFEST.json)",
            "figures_with_numbers": {
                FIG4B_PNG: "Fig. 4 (a: per-scenario broad-gen bars; b: table of Local gen./Broad gen. means)",
                FIG5_PNG: "Fig. 5 (Phy-Q distribution, integers)",
                FIG_LOCAL_PER_TEMPLATE_PNG: "Supplementary D (per-template local-gen bars)",
                FIG_HUMAN_PNG: "Supplementary F (human bars with 95% CI)",
            },
            "readme_result_tables": "none (README.md has scenario descriptions, setup and the play-data column "
                                    "description; no result tables)",
            "quotes": quotes,
        },
        "inputs_sha256": hashes,
        "definitions": {
            "human_average_rate": "(6 - attempts)/5 if attempts <= 5 else 0 (README 'Play Data' item 5; paper "
                                  "Results 'Human Performance'); violations in data: "
                                  f"{hum['average_rate_formula_violations']}",
            "human_scenario_pass_rate": "mean of average_rate over all (player, task) rows of the scenario",
            "heuristic_task_pass_rate": "mean over the task's attempts of LevelStatus == 'Pass' ('Playing' and "
                                        "'Fail' count as not passed)",
            "local_gen": "mean over the 75 templates of the per-template pass rate",
            "broad_gen_scenario": "mean over the scenario's test templates (paper Supp. E) of the per-template pass "
                                  "rate",
            "broad_gen_mean": "mean over the 15 scenarios",
            "phyq": "Z_agent = (1/13) sum_{m=3..15} (P_agent,m - P_human,m) / sigma_human,m; Phy-Q = 100 + Z_agent * "
                    "100/|Z_random| (main.tex lines "
                    f"{quotes['phyq_definition'][2]['main_tex_line']}-{quotes['phyq_definition'][3]['main_tex_line']}"
                    ", section 'Proposed Evaluation Settings')",
        },
        "summary": {"primary": C.summary(primary), "alternative_definitions": C.summary(alt),
                    "n_rows": len(rows)},
        "mismatches_primary": [{k: r[k] for k in ("quantity", "agent", "scenario", "recomputed", "published",
                                                  "abs_diff", "published_source")} for r in mism],
        "findings": findings,
        "phyq": variants,
        "human": hum,
        "heuristic": heur,
        "split_check": split_check,
        "comparisons": rows,
        "native_task_fidelity": fidelity(birds, code_constants(), tex_lines),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "harness-check.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    (OUT_DIR / "harness-check.md").write_text(render_md(result))
    print(json.dumps(result["summary"]))


def render_md(r):
    s = r["summary"]
    nf = r["native_task_fidelity"]
    ex = nf["execution_counts"]["sets"]
    lines = [
        "# Issue #128 A3: Phy-Q harness check",
        "",
        "Generated by `scripts/b127_phyq_harness_check.py` from phy-q/benchmark@514491c `playdata/` and arXiv "
        "2108.13696v3 (source tarball under `data/external/phyq/`). Full detail in `harness-check.json`.",
        "",
        f"Primary definitions: {s['primary']['matches']} match, {s['primary']['mismatches']} mismatch, "
        f"{s['primary']['not_published']} not published, {s['primary']['not_recomputable']} not recomputable. "
        f"Alternative definitions: {s['alternative_definitions']}.",
        "",
        "## Mismatches (primary)",
        "",
        "| quantity | agent | scenario | recomputed | published | source |",
        "|---|---|---|---|---|---|",
    ]
    for m in r["mismatches_primary"]:
        lines.append(f"| {m['quantity']} | {m['agent']} | {m['scenario'] or ''} | {m['recomputed']:.4f} | "
                     f"{m['published']} | {m['published_source']} |")
    lines += ["", "## Findings", ""] + [f"- {f}" for f in r["findings"]]
    lines += ["", "## Native-task fidelity (within-template Relational DQN)", ""]
    w = nf["what_a_run_is"]
    for k in ("templates", "train_test_split", "num_update_steps", "step_0_collection", "steps_1_to_15_collection",
              "evaluation"):
        lines.append(f"- {k}: {w[k]}")
    lines.append(f"- num_worker {w['num_worker']}, num_level_per_agent {w['num_level_per_agent']}, "
                 f"training_attempts_per_level {w['training_attempts_per_level']}, simulation_speed "
                 f"{w['simulation_speed']}")
    t = nf["published_targets"]["all_templates_mean"]
    lines.append(f"- Published target: Relational-Symbolic {t['Relational-Symbolic']}, Relational-Image "
                 f"{t['Relational-Image']} ({t['location']}); SD {t['sd']}. Per-template values: bar chart only.")
    lines += ["", "| variant | set | episodes | executions (lower-upper) | worker-h @20 s | worker-h @60 s |",
              "|---|---|---|---|---|---|"]
    for v, sets in ex.items():
        for name, d in sets.items():
            lines.append(f"| {v} | {name} | {d['episodes']} | {d['executions_lower']:.0f}-{d['executions_upper']:.0f}"
                         f" | {d['worker_hours_at_20s'][0]:.1f}-{d['worker_hours_at_20s'][1]:.1f} | "
                         f"{d['worker_hours_at_60s'][0]:.1f}-{d['worker_hours_at_60s'][1]:.1f} |")
    lines += ["", "worker_hours = executions x seconds_per_execution / 3600 (seconds_per_execution measured by the "
                  "E1 smoke). Variant 'paper_text' is [INFERENCE].", ""]
    lines += ["Release-code defects (read, not executed):", ""] + \
             [f"- {d}" for d in nf["release_code_defects_observed_by_reading"]] + [""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
