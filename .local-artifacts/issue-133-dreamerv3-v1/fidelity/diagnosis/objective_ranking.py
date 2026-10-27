"""#133 G-fid diagnosis (CPU, read-only): does the planning objective rank the dataset's ground-truth action sequences
above random ones, under several latent read-outs, and how good is open-loop prediction?"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/array/sukaih/Project/NovPhy")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import b127_dreamerv3_fidelity as fid  # noqa: E402

fid.official_imports()
from datasets.pusht_dset import PROPRIO_MEAN, PROPRIO_STD  # noqa: E402
from datasets.img_transforms import default_transform  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(0)
run = ROOT / ".local-artifacts/issue-133-dreamerv3-v1/fidelity/runs/tier-t-1"
t = pickle.loads((run / "plan_targets.pkl").read_bytes())
wm, _ = fid.load_pusht_model(ROOT / ".local-artifacts/issue-133-dreamerv3-v1/fidelity/model/model.pt", "cpu")
rssm = wm.dynamics
tf = default_transform(224)


def trans(obs):
    v = torch.tensor(obs["visual"]).permute(0, 1, 4, 2, 3).float() / 255.0
    b, n = v.shape[:2]
    v = tf(v.reshape(b * n, *v.shape[2:])).reshape(b, n, 3, 224, 224)
    p = (torch.tensor(obs["proprio"]).float() - PROPRIO_MEAN) / PROPRIO_STD
    return v, p


v0, p0 = trans(t["obs_0"])
vg, pg = trans(t["obs_g"])
pm = fid.PlanningModel(wm)
gt = torch.as_tensor(t["gt_actions"]).float()
print("gt", gt.shape, "targets", v0.shape)


def readouts(state):
    probs = rssm.get_dist(state).base_dist.probs
    b = probs.shape[0]
    stoch = state["stoch"].reshape(b, -1)
    return {"probs": probs.reshape(b, -1), "stoch_mode": stoch, "deter": state["deter"],
            "feat": torch.cat([stoch, state["deter"]], -1), "logit": state["logit"].reshape(b, -1),
            "image": wm.heads["decoder"](rssm.get_feat(state)[:, None])["image"].mode()[:, 0].reshape(b, -1),
            "proprio_dec": wm.heads["decoder"](rssm.get_feat(state)[:, None])["proprio"].mode()[:, 0]}


@torch.no_grad()
def roll(vis, pro, act):
    state = pm._posterior(vis, pro)
    for k in range(act.shape[1]):
        state = rssm.img_step(state, act[:, k], sample=False)
    return readouts(state)


@torch.no_grad()
def goal(vis, pro):
    return readouts(pm._posterior(vis, pro))


keys = ["probs", "stoch_mode", "deter", "feat", "logit", "image", "proprio_dec"]
ranks = {k: [] for k in keys}
gap = []
img_err = {"pred_gt": [], "copy_obs0": [], "goal_recon": []}
n_rand = 300
for i in range(len(gt)):
    g = goal(vg[i:i + 1, 0], pg[i:i + 1, 0])
    acts = torch.cat([gt[i:i + 1], torch.randn(n_rand, 5, 10)])
    r = roll(v0[i:i + 1, 0].expand(n_rand + 1, -1, -1, -1), p0[i:i + 1, 0].expand(n_rand + 1, -1), acts)
    for k in keys:
        loss = ((r[k] - g[k]) ** 2).mean(-1)
        ranks[k].append(int((loss < loss[0]).sum()))  # 0 = gt best
    target = fid.to_dreamer_image(vg[i:i + 1, 0] * 0.5 + 0.5).float().reshape(1, -1) / 255.0
    img_err["pred_gt"].append(float(((r["image"][0] - target) ** 2).mean()))
    start = fid.to_dreamer_image(v0[i:i + 1, 0] * 0.5 + 0.5).float().reshape(1, -1) / 255.0
    img_err["copy_obs0"].append(float(((start - target) ** 2).mean()))
    img_err["goal_recon"].append(float(((g["image"] - target) ** 2).mean()))
out = {k: {"median_rank_of_gt_among_301": float(np.median(v)), "share_gt_top30": float(np.mean(np.array(v) < 30)),
           "share_gt_best": float(np.mean(np.array(v) == 0))} for k, v in ranks.items()}
out["image_mse"] = {k: float(np.mean(v)) for k, v in img_err.items()}
print(json.dumps(out, indent=1))
Path("/tmp/i133diag.json").write_text(json.dumps(out, indent=1))
