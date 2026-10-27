"""#133 G-fid diagnosis 2: is the single-frame goal posterior in the space the rollout predicts?"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/array/sukaih/Project/NovPhy")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import b127_dreamerv3_fidelity as fid  # noqa: E402

fid.official_imports()
from datasets.pusht_dset import PushTDataset  # noqa: E402

torch.set_num_threads(4)
torch.manual_seed(0)
wm, _ = fid.load_pusht_model(ROOT / ".local-artifacts/issue-133-dreamerv3-v1/fidelity/model/model.pt", "cpu")
rssm = wm.dynamics
dset = PushTDataset(n_rollout=None, transform=None, data_path=str(ROOT / "data/external/dino_wm/datasets/pusht_noise/val"))


def probs(s):
    return rssm.get_dist(s).base_dist.probs.reshape(s["deter"].shape[0], -1)


def post_at(state, embed):
    """DreamerV3's representation q(z | h, x) of observation embedding x at deterministic state h."""
    x = rssm._obs_out_layers(torch.cat([state["deter"], embed], -1))
    return {**rssm._suff_stats_layer("obs", x), "deter": state["deter"]}


def l2(a, b):
    return float(((a - b) ** 2).mean())


rows = []
with torch.no_grad():
    for i in range(len(dset)):
        obs, act, _, _ = dset[i]
        n = min(11, (obs["visual"].shape[0] - 1) // 5 + 1)
        idx = torch.arange(0, 5 * n, 5)
        img = fid.to_dreamer_image(obs["visual"][idx]).float()
        pro = obs["proprio"][idx]
        a = torch.zeros(n, 10)
        for k in range(1, n):
            a[k] = act[5 * (k - 1):5 * k].reshape(-1)
        first = torch.zeros(1, n)
        first[0, 0] = 1
        o = wm.preprocess({"image": img[None], "proprio": pro[None], "is_first": first,
                           "is_terminal": torch.zeros(1, n)})
        embed = wm.encoder(o)
        state, posts = None, []
        for k in range(n):
            state, _ = rssm.obs_step(state, a[None, k], embed[:, k], o["is_first"][:, k], sample=False)
            posts.append(state)
        for k in range(5, n):
            single, _ = rssm.obs_step(None, None, embed[:, k], torch.ones(1), sample=False)
            s = posts[k - 5]
            for j in range(k - 4, k + 1):
                s = rssm.img_step(s, a[None, j], sample=False)
            rows.append({"filtered_vs_single_frame_posterior": l2(probs(posts[k]), probs(single)),
                         "openloop5_prior_vs_filtered_posterior": l2(probs(s), probs(posts[k])),
                         "openloop5_prior_vs_single_frame_posterior": l2(probs(s), probs(single)),
                         "openloop5_prior_vs_goal_posterior_at_prior_deter": l2(probs(s), probs(post_at(s, embed[:, k]))),
                         "deter_openloop5_vs_filtered": l2(s["deter"], posts[k]["deter"]),
                         "deter_single_vs_filtered": l2(single["deter"], posts[k]["deter"])})
print(json.dumps({k: float(np.mean([r[k] for r in rows])) for k in rows[0]}, indent=1), len(rows))
