"""Issue-123 G-port reference: the official H-JEPA code's forward pass and one optimizer step on fixed inputs.

Run in the ``hjepa`` env (official code at HJEPA_REF @ 97054ec, torch 2.7.1):
    python scripts/hjepa_port_reference.py <configs.json> <out_dir>

configs.json (written by ``run_hjepa_comparison --gport-configs``) maps an arm name to its create_world_model
config, the input kind and fixed crop starts. For every arm this builds the model with the official
``hjepa_utils.create_world_model``, feeds the full dense clip (every level-1 frame) through the official
``hjepa_forward`` with ``main_hjepa._normalize_uint8_images_on_device`` (the crop starts the official
``HJEPA._draw_crop_start`` would draw are fixed to the given values without consuming the generator), then
takes one step exactly as stable-pretraining 0.1.7's manual optimisation does: backward, global
``clip_grad_norm_`` 1.0 over the optimizer's parameters, the optimizer built by
``main_hjepa._build_hjepa_optimizer_factory`` (spt ``create_optimizer``), ``opt.step()``, and spt's
``LinearWarmupCosineAnnealingLR`` with its default warm-up (1 % of the total steps) stepped once. CPU, float32.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

H_JEPA = Path(os.environ.get("HJEPA_REF", "/mnt/array/sukaih/hjepa-ref/H-JEPA")) / "h_jepa"
SEED_INIT, SEED_INPUT, SEED_STEP = 20261018, 7, 11
BATCH = 4
TOTAL_STEPS = 9000


def main(configs_path, out_dir):
    sys.path.insert(0, str(H_JEPA))
    os.chdir(H_JEPA)
    from omegaconf import OmegaConf
    import stable_pretraining as spt
    import main_hjepa
    from hjepa_utils import create_world_model, hjepa_forward
    from models.hjepa import HJEPA

    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm, spec in json.loads(Path(configs_path).read_text()).items():
        cfg = OmegaConf.create(spec["config"])
        torch.manual_seed(SEED_INIT)
        model, losses = create_world_model(cfg)
        # AdaLN-zero blocks are identities at init (zero gates give their inner weights exactly zero gradients);
        # the check starts from a state with every block active: the zero-initialised modulation layers get
        # N(0, 0.02) weights (seeded), the same state on both sides.
        g_init = torch.Generator().manual_seed(SEED_INIT + 1)
        with torch.no_grad():
            for name, module in model.named_modules():
                if name.endswith("adaLN_modulation"):
                    module[-1].weight.copy_(torch.randn(module[-1].weight.shape, generator=g_init) * 0.02)
                    module[-1].bias.copy_(torch.randn(module[-1].bias.shape, generator=g_init) * 0.02)
        init = {k: v.clone() for k, v in model.state_dict().items()}
        n1, frameskip = spec["level1_steps"], int(cfg.data.dataset.level1.frameskip)
        g = torch.Generator().manual_seed(SEED_INPUT)
        if spec["input"] == "pixels":
            x = torch.randint(0, 256, (BATCH, n1, 3, 240, 320), generator=g, dtype=torch.uint8)
        else:
            x = torch.randn(BATCH, n1, 236, generator=g)
        action = torch.zeros(BATCH, n1, frameskip * 5)
        action[0, 0, :5] = torch.tensor([-0.25, 0.1, 1.0, 0.0, 1.0])
        action[2, 0, :5] = torch.tensor([-0.1, 0.15, 1.0, 0.0, 1.0])
        queue = [s for s in spec["crop_starts"] if s is not None]
        drawn = []

        def fixed_draw(self, seq_len, target_len, device):
            value = queue[len(drawn)]
            drawn.append(value)
            return value

        HJEPA._draw_crop_start = fixed_draw
        needed = model._draw_crops_and_level1_frames(n1, int(cfg.num_levels), "cpu")[1]
        drawn.clear()

        class Self(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = model
                for name, module in losses.items():
                    setattr(self, name, module)

            def log_dict(self, *_, **__):
                pass

        wrapper = Self()
        wrapper.train()
        batch = {"pixels_level1": x.clone(), "action_level1": action.clone()}
        torch.manual_seed(SEED_STEP)
        output = hjepa_forward(wrapper, batch, "fit", cfg, normalize_batch=main_hjepa._normalize_uint8_images_on_device)
        scalars = {k: float(v) for k, v in output.items() if "loss" in k and torch.is_tensor(v) and v.ndim == 0}
        embeds = {k: output[k].detach().clone() for k in output if k.startswith(("embed_", "action_"))
                  and torch.is_tensor(output[k])}
        optimizer = main_hjepa._build_hjepa_optimizer_factory(model, cfg)(None)
        scheduler = spt.optim.create_scheduler(optimizer, {"type": "LinearWarmupCosineAnnealingLR",
                                                           "warmup_steps": max(1, int(0.01 * TOTAL_STEPS)),
                                                           "max_steps": TOTAL_STEPS})
        output["loss"].backward()
        params = [p for group in optimizer.param_groups for p in group["params"]]
        grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        torch.save({"arm": arm, "config": spec["config"], "init_state": init, "inputs": {"x": x, "action": action},
                    "crop_starts": spec["crop_starts"], "official_needed_frames": list(needed),
                    "drawn": list(drawn), "scalars": scalars, "embeds": embeds, "grads": grads,
                    "grad_norm": float(norm), "post_state": {k: v.clone() for k, v in model.state_dict().items()},
                    "lrs_after_step": [group["lr"] for group in optimizer.param_groups],
                    "optimizer_defaults": {k: v for k, v in optimizer.defaults.items() if isinstance(v, (int, float, tuple))},
                    "torch": torch.__version__, "seeds": [SEED_INIT, SEED_INPUT, SEED_STEP], "batch": BATCH,
                    "total_steps": TOTAL_STEPS}, out_dir / f"reference-{arm}.pt")
        print(f"[hjepa-port-reference] {arm}: loss {scalars['loss']:.6f}, needed {needed}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
