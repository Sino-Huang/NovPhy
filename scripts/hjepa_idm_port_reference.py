"""Issue-125 G-port-IDM reference: the official H-JEPA code with its inverse-dynamics remedy, forward pass and two
optimizer steps on fixed CPU inputs.

Run in the ``hjepa`` env (official code at HJEPA_REF @ 97054ec, torch 2.7.1), CPU only:
    CUDA_VISIBLE_DEVICES= python scripts/hjepa_idm_port_reference.py <configs.json> <out_dir>

The protocol is #123's ``scripts/hjepa_port_reference.py`` (A2: the frozen step at the step-0 LR, then the
scheduler advanced to the end of warm-up and a positive-LR step recorded with gradients, AdamW state, parameters
and LRs), with the IDM remedy added on the official side:

- the model is the official ``hjepa_utils.create_world_model`` of the #123 arm config;
- for every level with ``idm_coeff > 0`` the official ``loss.InverseDynamicsLoss(loss.InverseDynamicsModel(
  state_dim, hidden_dim, action_dim))`` is attached as ``jepa.idm_loss_fn`` (create_world_model's own lines
  326-333; done after construction because those lines read ``predictor.action_dim``, which ``ARPredictor``
  does not accept), and for every level with ``action_sigreg_coeff > 0`` the official ``loss.SIGReg()`` is added
  as ``sigreg_action_level{l}`` (create_world_model lines 353-354);
- the loss is the official ``hjepa_utils.hjepa_forward`` on the full config (its lines 153-165 add the
  action-SIGReg and IDM terms), the optimizer ``main_hjepa._build_hjepa_optimizer_factory`` (per-level groups:
  the heads train in their level's group), spt's ``LinearWarmupCosineAnnealingLR``, global clip 1.0.

configs.json is written by ``run_hjepa_idm --gport-configs``: arm -> {config, idm: {level: [state_dim,
hidden_dim, action_dim]}, input, level1_steps, crop_starts}.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

H_JEPA = Path(os.environ.get("HJEPA_REF", "/mnt/array/sukaih/hjepa-ref/H-JEPA")) / "h_jepa"
SEED_INIT, SEED_INPUT, SEED_STEP, SEED_STEP2 = 20261019, 7, 11, 13
BATCH = 4
TOTAL_STEPS = 9000


def main(configs_path, out_dir):
    sys.path.insert(0, str(H_JEPA))
    os.chdir(H_JEPA)
    from omegaconf import OmegaConf
    import stable_pretraining as spt
    import main_hjepa
    from hjepa_utils import create_world_model, hjepa_forward
    from loss import InverseDynamicsLoss, InverseDynamicsModel, SIGReg
    from models.hjepa import HJEPA

    torch.set_num_threads(8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm, spec in json.loads(Path(configs_path).read_text()).items():
        cfg = OmegaConf.create(spec["config"])
        base = OmegaConf.create(spec["config"])
        for level in range(1, int(cfg.num_levels) + 1):
            for key in ("idm_coeff", "idm", "action_sigreg_coeff"):
                base[f"level{level}"].pop(key, None)
        torch.manual_seed(SEED_INIT)
        model, losses = create_world_model(base)
        for level in range(1, int(cfg.num_levels) + 1):
            if float(cfg[f"level{level}"].get("idm_coeff", 0.0)) > 0:
                state_dim, hidden_dim, action_dim = spec["idm"][str(level)]
                model.get_level(level).idm_loss_fn = InverseDynamicsLoss(
                    InverseDynamicsModel(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim))
        for level in range(1, int(cfg.num_levels) + 1):
            if float(cfg[f"level{level}"].get("action_sigreg_coeff", 0.0)) > 0:
                losses[f"sigreg_action_level{level}"] = SIGReg()
        # AdaLN-zero blocks are identities at init; as in #123's reference, the zero-initialised modulation layers
        # get seeded N(0, 0.02) weights so every block has a nonzero gradient (the same state on both sides)
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
        # launches on the launch transition of clips 0 and 2 (frame 0), no-ops elsewhere; crop starts 0 so the
        # level-1 crop holds the launch transition
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
        step1_lrs = [group["lr"] for group in optimizer.param_groups]
        post_state = {k: v.clone() for k, v in model.state_dict().items()}
        step1_drawn = list(drawn)
        warmup = max(1, int(0.01 * TOTAL_STEPS))
        while scheduler.last_epoch < warmup:
            scheduler.step()
        step2_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.zero_grad(set_to_none=True)
        drawn.clear()
        torch.manual_seed(SEED_STEP2)
        batch2 = {"pixels_level1": x.clone(), "action_level1": action.clone()}
        output2 = hjepa_forward(wrapper, batch2, "fit", cfg, normalize_batch=main_hjepa._normalize_uint8_images_on_device)
        output2["loss"].backward()
        grads2 = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
        norm2 = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        state2 = {name: {k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in optimizer.state[p].items()}
                  for name, p in model.named_parameters() if p in optimizer.state}
        torch.save({"arm": arm, "config": spec["config"], "idm": spec["idm"], "init_state": init,
                    "inputs": {"x": x, "action": action}, "crop_starts": spec["crop_starts"],
                    "official_needed_frames": list(needed), "drawn": step1_drawn, "scalars": scalars,
                    "embeds": embeds, "grads": grads, "grad_norm": float(norm), "post_state": post_state,
                    "lrs_after_step": step1_lrs,
                    "optimizer_defaults": {k: v for k, v in optimizer.defaults.items() if isinstance(v, (int, float, tuple))},
                    "torch": torch.__version__, "seeds": [SEED_INIT, SEED_INPUT, SEED_STEP], "batch": BATCH,
                    "total_steps": TOTAL_STEPS,
                    "step2": {"seed": SEED_STEP2, "warmup_steps": warmup, "lrs_at_step": step2_lrs,
                              "drawn": list(drawn),
                              "scalars": {k: float(v) for k, v in output2.items()
                                          if "loss" in k and torch.is_tensor(v) and v.ndim == 0},
                              "grads": grads2, "grad_norm": float(norm2),
                              "post_state": {k: v.clone() for k, v in model.state_dict().items()},
                              "optimizer_state": state2,
                              "lrs_after_step": [group["lr"] for group in optimizer.param_groups]}},
                   out_dir / f"reference-{arm}.pt")
        idm_terms = {k: round(v, 6) for k, v in scalars.items() if k.startswith(("idm_loss", "sigreg_loss_action"))}
        print(f"[hjepa-idm-port-reference] {arm}: loss {scalars['loss']:.6f}, IDM terms {idm_terms}, step-2 loss "
              f"{float(output2['loss']):.6f} at LRs {step2_lrs}, needed {needed}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
