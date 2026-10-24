"""#131 D1 THICK G-fid: PyTorch port of the official THICK Dreamer agent, DreamerV2 and Plan2Explore.

Ported from the vendored reference (``vendor/thick``): ``common/nets.py`` ``EnsembleRSSM`` (the DreamerV2 low
level), ``agent.py`` (``Agent.train`` / ``policy``, ``WorldModel.loss`` for ``c_rssm: False``, ``WorldModel.imagine``,
``WorldModel.thick_imagine``, ``ActorCritic`` with THICK Dreamer's coarse critic and long-horizon target),
``common/other.py`` (``lambda_return``, ``long_term_return``, ``StreamNorm``, ``schedule``), ``common/thick_utils.py``
(``THICKNet.sample_step``) and ``expl.py`` (``Plan2Explore``). The THICK world model itself is ``port.WorldModel``
(G-port verified, unchanged). Held to G-port by ``scripts/b127_thick_port_reference.py --agent-reference /
--agent-compare``.

Ported paths: one-hot (discrete) actions, so ``actor_grad`` is ``reinforce`` (upstream ``auto``) and no gradient
flows through the imagined dynamics; the imagination is therefore computed without autograd (the actor, critic and
coarse critic get exactly the gradients of their losses, as upstream's tapes give them). ``pred_discount`` False
(no discount head; VisualPinPad has no early termination).
"""
from __future__ import annotations

import torch
from torch import nn

from world_model.baselines.thick import port


def schedule(string, step=0):
    """``common.other.schedule`` (constant values only are used by the ported configs)."""
    return float(string)


# ---------------------------------------------------------------------------
# DreamerV2 low level (EnsembleRSSM) and world model (c_rssm False)
# ---------------------------------------------------------------------------


class EnsembleRSSM(nn.Module):
    """``common.EnsembleRSSM`` (discrete latents, ensemble 1)."""

    def __init__(self, embed_dim, act_dim, ensemble=5, stoch=30, deter=200, hidden=200, discrete=False, act="elu",
                 norm="none", std_act="softplus", min_std=0.1, rnn_norm="layer_norm"):
        super().__init__()
        if not discrete or ensemble != 1:
            raise NotImplementedError("ported: discrete latents, ensemble 1")
        self.stoch, self.discrete, self.deter, self.hidden, self.act_dim = stoch, discrete, deter, hidden, act_dim
        self.act = port.get_act(act)
        flat = stoch * discrete
        self.cell = port.GRUCell(hidden, deter, norm=(rnn_norm == "layer_norm"))
        self.m = nn.ModuleDict({
            "img_in": port.Dense(flat + act_dim, hidden), "img_in_norm": port.NormLayer(norm, hidden),
            "img_out_0": port.Dense(deter, hidden), "img_out_norm_0": port.NormLayer(norm, hidden),
            "img_dist_0": port.Dense(hidden, flat),
            "obs_out": port.Dense(deter + embed_dim, hidden), "obs_out_norm": port.NormLayer(norm, hidden),
            "obs_dist": port.Dense(hidden, flat)})

    @property
    def feat_dim(self):
        return self.stoch * self.discrete + self.deter

    def initial(self, batch, device=None):
        z = lambda *s: torch.zeros(batch, *s, device=device)
        return {"logit": z(self.stoch, self.discrete), "stoch": z(self.stoch, self.discrete),
                "gates": z(self.deter), "deter": z(self.deter)}

    def _flat(self, x):
        return x.reshape(*x.shape[:-2], self.stoch * self.discrete)

    def _logits(self, x, prefix):
        x = self.act(self.m[f"{prefix}_norm" if prefix == "obs_out" else "img_out_norm_0"](self.m[prefix](x)))
        x = self.m["obs_dist" if prefix == "obs_out" else "img_dist_0"](x)
        return x.reshape(*x.shape[:-1], self.stoch, self.discrete)

    def img_step(self, prev_state, prev_action, sample=True, rnn_sample=True, noise=None):
        x = torch.cat([self._flat(prev_state["stoch"]), prev_action], -1)
        x = self.act(self.m["img_in_norm"](self.m["img_in"](x)))
        deter, gates = self.cell(x, prev_state["deter"])
        logit = self._logits(deter, "img_out_0")
        dist = port.OneHot(logit)
        stoch = dist.sample(noise) if sample else dist.mode()
        return {"stoch": stoch, "deter": deter, "gates": gates, "logit": logit}

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True, rnn_sample=True, noise=None):
        keep = 1.0 - is_first.float()
        mask = lambda x: keep.reshape(-1, *([1] * (x.dim() - 1))) * x
        prev_state = {k: mask(v) for k, v in prev_state.items()}
        prior = self.img_step(prev_state, mask(prev_action), sample, rnn_sample, noise)
        logit = self._logits(torch.cat([prior["deter"], embed], -1), "obs_out")
        dist = port.OneHot(logit)
        stoch = dist.sample(noise) if sample else dist.mode()
        return {"stoch": stoch, "deter": prior["deter"], "gates": prior["gates"], "logit": logit}, prior

    def observe(self, embed, action, is_first, state=None, sample=True, rnn_sample=True, noise=None):
        if state is None:
            state = self.initial(action.shape[0], action.device)
        posts, priors = [], []
        for t in range(action.shape[1]):
            state, prior = self.obs_step(state, action[:, t], embed[:, t], is_first[:, t], sample, rnn_sample, noise)
            posts.append(state)
            priors.append(prior)
        stack = lambda seq: {k: torch.stack([s[k] for s in seq], 1) for k in seq[0]}
        return stack(posts), stack(priors)

    def get_feat(self, state):
        return torch.cat([self._flat(state["stoch"]), state["deter"]], -1)

    kl_loss = port.ContextRSSM.kl_loss


class RSSMWorldModel(nn.Module):
    """``agent.WorldModel`` with ``c_rssm: False`` (DreamerV2 / Plan2Explore world model, no high level)."""

    def __init__(self, config, obs_space, act_dim):
        super().__init__()
        if config.c_rssm or config.hierarchical or config.pred_discount:
            raise NotImplementedError("ported: c_rssm False, hierarchical False, pred_discount False")
        self.config, self.act_dim = config, act_dim
        shapes = {k: tuple(v) for k, v in obs_space.items()}
        self.encoder = port.Encoder(shapes, **config.encoder)
        self.rssm = EnsembleRSSM(self.encoder.out_dim, act_dim, **config.rssm)
        self.heads = nn.ModuleDict({"decoder": port.Decoder(shapes, self.rssm.feat_dim, **config.decoder),
                                    "reward": port.MLP(self.rssm.feat_dim, [], **config.reward_head)})
        self.ctxt_heads = nn.ModuleDict()
        self.grad_heads = tuple(config.grad_heads)
        self.loss_scales = dict(config.loss_scales)
        self.hierarchical = False

    preprocess = port.WorldModel.preprocess

    def model_named_parameters(self):
        modules = [("encoder", self.encoder), ("rssm", self.rssm)] + [(f"heads.{k}", v) for k, v in self.heads.items()]
        return [(f"{prefix}.{n}", p) for prefix, module in modules for n, p in module.named_parameters()]

    def make_optimizers(self):
        return port.TFOptimizer(self.model_named_parameters(), name="model", **self.config.model_opt), None

    def loss(self, data, state=None, noise=None):
        data = self.preprocess(data)
        embed = self.encoder(data)
        post, prior = self.rssm.observe(embed, data["action"], data["is_first"], state, noise=noise)
        kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
        gate_activity = post["gates"].mean()  # GRU update gates (stop-gradient upstream): a constant loss term
        losses = {"kl": kl_loss, "gate_activity": gate_activity, "ctxt_sparsity": gate_activity,
                  "ctxt_kl": torch.zeros((), device=embed.device)}
        feat = self.rssm.get_feat(post)
        likes = {}
        for name, head in self.heads.items():
            out = head(feat if name in self.grad_heads else feat.detach())
            dists = out if isinstance(out, dict) else {name: out}
            for key, dist in dists.items():
                likes[key] = dist.log_prob(data[key]).float()
                losses[key] = -likes[key].mean()
        model_loss = sum(self.loss_scales.get(k, 1.0) * v for k, v in losses.items())
        outs = {"embed": embed, "feat": feat, "post": post, "prior": prior, "likes": likes, "kl": kl_value}
        return model_loss, {k: v[:, -1] for k, v in post.items()}, outs, losses


# ---------------------------------------------------------------------------
# behaviour learning
# ---------------------------------------------------------------------------


def lambda_return(reward, value, pcont, bootstrap, lambda_):
    """``common.other.lambda_return`` along axis 0."""
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns, last = [], bootstrap
    for index in reversed(range(reward.shape[0])):
        last = inputs[index] + pcont[index] * lambda_ * last
        returns.append(last)
    return torch.stack(returns[::-1], 0)


def long_term_return(reward, value, pcont, inter_reward, inter_discount):
    expected = reward + pcont * value
    return inter_reward + inter_discount * expected, expected


class StreamNorm:
    """``common.other.StreamNorm`` (the magnitude is float64, as upstream)."""

    def __init__(self, shape=(), momentum=0.99, scale=1.0, eps=1e-8):
        self.shape, self.momentum, self.scale, self.eps = tuple(shape), momentum, scale, eps
        self.mag = torch.ones(self.shape, dtype=torch.float64)

    def __call__(self, inputs):
        self.update(inputs)
        return self.transform(inputs)

    def update(self, inputs):
        batch = inputs.reshape((-1,) + self.shape)
        mag = batch.abs().mean(0).double()
        self.mag = self.momentum * self.mag.to(mag.device) + (1 - self.momentum) * mag

    def transform(self, inputs):
        values = inputs.reshape((-1,) + self.shape)
        values = values / (self.mag.to(inputs.dtype).to(inputs.device)[None] + self.eps)
        values = values * self.scale
        return values.reshape(inputs.shape)

    def state_dict(self):
        return {"mag": self.mag.cpu()}

    def load_state_dict(self, state):
        self.mag = state["mag"].clone()


def thick_sample_step(thick_net, inps, noise):
    """``THICKNet.sample_step(inps, sample=True)``: prior high-level action, predicted latent, sampled action and
    elapsed time (``thick_readouts.sample_skip`` False)."""
    hl = port.OneHot(thick_net.hl_prior_logits(inps)).sample(noise)
    feat = thick_net.body(torch.cat([inps, hl.to(inps.dtype)], -1))
    logit = thick_net.state_logits(feat)
    stoch = port.OneHot(logit).sample(noise)
    return {"thick_stoch": stoch, "thick_logit": logit,
            "thick_act": thick_net.thick_heads["action"](feat).sample(noise),
            "thick_time": thick_net.thick_heads["time"](feat).sample(noise),
            "reward": thick_net.thick_heads["reward"](feat).mode()}


class ActorCritic(nn.Module):
    """``agent.ActorCritic`` (with THICK Dreamer's coarse critic when ``thick_dreamer``)."""

    def __init__(self, config, act_dim, feat_dim, coarse_dim=None):
        super().__init__()
        self.config = config
        actor = dict(config.actor)
        actor["dist"] = "onehot" if actor["dist"] == "auto" else actor["dist"]
        actor_grad = "reinforce" if config.actor_grad == "auto" else config.actor_grad
        if actor["dist"] != "onehot" or actor_grad != "reinforce":
            raise NotImplementedError("ported: one-hot actions with reinforce gradients")
        self.actor = port.MLP(feat_dim, act_dim, **actor)
        self.critic = port.MLP(feat_dim, [], **config.critic)
        self.thick_dreamer = bool(config.thick_dreamer)
        if self.thick_dreamer:
            self.coarse_critic = port.MLP(coarse_dim, [], **config.critic)
            if config.slow_target:
                self.coarse_target_critic = port.MLP(coarse_dim, [], **config.critic)
        if config.slow_target:
            self.target_critic = port.MLP(feat_dim, [], **config.critic)
        self.updates = 0
        self.rewnorm = StreamNorm(**config.reward_norm)
        critic_named = [(f"critic.{n}", p) for n, p in self.critic.named_parameters()]
        if self.thick_dreamer:
            critic_named += [(f"coarse_critic.{n}", p) for n, p in self.coarse_critic.named_parameters()]
        self.actor_opt = port.TFOptimizer([(f"actor.{n}", p) for n, p in self.actor.named_parameters()],
                                          name="actor", **config.actor_opt)
        self.critic_opt = port.TFOptimizer(critic_named, name="critic", **config.critic_opt)

    def _target(self):
        return self.target_critic if self.config.slow_target else self.critic

    def _coarse_target(self):
        return self.coarse_target_critic if self.config.slow_target else self.coarse_critic

    def train_step(self, wm, start, reward_fn, noise):
        cfg = self.config
        with torch.no_grad():
            seq = imagine(wm, self.actor, start, cfg.imag_horizon, noise)
            seq["reward"] = self.rewnorm(reward_fn(seq))
            target = self.target(seq)
        actor_loss, actor_metrics = self.actor_loss(seq, target)
        if self.thick_dreamer:
            with torch.no_grad():
                hl_seq = thick_imagine(wm, seq, noise)
                hl_seq["reward"] = self.rewnorm(hl_seq["mean_reward"])
                target = self.thick_dreamer_target(seq, hl_seq)
            critic_loss = self.thick_dreamer_loss(seq, target)
        else:
            critic_loss = self.critic_loss(seq, target)
        actor_norm = self.actor_opt.step(actor_loss)
        critic_norm = self.critic_opt.step(critic_loss)
        self.update_slow_target()
        return {"actor_loss": actor_loss.detach(), "critic_loss": critic_loss.detach(), "actor_grad_norm": actor_norm,
                "critic_grad_norm": critic_norm, "critic_target": target.mean(), **actor_metrics}

    def target(self, seq):
        value = self._target()(seq["feat"]).mode()
        return lambda_return(seq["reward"][:-1].float(), value[:-1], seq["discount"][:-1].float(), value[-1],
                             self.config.discount_lambda)

    def actor_loss(self, seq, target):
        policy = self.actor(seq["feat"][:-2].detach())
        baseline = self._target()(seq["feat"][:-2]).mode()
        advantage = (target[1:] - baseline).detach()
        action = seq["action"][1:-1].detach()
        objective = policy.log_prob(action) * advantage
        ent = policy.entropy()
        objective = objective + schedule(self.config.actor_ent) * ent
        weight = seq["weight"].detach()
        return -(weight[:-2] * objective).mean(), {"actor_ent": ent.mean().detach()}

    def critic_loss(self, seq, target):
        dist = self.critic(seq["feat"][:-1])
        return -(dist.log_prob(target.detach()) * seq["weight"].detach()[:-1]).mean()

    def thick_dreamer_loss(self, seq, target):
        target = target.detach()
        weight = seq["weight"].detach()
        loss = -(self.critic(seq["feat"][:-1]).log_prob(target) * weight[:-1]).mean()
        coarse = -(self.coarse_critic(seq["ctxt_out"][:-1]).log_prob(target) * weight[:-1]).mean()
        return loss + coarse

    def thick_dreamer_target(self, seq, hl_seq):
        target = self.target(seq)
        hl_value = self._coarse_target()(hl_seq["ctxt_out"]).mode()
        hl_target, _ = long_term_return(hl_seq["reward"][:-1].float(), hl_value[:-1], hl_seq["discount"][:-1].float(),
                                        hl_seq["inter_reward"][:-1].float(), hl_seq["inter_discount"][:-1].float())
        psi = self.config.critic_psi
        return psi * target + (1 - psi) * hl_target

    @torch.no_grad()
    def update_slow_target(self):
        if self.config.slow_target:
            if self.updates % self.config.slow_target_update == 0:
                mix = 1.0 if self.updates == 0 else float(self.config.slow_target_fraction)
                pairs = list(zip(self.critic.parameters(), self.target_critic.parameters()))
                if self.thick_dreamer:
                    pairs += list(zip(self.coarse_critic.parameters(), self.coarse_target_critic.parameters()))
                for s, d in pairs:
                    d.copy_(mix * s + (1 - mix) * d)
            self.updates += 1

    def optimizer_states(self):
        return {"actor": self.actor_opt.state_dict(), "critic": self.critic_opt.state_dict(), "updates": self.updates,
                "rewnorm": self.rewnorm.state_dict()}

    def load_optimizer_states(self, state):
        self.actor_opt.load_state_dict(state["actor"])
        self.critic_opt.load_state_dict(state["critic"])
        self.updates = int(state["updates"])
        self.rewnorm.load_state_dict(state["rewnorm"])


def imagine(wm, actor, start, horizon, noise):
    """``WorldModel.imagine`` (``actor_seq_inp: feat``; no discount head)."""
    rssm = wm.rssm
    c_rssm = isinstance(rssm, port.ContextRSSM)
    start = {k: v.reshape(-1, *v.shape[2:]) for k, v in start.items()}
    start["feat"] = rssm.get_feat(start)
    start["ctxt_out"] = rssm.get_coarse_pred(start) if c_rssm else start["feat"]
    start["action"] = torch.zeros_like(actor(start["feat"]).mode())
    seq = {k: [v] for k, v in start.items()}
    for _ in range(horizon):
        action = actor(seq["feat"][-1].detach()).sample(noise)
        state = rssm.img_step({k: v[-1] for k, v in seq.items()}, action, noise=noise)
        feat = rssm.get_feat(state)
        ctxt_out = rssm.get_coarse_pred(state) if c_rssm else feat
        for key, value in {**state, "action": action, "feat": feat, "ctxt_out": ctxt_out}.items():
            seq[key].append(value)
    seq = {k: torch.stack(v, 0) for k, v in seq.items()}
    disc = wm.config.discount * torch.ones(seq["feat"].shape[:-1], device=seq["feat"].device)
    seq["mean_reward"] = wm.heads["reward"](seq["feat"]).mode()
    seq["inter_reward"] = torch.zeros_like(seq["mean_reward"])
    seq["discount"] = disc
    seq["is_terminal"] = torch.zeros_like(disc)
    seq["delta_time"] = torch.ones_like(disc)
    seq["weight"] = torch.cumprod(torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0)
    return seq


def thick_imagine(wm, inp_seq, noise):
    """``WorldModel.thick_imagine``: one high-level step from every state of the imagined sequence."""
    rssm = wm.rssm
    max_len = wm.config.replay.maxlen
    out = {k: [] for k in ("stoch", "context", "ctxt_out", "action", "mean_reward", "delta_time", "inter_reward",
                           "ctxt_stoch", "prev_stoch", "prev_action")}
    for t in range(inp_seq["context"].shape[0]):
        inputs = port.gen_thick_inps(inp_seq["context"][t], inp_seq["stoch"][t])
        thick = thick_sample_step(wm.thick_net, inputs, noise)
        prior = {"stoch": thick["thick_stoch"], "context": inp_seq["context"][t], "deter": inp_seq["deter"][t]}
        ll = rssm.img_step(prior, thick["thick_act"], noise=noise)
        full = {"stoch": ll["ctxt_stoch"], "context": ll["context"],
                "delta_time": torch.clamp(thick["thick_time"].squeeze(-1), 0, max_len) + 1,
                "action": thick["thick_act"], "ctxt_stoch": ll["ctxt_stoch"], "prev_stoch": ll["prev_stoch"],
                "prev_action": ll["prev_action"]}
        ctxt_out = rssm.get_coarse_pred(full)
        reward = wm.ctxt_heads["reward"](ctxt_out).mode()
        for key, value in {**full, "ctxt_out": ctxt_out, "mean_reward": reward,
                           "inter_reward": thick["reward"].squeeze(-1)}.items():
            out[key].append(value)
    seq = {k: torch.stack(v, 0) for k, v in out.items()}
    disc = wm.config.discount * torch.ones(seq["ctxt_out"].shape[:-1], device=seq["ctxt_out"].device)
    seq["is_terminal"] = torch.zeros_like(disc)
    seq["inter_discount"] = torch.pow(disc, seq["delta_time"])
    seq["discount"] = disc
    return seq


class Plan2Explore(nn.Module):
    """``expl.Plan2Explore`` (disagreement of ``disag_models`` one-step heads; ``disag_target: stoch``)."""

    def __init__(self, config, act_dim, feat_dim, coarse_dim=None):
        super().__init__()
        self.config = config
        if config.disag_target != "stoch" or config.disag_log or config.expl_extr_scale:
            raise NotImplementedError("ported: disag_target stoch, disag_log False, expl_extr_scale 0")
        self.ac = ActorCritic(config, act_dim, feat_dim, coarse_dim)
        size = config.rssm.stoch * config.rssm.discrete
        in_dim = feat_dim + (act_dim if config.disag_action_cond else 0)
        self.networks = nn.ModuleList([port.MLP(in_dim, size, **config.expl_head)
                                       for _ in range(config.disag_models)])
        self.opt = port.TFOptimizer([(f"networks.{n}", p) for n, p in self.networks.named_parameters()],
                                    name="expl", **config.expl_opt)
        self.intr_rewnorm = StreamNorm(**config.expl_reward_norm)

    @property
    def actor(self):
        return self.ac.actor

    def train_step(self, wm, start, context, data, noise):
        stoch = start["stoch"].reshape(*start["stoch"].shape[:-2], -1)
        inputs = context["feat"]
        if self.config.disag_action_cond:
            inputs = torch.cat([inputs, data["action"].to(inputs.dtype)], -1)
        offset = self.config.disag_offset
        targets, inputs = stoch.detach(), inputs.detach()
        if offset:
            targets, inputs = targets[:, offset:], inputs[:, :-offset]
        loss = -sum(head(inputs).log_prob(targets).mean() for head in self.networks)
        expl_norm = self.opt.step(loss)
        metrics = self.ac.train_step(wm, start, self.intr_reward, noise)
        return {"expl_loss": loss.detach(), "expl_grad_norm": expl_norm, **metrics}

    def intr_reward(self, seq):
        inputs = seq["feat"]
        if self.config.disag_action_cond:
            inputs = torch.cat([inputs, seq["action"].to(inputs.dtype)], -1)
        preds = torch.stack([head(inputs).mode() for head in self.networks], 0)
        disag = preds.std(0, unbiased=False).mean(-1)
        return self.config.expl_intr_scale * self.intr_rewnorm(disag)

    def optimizer_states(self):
        return {"opt": self.opt.state_dict(), "ac": self.ac.optimizer_states(),
                "intr_rewnorm": self.intr_rewnorm.state_dict()}

    def load_optimizer_states(self, state):
        self.opt.load_state_dict(state["opt"])
        self.ac.load_optimizer_states(state["ac"])
        self.intr_rewnorm.load_state_dict(state["intr_rewnorm"])


# ---------------------------------------------------------------------------
# the agent
# ---------------------------------------------------------------------------


class Agent(nn.Module):
    """``agent.Agent``: ``train`` (world model, high level, task behaviour, exploration behaviour) and ``policy``."""

    def __init__(self, config, obs_space, act_dim):
        super().__init__()
        self.config = config
        if config.c_rssm:
            self.wm = port.WorldModel(config, obs_space, act_dim, cont_action=False)
            coarse = self.wm.rssm.coarse_dim
        else:
            self.wm = RSSMWorldModel(config, obs_space, act_dim)
            coarse = None
        feat = self.wm.rssm.feat_dim
        self.task_behavior = ActorCritic(config, act_dim, feat, coarse)
        self.greedy = config.expl_behavior == "greedy"
        if not self.greedy:
            if config.expl_behavior != "Plan2Explore":
                raise NotImplementedError(config.expl_behavior)
            self.expl_behavior = Plan2Explore(config, act_dim, feat, coarse)
        self.model_opt, self.thick_opt = self.wm.make_optimizers()
        if config.train_plan or config.eval_plan or config.hl_plan:
            raise NotImplementedError("ported: no planning")

    def reward_fn(self, seq):
        return self.wm.heads["reward"](seq["feat"]).mode()

    def train_step(self, data, noise):
        model_loss, _, outs, losses = self.wm.loss(data, noise=noise)
        metrics = {f"{k}_loss": v.detach() for k, v in losses.items()}
        metrics["model_loss"] = model_loss.detach()
        metrics["model_grad_norm"] = self.model_opt.step(model_loss)
        if self.wm.hierarchical:
            total, thick_losses, _ = self.wm.thick_loss(data, noise=noise)
            metrics.update({f"{k}_loss": v.detach() for k, v in thick_losses.items()})
            metrics["thick_net_loss"] = total.detach()
            metrics["thick_net_grad_norm"] = self.thick_opt.step(total)
        start = {k: v.detach() for k, v in outs["post"].items()}
        metrics.update(self.task_behavior.train_step(self.wm, start, self.reward_fn, noise))
        if not self.greedy:
            context = {"feat": outs["feat"].detach()}
            expl = self.expl_behavior.train_step(self.wm, start, context, data, noise)
            metrics.update({f"expl_{k}": v for k, v in expl.items()})
        return metrics

    @torch.no_grad()
    def policy(self, obs, state=None, mode="train", noise=None):
        """``Agent.policy`` for a batch of observations (torch tensors); returns (action, (latent, action))."""
        rssm = self.wm.rssm
        n = obs["reward"].shape[0]
        device = obs["reward"].device
        if state is None:
            state = (rssm.initial(n, device), torch.zeros(n, rssm.act_dim, device=device))
        latent, action = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        sample = mode == "train" or not self.config.eval_state_mean
        rnn_sample = mode == "train" or not self.config.eval_ctxt_mean
        latent, _ = rssm.obs_step(latent, action, embed, obs["is_first"], sample, rnn_sample, noise)
        feat = rssm.get_feat(latent)
        if mode == "eval":
            action = self.task_behavior.actor(feat).mode()
        elif mode == "explore":
            actor = self.task_behavior.actor if self.greedy else self.expl_behavior.actor
            action = actor(feat).sample(noise)
        elif mode == "train":
            action = self.task_behavior.actor(feat).sample(noise)
        else:
            raise NotImplementedError(mode)
        return action, (latent, action)

    def optimizers(self):
        out = [self.model_opt] + ([self.thick_opt] if self.thick_opt is not None else [])
        out += [self.task_behavior.actor_opt, self.task_behavior.critic_opt]
        if not self.greedy:
            expl = self.expl_behavior
            out += [expl.opt, expl.ac.actor_opt, expl.ac.critic_opt]
        return out

    def to(self, *args, **kwargs):
        """``nn.Module.to`` that also moves the Adam moments and rebinds each optimizer to the module's parameters
        (``nn.Module.to`` may replace parameter objects, e.g. for the meta device)."""
        qualified = {id(p): n for n, p in self.named_parameters()}
        slots = [[qualified[id(p)] for _, p in opt.named] for opt in self.optimizers()]
        module = super().to(*args, **kwargs)
        params = dict(self.named_parameters())
        device = next(self.parameters()).device
        for opt, names in zip(self.optimizers(), slots):
            opt.named = [(label, params[q]) for (label, _), q in zip(opt.named, names)]
            opt.m = [t.to(device) for t in opt.m]
            opt.v = [t.to(device) for t in opt.v]
        return module

    def optimizer_states(self):
        out = {"model": self.model_opt.state_dict(), "task": self.task_behavior.optimizer_states()}
        if self.thick_opt is not None:
            out["thick"] = self.thick_opt.state_dict()
        if not self.greedy:
            out["expl"] = self.expl_behavior.optimizer_states()
        return out

    def load_optimizer_states(self, state):
        self.model_opt.load_state_dict(state["model"])
        self.task_behavior.load_optimizer_states(state["task"])
        if self.thick_opt is not None:
            self.thick_opt.load_state_dict(state["thick"])
        if not self.greedy:
            self.expl_behavior.load_optimizer_states(state["expl"])
