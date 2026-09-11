"""Paired native-time fitting over the common observed-history representation."""
import torch
from torch import nn
from torch.nn import functional as F

from world_model.model import Abstraction
from world_model.training.cnn_hybrid import linear_macs
from world_model.training.cohort_v2_visual_parser import ConvSlotVisualPredicateParser, CohortV2VisualParserConfig
from world_model.training.native_history_model import STATE_DIM, PAIRS, CONTINUOUS_PAIRS
from world_model.training.native_history_data import VOCABULARY, VISUAL_DIM, KIND_INDICES, visual_carriers, history_events
from world_model.training.observed_history import ObservedHistoryEncoder

ENDPOINT = 11250
REFERENCE_TRANSITION_MACS = 2063232  # Full-width 22-slot hybrid micro transition; common to both arms.


class NativeVisualParser(ConvSlotVisualPredicateParser):
    """Shared visual-only perception: no symbolic readout or supervision."""
    def __init__(self):
        super().__init__(CohortV2VisualParserConfig(image_height=64, image_width=96), VOCABULARY)
        del self.relation_head
        del self.macro_head

    def forward(self, images):
        slots = self.slot_features(self.backbone(self.encode_images(images)))
        return {"presence_logits": self.presence_head(slots).squeeze(-1),
                "centers": self.center_head(slots).sigmoid(), "kind_logits": self.kind_head(slots)}


def perception_loss(model, batch):
    output = model(batch["images"])
    present = batch["presence"].bool()
    loss = 4 * F.binary_cross_entropy_with_logits(output["presence_logits"], batch["presence"])
    probability = output["presence_logits"].sigmoid()
    for kind in ("pig", "block"):
        indices = [i for i, name in enumerate(VOCABULARY) if name.startswith(kind + ":")]
        loss = loss + F.mse_loss(probability[:, indices].sum(1), batch["presence"][:, indices].sum(1))
    if bool(present.any()):
        loss = loss + F.smooth_l1_loss(output["centers"][present], batch["centers"][present])
        expected = torch.tensor(KIND_INDICES, device=present.device)
        loss = loss + F.cross_entropy(output["kind_logits"][present], expected[None].expand(len(present), -1)[present])
    return loss


@torch.no_grad()
def encode_visual(parser, tensors, batch_size=32):
    device = next(parser.parameters()).device
    outputs = []
    for images in tensors["images"].split(batch_size):
        outputs.append({k: v.cpu() for k, v in parser(images.to(device)).items()})
    output = {k: torch.cat([o[k] for o in outputs]) for k in outputs[0]}
    return visual_carriers(output, tensors["timestamps"])


class CommonHistoryFit(nn.Module):
    def __init__(self):
        super().__init__()
        self.history = ObservedHistoryEncoder(carrier_dim=VISUAL_DIM)
        self.auxiliary = nn.Linear(STATE_DIM + 1, VISUAL_DIM)

    def loss(self, visual, timestamps, segments):
        carrier = encode_history(self.history, visual, timestamps, segments)
        elapsed = torch.log1p(timestamps[1:] - timestamps[:-1]).to(carrier)
        predicted = self.auxiliary(torch.cat((carrier[:-1], elapsed[:, None]), -1))
        return F.mse_loss(predicted, visual[1:])


def encode_history(history, visual, timestamps, segments):
    events, indices = history_events(visual, timestamps, segments)
    memories, _ = history(**events)
    return torch.cat((visual, memories[0, indices]), -1)


def transition_batch(shard, carrier, horizon, generator, batch_size=32, *, symbolic):
    """Uniform observed starts; absent exact endpoints are masked, never replaced."""
    starts = torch.randint(len(carrier), (batch_size,), generator=generator)
    values = carrier.new_zeros(batch_size, 5, STATE_DIM)
    masks = torch.zeros(batch_size, 5, dtype=torch.bool)
    actions = torch.zeros(batch_size, 5)
    batch = {"z": values, "available": masks, "action": actions}
    label_names = ("relations", "relation_mask", "macros", "macro_mask") if symbolic else ()
    for name in label_names:
        source = shard["tensors"][name]
        batch[name] = source.new_zeros((batch_size, 5) + source.shape[1:])
    steps = shard["tensors"]["fixed_steps"].tolist()
    ranges = [(s, {steps[i]: i for i in range(s["start"], s["stop"])}) for s in shard["segment_ranges"]]
    for row, start in enumerate(starts.tolist()):
        segment, lookup = next((s, lookup) for s, lookup in ranges if s["start"] <= start < s["stop"])
        actions[row] = segment["action"]
        for offset in range(5):
            end = lookup.get(steps[start] + offset * horizon)
            if end is not None:
                values[row, offset] = carrier[end]
                masks[row, offset] = True
                for name in label_names:
                    batch[name][row, offset] = shard["tensors"][name][end]
    return batch


def continuous_loss(model, z, available, action, pair):
    """Pure fitting surface accepts no symbolic target dictionary."""
    losses = []
    current = z[:, 0]
    for offset in range(4):
        mask = available[:, offset] & available[:, offset + 1]
        if not bool(mask.any()):
            break
        local = model.carrier(z[:, offset], action, pair)
        current = local if offset == 0 else model.carrier(current, action, pair)
        target = z[:, offset + 1]
        losses.append(F.mse_loss(local[mask], target[mask]) + F.mse_loss(current[mask], target[mask])
                      + .01 * F.relu(current[mask].abs() - 2).square().mean())
    return torch.stack(losses).mean() if losses else None


def symbolic_loss(model, z, labels, masks, mode):
    logits, availability = model.symbols(z, mode)
    available = masks.any(1).any(1) if mode == Abstraction.MICRO else masks
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=z.new_tensor(5.))
    return ((raw * masks).sum() / masks.sum().clamp_min(1)
            + F.binary_cross_entropy_with_logits(availability, available.to(z)))


def hybrid_loss(model, batch, pair):
    loss = continuous_loss(model, batch["z"], batch["available"], batch["action"], pair)
    if loss is None or pair.abstraction == Abstraction.CONTINUOUS:
        return loss
    name = "relations" if pair.abstraction == Abstraction.MICRO else "macros"
    mask_name = "relation_mask" if pair.abstraction == Abstraction.MICRO else "macro_mask"
    terms = []
    for offset in range(5):
        valid = batch["available"][:, offset]
        if bool(valid.any()):
            terms.append(symbolic_loss(model, batch["z"][valid, offset], batch[name][valid, offset],
                                       batch[mask_name][valid, offset], pair.abstraction))
    return loss + torch.stack(terms).mean()


class NativeHistoryController(nn.Module):
    def __init__(self, pure, width=None):
        super().__init__()
        self.pairs = CONTINUOUS_PAIRS if pure else PAIRS
        if width is None:
            # Count only; no trained values or outcome-dependent width choice.
            count = lambda w, outputs: (STATE_DIM + 7) * w + (w + 1) * outputs
            target = count(128, 9)
            width = min(range(128, 161), key=lambda w: (abs(count(w, 3) - target), w)) if pure else 128
        self.layers = nn.Sequential(nn.Linear(STATE_DIM + 6, width), nn.SiLU(), nn.Linear(width, len(self.pairs)))

    def forward(self, carrier, action, remaining):
        if carrier.ndim != 2 or carrier.shape[1] != STATE_DIM or action.shape != (len(carrier), 5):
            raise ValueError("native controller needs current carrier, shot context and remaining time")
        value = self.layers(torch.cat((carrier, action, remaining[:, None].to(carrier) / ENDPOINT), -1))
        horizons = torch.tensor([p.delta for p in self.pairs], device=carrier.device)
        return value.masked_fill(horizons[None] > remaining[:, None], -torch.inf)


@torch.no_grad()
def controller_targets(model, carrier, fixed_steps, action):
    """Training-only native-time DP; return no labels beyond available endpoints."""
    keep = [i for i, step in enumerate(fixed_steps) if (step - fixed_steps[0]) % 50 == 0
            and step - fixed_steps[0] <= ENDPOINT]
    if len(keep) < 2:
        return None
    z = carrier[keep]
    steps = [fixed_steps[i] for i in keep]
    lookup = {s: i for i, s in enumerate(steps)}
    costs = z.new_full((len(z), len(model.pairs)), torch.inf)
    for column, pair in enumerate(model.pairs):
        rows = [i for i, s in enumerate(steps) if s + pair.delta in lookup]
        if not rows:
            continue
        ends = [lookup[steps[i] + pair.delta] for i in rows]
        predicted = model.carrier(z[rows], action[None].expand(len(rows), -1), pair)
        costs[rows, column] = pair.delta * .0004 * (predicted - z[ends]).square().mean(-1) + .01 * linear_macs(model, pair) / REFERENCE_TRANSITION_MACS
    values = z.new_zeros(len(z))
    labels = torch.zeros(len(z) - 1, dtype=torch.long, device=z.device)
    for index in range(len(z) - 2, -1, -1):
        choices = costs[index].clone()
        for column, pair in enumerate(model.pairs):
            end = lookup.get(steps[index] + pair.delta)
            if end is not None:
                choices[column] += values[end]
        values[index], labels[index] = choices.min(0)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("controller teacher lacks a finite exact-endpoint path")
    return {"z": z[:-1], "action": action[None].expand(len(z) - 1, -1), "labels": labels,
            "remaining": torch.tensor([steps[-1] - s for s in steps[:-1]], device=z.device)}


@torch.no_grad()
def rollout(model, controller, initial, action, *, endpoint=ENDPOINT, fixed_pair=None):
    """Deployment surface: no observations from the future and no oracle labels."""
    if endpoint <= 0 or endpoint % 50 or (fixed_pair is not None and (fixed_pair not in model.pairs or endpoint % fixed_pair.delta)):
        raise ValueError("native rollout endpoint must have an exact permitted path")
    current = initial.reshape(1, STATE_DIM)
    action = action.reshape(1, 5)
    elapsed, records = 0, []
    while elapsed < endpoint:
        pair = fixed_pair
        if pair is None:
            remaining = torch.tensor([endpoint - elapsed], device=current.device)
            pair = controller.pairs[int(controller(current, action, remaining).argmax(-1))]
        current = model.carrier(current, action, pair)
        if not bool(torch.isfinite(current).all()):
            raise ValueError("nonfinite native recursive prediction")
        controller_macs = 0 if fixed_pair else sum(m.in_features * m.out_features for m in controller.modules() if isinstance(m, nn.Linear))
        records.append({"start_native_step": elapsed, "horizon_native_steps": pair.delta,
                        "mode": str(pair.abstraction), "transition_calls": 1,
                        "controller_calls": int(fixed_pair is None),
                        "symbol_decoder_calls": int(pair.abstraction != Abstraction.CONTINUOUS),
                        "linear_macs": linear_macs(model, pair) + controller_macs})
        elapsed += pair.delta
    return current[0], records
