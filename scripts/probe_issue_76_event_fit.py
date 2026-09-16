"""Training-only differential probe of event-head class-weight normalization."""

import argparse
from collections import Counter

import torch
from torch.nn import functional as F

from scripts import run_issue_76_balanced_event as run
from scripts import run_issue_76_event_model as old
from scripts.issue_76_scaled_fit import backward_clipped
from world_model.training.event_ranking import CLEAR_INDEX, STOP_KINDS


class ScaledReadout(torch.nn.Module):
    def __init__(self, plan):
        super().__init__()
        self.head = run._head(plan, "cuda")

    def forward(self, initial, endpoint, action):
        action = torch.cat((action[:, :2] * 6, action[:, 2:]), dim=1)
        return self.head(initial, endpoint, action)


class FactoredReadout(torch.nn.Module):
    def __init__(self, state_dim, scale):
        super().__init__()
        self.scale = scale
        self.state = torch.nn.Linear(2 * state_dim, 256)
        self.action = torch.nn.Linear(5, 256)
        self.output = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.SiLU(),
                                          torch.nn.Linear(256, len(STOP_KINDS)))

    def forward(self, initial, endpoint, action):
        action = torch.cat((action[:, :2] * self.scale, action[:, 2:]), dim=1)
        state = F.silu(self.state(torch.cat((initial, endpoint), dim=1)))
        return self.output(state * (1 + self.action(action).tanh()))


def probe(updates, interaction):
    plan = run.load_plan()
    features = run._features(plan, plan["seeds"][0])
    groups = old._role_groups(features["rows"], "training")
    training = torch.tensor([index for group in groups for index in group])
    position = {int(index): offset for offset, index in enumerate(training)}
    batches = [torch.tensor([position[index] for index in group], device="cuda") for group in groups]
    initial = features["initial"][training].cuda()
    endpoint = features["endpoints"]["hybrid"][training, 0].cuda()
    action = features["actions"][training].cuda()
    labels = features["labels"][training].cuda()
    weights = torch.tensor([plan["training_class_weights"][name] for name in STOP_KINDS], device="cuda")
    variants = (("scaled_action", "factored_action", "factored_scaled_action") if interaction
                else ("lineage_normalized", "global_class_balanced"))
    for variant in variants:
        torch.manual_seed(plan["seeds"][0])
        if variant == "scaled_action":
            head = ScaledReadout(plan)
        elif variant.startswith("factored"):
            head = FactoredReadout(initial.shape[1], 6 if variant.endswith("scaled_action") else 1).cuda()
        else:
            head = run._head(plan, "cuda")
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        for _ in range(updates):
            optimizer.zero_grad(set_to_none=True)
            logits = head(initial, endpoint, action)
            ranking = logits.new_zeros(())
            for group, batch in zip(groups, batches, strict=True):
                clear = labels[batch] == CLEAR_INDEX
                if features["rows"][group[0]]["paired_ranking_admissible"] and bool(clear.any()):
                    scores = logits[batch, CLEAR_INDEX]
                    ranking = ranking + torch.logsumexp(scores, 0) - torch.logsumexp(scores[clear], 0)
            ranking = ranking / len(groups)
            if variant == "lineage_normalized":
                classification = torch.stack([
                    F.cross_entropy(logits[batch], labels[batch], weight=weights) for batch in batches]).mean()
            else:
                classification = F.cross_entropy(logits, labels, weight=weights, reduction="none").mean()
            loss = classification + ranking
            backward_clipped(loss, head.parameters(), backward_scale=1.0)
            optimizer.step()
        with torch.no_grad():
            probability = head(initial, endpoint, action).softmax(-1)[:, CLEAR_INDEX].cpu()
        scores = torch.zeros(len(features["rows"]))
        scores[training] = probability
        metric = old._metrics(features["rows"], features["labels"], scores, "training")
        print({"variant": variant, "updates": updates, "loss": float(loss.detach()),
               "training_hits": metric["clear_hits"], "informative_groups": metric["informative_groups"],
               "clear_probability_mean": float(probability.mean()),
               "selected_ordinals": dict(Counter(row["selected_ordinal"] for row in metric["records"]))}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--interaction", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    probe(args.updates, args.interaction)


if __name__ == "__main__":
    main()
