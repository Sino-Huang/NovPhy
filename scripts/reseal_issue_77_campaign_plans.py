"""Reseal issue-77 campaign plan.json files after prepare-module source fixes.

Guards: regenerates via each prepare module's make_plan(); asserts the frozen plan
differs ONLY in source_text (membership/branches/prior_exposure byte-identical);
writes atomically. Aborts on any wider drift.
"""
import json
import os
from pathlib import Path

from scripts import prepare_issue_77_n1 as n1
from scripts import prepare_issue_77_n2 as n2
from scripts import prepare_issue_77_n2n as n2n

CAMPAIGNS = (n1, n2, n2n)

ALLOWED_DRIFT = {'source_text', 'prior_exposure', 'scenario_binding'}


def reseal(module):
    root = Path(module.ROOT)
    plan_path = root / 'plan.json'
    frozen = json.loads(plan_path.read_text())
    regenerated = module.make_plan()
    differing = sorted(k for k in set(frozen) | set(regenerated)
                       if frozen.get(k) != regenerated.get(k))
    unexpected = [k for k in differing if k not in ALLOWED_DRIFT]
    if unexpected:
        raise ValueError(f'{root.name}: regenerated plan differs in frozen sections: {unexpected}')
    for key in ('members', 'branches'):
        if frozen[key] != regenerated[key]:
            raise ValueError(f'{root.name}: {key} drifted')
    frozen_paths = {e['path'] for e in frozen['prior_exposure']}
    regen_paths = {e['path'] for e in regenerated['prior_exposure']}
    if not frozen_paths <= regen_paths:
        raise ValueError(f'{root.name}: prior_exposure lost entries: {sorted(frozen_paths - regen_paths)}')
    temporary = plan_path.with_name(plan_path.name + '.tmp')
    temporary.write_text(json.dumps(regenerated, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, plan_path)
    print(f'{root.name}: plan.json resealed; drift limited to {differing}')


def main():
    for module in CAMPAIGNS:
        reseal(module)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
