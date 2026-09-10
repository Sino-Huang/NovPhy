"""Explicit read-only second-family coverage supplement; original #73 unchanged."""
import argparse
import json
from pathlib import Path

from scripts import run_issue_73_headroom_diagnostic as diagnostic


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--validate',action='store_true')
    args=parser.parse_args(argv)
    args.issue72=diagnostic.matched.OUTPUT;args.issue71=diagnostic.matched.hybrid.ROOT
    args.parser=diagnostic.matched.hybrid.repair.ROOT;args.videos=diagnostic.VIDEOS
    args.output=diagnostic.OUTPUT;args.gallery=diagnostic.GALLERY
    original=diagnostic.load_plan(args)
    root=diagnostic.OUTPUT/'family-supplement'
    definition={
        'identity':'issue-73-second-family-training-audit-v1',
        'reason':'original uniform stride-250 sample aliases alternating generator-family order; original report retained',
        'original_plan_identity':original['identity'],
        'training_lineage_indices':list(range(10,3001,250)),
        'selection':'each original sampled controller-training index + 5; metadata-based, not outcome-based',
        'role':'training','source_code':Path(__file__).read_text(),
        'analysis_source_code':original['source_code'],
        'no_new_training_or_collection':True,'final_evaluation_opened':False}
    path=root/'plan.json'
    if path.exists():
        if diagnostic.read(path)!=definition:raise ValueError('supplement freeze differs')
    elif args.validate:raise ValueError('supplement not run')
    else:diagnostic.write(path,definition)
    source=diagnostic.matched.hybrid.repair.load_plan(args.parser)
    families={}
    for index in definition['training_lineage_indices']:
        record=source['training_records'][index-1]
        raw=diagnostic.read(Path(source['training_release'])/record['path']/'trajectory.json')
        if raw['exposure_role']!='training':raise ValueError('supplement crossed role')
        families[str(index)]=raw['generator_family']
    if set(families.values())!={'type010102'}:raise ValueError('second-family sampling assumption differs')
    args.output=root
    plan={**original,'definitions':{**original['definitions'],'training_lineage_indices':definition['training_lineage_indices']}}
    diagnostic.torch.set_num_threads(4)
    if not args.validate:diagnostic.training_audit(args,plan)
    rows=[diagnostic.read(root/'training'/f'lineage-{i:04d}.json') for i in definition['training_lineage_indices']]
    first=[w for r in rows for w in r['windows'] if w['used_for_controller_labels']]
    later=[w for r in rows for w in r['windows'] if not w['used_for_controller_labels']]
    result={'identity':definition['identity'],'families':families,'lineages':len(rows),
            'controller_windows':len(first),'controller_windows_entirely_prelaunch':sum(w['entirely_prelaunch'] for w in first),
            'later_windows':len(later),'later_windows_entirely_prelaunch':sum(w['entirely_prelaunch'] for w in later),
            'macro_positive_first':diagnostic.np.sum([w['positive_macro_labels'] for w in first],axis=0).tolist(),
            'macro_positive_later':diagnostic.np.sum([w['positive_macro_labels'] for w in later],axis=0).tolist(),
            'final_evaluation_opened':False,'original_report_unchanged':True}
    if args.validate:
        if diagnostic.read(root/'result.json')!=result:raise ValueError('supplement aggregates differ')
        diagnostic.log('second-family supplement exact aggregate validation passed')
    else:
        diagnostic.write(root/'result.json',result)
        print(json.dumps(result,indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
