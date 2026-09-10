"""Read-only canonical/aligned prefab comparison for #76; no asset substitution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "scripts/issue_76_asset_audit.py"
NAMES = ("BasicBig","PinkBigPig")
BODY_FIELDS = ("m_Mass","m_GravityScale","m_LinearDrag","m_AngularDrag","m_CollisionDetection")


def inspect_assets(path):
    import UnityPy
    if UnityPy.__version__ != "1.25.3": raise ValueError("asset audit requires the inspected UnityPy==1.25.3 reader")
    files = {}
    def load(file):
        file = Path(file)
        if file not in files: files[file] = next(iter(UnityPy.load(str(file)).files.values()))
        return files[file]
    assets = load(path)
    result = {}
    for obj in assets.objects.values():
        if obj.type.name != "GameObject" or obj.peek_name() not in NAMES: continue
        game_object = obj.read_typetree()
        components,body,scripts = [],None,[]
        for entry in game_object["m_Component"]:
            pointer = entry["component"]
            if pointer["m_FileID"] != 0: raise ValueError("prefab component is outside its resource file")
            component = assets.objects[pointer["m_PathID"]]
            components.append(component.type.name)
            if component.type.name == "Rigidbody2D":
                values = component.read_typetree()
                body = {key:values[key] for key in BODY_FIELDS}
            if component.type.name == "MonoBehaviour":
                # Only the base header is available in stripped player typetrees;
                # do not invent interpretations of the remaining serialized fields.
                pointer = component.read_typetree(check_read=False)["m_Script"]
                source = assets if pointer["m_FileID"] == 0 else load(path.parent/assets.externals[pointer["m_FileID"]-1].path)
                script = source.objects[pointer["m_PathID"]].read_typetree()
                scripts.append({"class":script["m_ClassName"],"assembly":script["m_AssemblyName"]})
        result[game_object["m_Name"]] = {"resource_object_path_id":obj.path_id,"components":components,
                                       "rigidbody":body,"scripts":scripts}
    return {"source":str(path),"source_bytes":path.stat().st_size,"prefabs":result}


def compare(canonical,aligned):
    source = canonical["prefabs"]; target = aligned["prefabs"]
    missing = sorted(set(source)-set(target))
    body_differences = {k:{"canonical":source["BasicBig"]["rigidbody"][k],"aligned":target["BasicBig"]["rigidbody"][k]}
                        for k in BODY_FIELDS if source["BasicBig"]["rigidbody"][k] != target["BasicBig"]["rigidbody"][k]}
    return {"missing_aligned_prefabs":missing,"normal_basic_big_body_differences":body_differences,
            "canonical_basic_big_scripts":source["BasicBig"]["scripts"],"aligned_basic_big_scripts":target["BasicBig"]["scripts"],
            "canonical_normal_novel_body_differences":{k:[source["BasicBig"]["rigidbody"][k],source["PinkBigPig"]["rigidbody"][k]]
                for k in BODY_FIELDS if source["BasicBig"]["rigidbody"][k] != source["PinkBigPig"]["rigidbody"][k]},
            "appearance_only_equivalence_established":False,
            "conclusion":"canonical asset/physics and hit-history migration required; do not replace PinkBigPig with a normal pig or classify load failure as model failure"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--publish",action="store_true"); modes.add_argument("--validate",action="store_true")
    parser.add_argument("--canonical",type=Path,default=ROOT/"sciencebirdsgames/Linux/9001_Data/resources.assets")
    parser.add_argument("--aligned",type=Path,default=ROOT/".local-artifacts/issue-76-compatibility-v1/runtimes/issue-76-compatibility-01/9001_Data/resources.assets")
    parser.add_argument("--output",type=Path,default=ROOT/"data/issue-76-compatibility/asset-audit.json")
    args = parser.parse_args()
    canonical,aligned = inspect_assets(args.canonical),inspect_assets(args.aligned)
    report = {"schema":"issue_76_canonical_asset_compatibility_audit_v1","reader":"UnityPy==1.25.3",
              "canonical":canonical,"aligned":aligned,"comparison":compare(canonical,aligned),
              "source_text":{SOURCE:(ROOT/SOURCE).read_text()},"metadata_only":True,"assets_changed":False,
              "no_gameplay_or_outcome_reads":True}
    content = json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n"
    if args.publish:
        if args.output.exists() and args.output.read_text() != content: raise ValueError("asset audit already exists with different evidence")
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(content)
        print(f"[issue-76-asset-audit] published {args.output}",flush=True)
    else:
        if args.output.read_text() != content: raise ValueError("asset audit differs from source prefab metadata")
        print("[issue-76-asset-audit] exact read-only asset audit validation passed; no asset substitution",flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
