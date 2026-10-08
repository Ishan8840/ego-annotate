"""Fail closed unless all semantic outputs match the reference pipeline."""
import argparse,hashlib,json
from pathlib import Path

VOLATILE={'generated_at','path','source_interval_path','elapsed_s','usage'}

def normalize(obj):
    if isinstance(obj,dict):return {k:normalize(v) for k,v in obj.items() if k not in VOLATILE}
    if isinstance(obj,list):return [normalize(v) for v in obj]
    return obj


def compare(reference,candidate):
    names=['captions/captions.jsonl','captions/spans.jsonl','quality/analysis.json','quality/composite.json','audit/audit.json']
    checks=[]
    for name in names:
        a,b=reference/name,candidate/name
        if not a.exists() or not b.exists():
            checks.append(dict(file=name,identical=False,error='missing output'));continue
        read=lambda p: [json.loads(x) for x in p.read_text().splitlines()] if p.suffix=='.jsonl' else json.loads(p.read_text())
        na,nb=normalize(read(a)),normalize(read(b))
        checks.append(dict(file=name,identical=na==nb,
                          reference_hash=hashlib.sha256(json.dumps(na,sort_keys=True).encode()).hexdigest(),
                          candidate_hash=hashlib.sha256(json.dumps(nb,sort_keys=True).encode()).hexdigest()))
    return dict(all_semantic_outputs_identical=all(c['identical'] for c in checks),checks=checks,
                excluded_fields=sorted(VOLATILE),interpretation='Parity with the incumbent, not a claim that its labels are correct.')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=compare(a.reference,a.candidate);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['all_semantic_outputs_identical'] else 1)
if __name__=='__main__':main()
