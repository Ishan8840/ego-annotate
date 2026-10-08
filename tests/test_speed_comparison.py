"""Regression gate must reject changed labels, evidence, hashes, and missing files."""
import importlib.util
import json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('comparison',Path(__file__).parents[1]/'scripts/compare_speed_outputs.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

FILES=['captions/captions.jsonl','captions/spans.jsonl','quality/analysis.json','quality/composite.json','audit/audit.json']

def pair(tmp_path):
    roots=[tmp_path/'before',tmp_path/'after']
    for root in roots:
        for name in FILES:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({'text':'pick cup','sha256':'abc','evidence_t':1.5,'score':3,'generated_at':'yesterday','path':'/old/path','usage':{'tokens':7}})+'\n')
    return roots

def test_comparison_ignores_only_declared_execution_metadata(tmp_path):
    a,b=pair(tmp_path)
    for name in FILES:
        path=b/name;row=json.loads(path.read_text());row.update(generated_at='today',path='/new/path',usage={'tokens':8});path.write_text(json.dumps(row)+'\n')
    assert m.compare(a,b)['all_semantic_outputs_identical']

@pytest.mark.parametrize('key,value',[('text','drop cup'),('sha256','other'),('evidence_t',2.0),('score',2)])
def test_comparison_rejects_changed_semantic_fields(tmp_path,key,value):
    a,b=pair(tmp_path);path=b/FILES[0];row=json.loads(path.read_text());row[key]=value;path.write_text(json.dumps(row)+'\n')
    assert not m.compare(a,b)['all_semantic_outputs_identical']

def test_comparison_fails_closed_when_output_is_missing(tmp_path):
    a,b=pair(tmp_path);(b/FILES[-1]).unlink()
    assert not m.compare(a,b)['all_semantic_outputs_identical']
