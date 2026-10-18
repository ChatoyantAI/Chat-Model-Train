import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation.roleplay_benchmarks import coser_rejudge as r

ROOT = Path(__file__).resolve().parents[2]


class Store:
    def __init__(self, *args, **kwargs): self.writer = self; self.ledger = self
    def write(self, path, value): r.write_json(path, value)
    def append(self, value): pass
    def flush(self): pass
    def close(self): pass
    def check(self): pass


class RejudgeTests(unittest.TestCase):
    def test_labels_preserve_content_thoughts_and_existing_prefixes(self):
        sim=[{'role':'A','content':'[secret]\n(acts) Hello'},
             {'role':'B','content':'**B:** Yes'},
             {'role':'Environment','content':'Rain'},
             {'role':'A','content':'B: I am confused'}]
        before=json.dumps(sim)
        self.assertEqual(r.labelled_transcript(sim), 'A: [secret]\n(acts) Hello\n\n**B:** Yes\n\nRain\n\nA: B: I am confused')
        self.assertEqual(json.dumps(sim),before)

    def test_schema_and_original_score_formula(self):
        c=r.parse_critique('```json\n{"Storyline Quality":{"flaws":[{"severity":3},{"severity":2},{"severity":2},{"severity":2}]}}\n```','Storyline Quality')
        self.assertEqual(r.score(c,20),85)
        for text in ['{}','{"Storyline Quality":{"flaws":[{"severity":true}]}}',
                     '{"Storyline Quality":{"flaws":[{"severity":0}]}}']:
            with self.assertRaises(ValueError):r.parse_critique(text,'Storyline Quality')

    def test_paired_prepare_run_and_zero_charge_replay(self):
        cfg=json.loads((ROOT/'configs/evaluation/roleplay_benchmarks/historical/gemma4-e2b-it-coser-20260926.json').read_text())
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{
            cfg['judge']['base_url_env']:'http://localhost:9911/v1',cfg['judge']['api_key_env']:'test'}):
            root=Path(tmp);sources=[];before={}
            for group in range(2):
                src=root/f'source-{group}';child=src/'coser';out=child/'outputs/full-public-v1/candidate/coser'
                out.mkdir(parents=True);r.write_json(src/'manifest.json',{'config':cfg});sources.append(src)
                sim=[{'role':'A','content':('A: ' if group else '')+'[thought] Hello'}, {'role':'Environment','content':'Rain'}]
                text='\n\n'.join(m['content'] for m in sim)
                result={'index':0,'book':'Book','scores':{d:5 for d in r.DIMENSIONS},'cases':{'Book-0-0':{'simulation':sim,'simulation_str':text}}}
                path=out/'case-0000/result.json';r.write_json(path,result)
                r.write_json(out/'status-0000.json',{'index':0,'status':'completed','result':str(path.relative_to(child))})
                for i,dim in enumerate(r.DIMENSIONS):
                    q={'model':cfg['judge']['model'],'messages':[{'role':'system','content':'Evaluate in terms of '+dim},{'role':'user','content':text}],'temperature':0,'max_tokens':8192,'stream':False}
                    response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({dim:{'flaws':[]}})}}]}
                    r.write_json(src/f'transport-journal/aa/key/{i}.outcome.json',{'model':cfg['judge']['model'],'status':'response','request':q,'response':response})
                for path in src.rglob('*'):
                    if path.is_file():before[path]=path.read_bytes()
            dest=root/'new';m=r.prepare(sources,dest)
            self.assertEqual(m['expected_judgments'],8)
            self.assertEqual([x['changed_judgments'] for x in m['sources']],[4,0])
            calls=[]
            def transport(role,payload,timeout):
                self.assertEqual(role['model'],cfg['judge']['model'])
                self.assertTrue(payload['messages'][1]['content'].startswith('A: [thought]'))
                dim=payload['messages'][0]['content'].split('in terms of ')[1]
                calls.append(payload)
                return {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({dim:{'flaws':[]}})}}], 'usage':{'estimated_cost':.01}}
            with patch.object(r,'DerivedStore',Store),patch.object(r,'UsageLedger',Store),patch.object(r.api,'transport',transport):
                first=r.run(dest,True);self.assertEqual(first['status'],'completed');self.assertEqual(len(calls),8)
                second=r.run(dest,True);self.assertEqual(second['status'],'completed');self.assertEqual(len(calls),8)
                self.assertEqual(second['transport']['replays'],8)
            for path,raw in before.items():self.assertEqual(path.read_bytes(),raw)
            # Hash-frozen requests cannot be silently changed on resume.
            (dest/'inputs.json').write_text('{}')
            with self.assertRaises(ValueError):r.run(dest,True)


if __name__=='__main__':unittest.main()
