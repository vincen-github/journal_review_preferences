import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from journal_prefs.data import fingerprint, read_json, write_json
from tools import luna_rule_pilot as pilot
from tools import codex_json_generation as generation
from tools import run_codex_baseline as scoring


class LunaTests(unittest.TestCase):
    def test_exact_luna_generation_route_and_no_sol_fallback(self):
        events=[{'type':'item.completed','item':{'type':'agent_message','text':'{"rules":[]}'}}]
        result=SimpleNamespace(returncode=0,stdout='\n'.join(json.dumps(e) for e in events),stderr='route slug="gpt-6-luna"')
        with tempfile.TemporaryDirectory() as directory, patch.object(generation.subprocess,'run',return_value=result) as mocked:
            _,receipt=generation.generate({'examples':['SYNTHETIC']},'Return JSON',{'type':'object'},Path(directory)/'run','FAKE',True,model='gpt-6-luna')
            self.assertEqual(receipt['settings']['model'],'gpt-6-luna')
            self.assertTrue(receipt['reported_model_slug']); self.assertFalse(receipt['reported_sol_slug'])
            self.assertEqual(mocked.call_count,1)
            args=mocked.call_args.args[0]; self.assertEqual(args[args.index('--model')+1],'gpt-6-luna')
        result.stderr='route slug="gpt-6.1-sol"'
        with tempfile.TemporaryDirectory() as directory, patch.object(generation.subprocess,'run',return_value=result) as mocked:
            with self.assertRaises(ValueError):
                generation.generate({},'Return JSON',{'type':'object'},Path(directory)/'run','FAKE',True,model='gpt-6-luna')
            self.assertEqual(mocked.call_count,1)

    def test_luna_score_settings_and_model_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); text='SYNTHETIC manuscript'
            write_json(root/'in.json',[{'paper_id':'S','title':'SYNTHETIC','text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest()}])
            answer={'score':5,'review':'SYNTHETIC','evidence':[{'quote':text,'comment':'SYNTHETIC'}]}
            result=SimpleNamespace(returncode=0,stderr='route slug="gpt-6-luna"',stdout=json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps(answer)}}))
            with patch.object(scoring.subprocess,'run',return_value=result) as mocked, patch('builtins.print'):
                manifest=scoring.run(root/'in.json',root/'out','FAKE',True,model='gpt-6-luna')
            self.assertEqual(manifest['settings']['model'],'gpt-6-luna')
            self.assertTrue(manifest['runs'][0]['reported_model_slug'])
            self.assertEqual(mocked.call_count,1)

    def test_prepare_train_only_and_frozen_subset_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source'; sol=root/'sol'; out=root/'luna'
            source.mkdir(); (sol/'inputs').mkdir(parents=True)
            groups={g:[g+str(i) for i in range(n)] for g,n in (('train',60),('validation',20),('test',20))}
            write_json(source/'partition.json',{'ids':groups,'partition_sha256':fingerprint(groups)})
            history=[]
            for g in groups:
                rows=[{'paper_id':i,'title':'SYNTHETIC','text':'SYNTHETIC '+i,'text_sha256':hashlib.sha256(('SYNTHETIC '+i).encode()).hexdigest()} for i in groups[g]]
                if g=='train': rows=rows[:6]
                write_json(sol/'inputs'/(g+'.json'),rows)
                if g=='train': history=[dict(r,reviews=[],individual_scores=[5],human_mean=5) for r in rows]
            write_json(sol/'train_history.json',history)
            (sol/'runs').mkdir()
            for c in ('A','B'):
                (sol/'generation_runs'/c).mkdir(parents=True)
                write_json(sol/'generation_runs'/c/'receipt.json',{'status':'completed','subprocess_attempts':1})
            catalog=root/'catalog.json'; write_json(catalog,{'models':[{'slug':pilot.MODEL,'visibility':'list','supported_reasoning_levels':[{'effort':'low'}]}]})
            prior={'protocol_sha256':'SYNTHETIC','generation_excerpt_policy':'SYNTHETIC'}
            def local_read(path):
                self.assertNotIn('.labels',str(path)); self.assertNotIn('metadata_frame',str(path)); return read_json(path)
            with patch.object(pilot,'verify_sol_inputs',return_value=prior), patch.object(pilot,'read_json',side_effect=local_read):
                p=pilot.prepare(source,sol,out,catalog)
                pilot.check(source,out)
            self.assertEqual(p['total_calls_planned_including_sol'],76)
            self.assertEqual(p['ids']['test'],groups['test'][::2])
            self.assertEqual(len(p['ids']['train']),6)
            with patch.object(pilot,'run') as model:
                with self.assertRaises(FileNotFoundError): pilot.score(source,out,'G','test','FAKE',True)
                model.assert_not_called()
            changed=read_json(out/'inputs/test.json'); changed[0]['human_score']=8; write_json(out/'inputs/test.json',changed)
            with self.assertRaises(ValueError): pilot.check(source,out)
