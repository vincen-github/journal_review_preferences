from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from journal_prefs.__main__ import main
from journal_prefs.data import read_json, write_json, fingerprint

ROOT = Path(__file__).resolve().parent.parent


class AssessmentCLITests(unittest.TestCase):
    def test_no_score_cli_example_is_offline_and_snapshotted(self):
        with tempfile.TemporaryDirectory() as directory, patch('journal_prefs.assessment.make_model') as provider, \
             patch('socket.socket', side_effect=AssertionError('No network')), redirect_stdout(io.StringIO()):
            out = Path(directory) / 'output'
            task = ROOT / 'examples/assessment_task.synthetic.json'
            args = ['assess', '--task', str(task), '--request', str(ROOT / 'examples/assessment_request.synthetic.json'), '--out', str(out)]
            self.assertEqual(main(args), 0)
            result = read_json(out / 'assessment.json')
            self.assertTrue(result['synthetic_output'] and result['human_review_required'])
            self.assertFalse(result['automatic_editorial_decision'])
            self.assertNotIn('score', result['output'])
            self.assertIn('recommendation', result['output'])
            self.assertIn('editor_note', result['output'])
            self.assertIn('custom', result['output'])
            self.assertEqual(read_json(out / 'task_snapshot.json')['task_sha256'], fingerprint(read_json(task)))
            with redirect_stderr(io.StringIO()): self.assertEqual(main(args), 2)
            provider.assert_not_called()

    def test_target_label_rejected_before_provider_and_output(self):
        with tempfile.TemporaryDirectory() as directory, patch('journal_prefs.assessment.make_model') as provider, redirect_stderr(io.StringIO()):
            root = Path(directory)
            request = read_json(ROOT / 'examples/assessment_request.synthetic.json')
            request['target']['score'] = 9
            write_json(root / 'request.json', request)
            result = main(['assess', '--task', str(ROOT / 'examples/assessment_task.synthetic.json'), '--request', str(root / 'request.json'), '--out', str(root / 'out')])
            self.assertEqual(result, 2)
            self.assertFalse((root / 'out').exists())
            provider.assert_not_called()
