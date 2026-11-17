"""Checks for final provenance, diagnostic semantics and honest report claims."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from finalize_study import trace_diagnostics, load_verified, tables, ARCHIVE_SHA, EVALUATED_SHA


class FinalHandoffTests(unittest.TestCase):
    def test_trace_comparison_distinguishes_cap_from_divergence(self):
        rows=[]
        for c,tokens in [('C',[1,2,3]),('D256',[1,2]),('D512',[1,8]),('D1024',[1,2,3]),('D2048',[1,2,3]),('E1',[1,9]),('E2',[1,2,3])]:
            rows.append({'question_id':'synthetic','condition':c,'forced_wait_count':1 if c=='E1' else 0,
                         'segments':[{'origin':'injected','phase':'reasoning','token_ids':[999]},
                                     {'origin':'model','phase':'reasoning','token_ids':tokens,'removed_stop_token_id':151645}]})
        d=trace_diagnostics(rows)
        self.assertEqual(d['first_reasoning_segments_vs_C']['D256']['prefix_compatible'],1)
        self.assertEqual(d['first_reasoning_segments_vs_C']['D256']['exact_first_segment_matches'],0)
        self.assertEqual(d['first_reasoning_segments_vs_C']['D512']['prefix_compatible'],0)
        self.assertEqual(d['first_reasoning_segments_vs_C']['E1']['exact_first_segment_matches'],0)
        self.assertEqual(d['actual_wait_count_histogram']['E1'],{'1':1})
        self.assertEqual(d['removed_reasoning_stop_ids']['C'],{'151645':1})

    def test_altered_archive_rejected_before_reporting(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'wrong.zip';p.write_bytes(b'not the frozen result archive')
            with self.assertRaisesRegex(ValueError,'Archive differs'):
                load_verified(p)

    def test_final_results_keep_complete_gpu_denominators(self):
        d=json.loads((ROOT/'docs/final_results.json').read_text())
        self.assertEqual(d['archive_sha256'],ARCHIVE_SHA)
        self.assertEqual(d['evaluated_sha256'],EVALUATED_SHA)
        self.assertEqual(sum(s['correct'] for s in d['conditions'].values()),1250)
        self.assertEqual(sum(s['resolved_grades'] for s in d['conditions'].values()),4500)
        self.assertTrue(all(s['planned']==500 and s['complete'] for s in d['conditions'].values()))
        self.assertEqual(d['mac_partial_outputs_excluded'],1287)
        self.assertEqual(d['paired_vs_C']['E2']['transitions']['wrong_to_right'],19)
        self.assertEqual(d['paired_vs_C']['E2']['transitions']['right_to_wrong'],18)
        self.assertFalse(d['further_training'])
        scores, medians, paired=tables(d)
        self.assertIn('N/A',scores)
        self.assertIn('[-2.2, 2.6]',paired)
        self.assertIn('| D1024 |',scores)

    def test_report_preserves_negative_findings_and_caveats(self):
        s=(ROOT/'docs/paper.md').read_text()
        for phrase in ['43.0%','28.6%','82.1%','not statistical equivalence','pre-intervention','Limitations','Hardware and measured runtime']:
            # Wording can be edited without changing the underlying scientific checks.
            if phrase=='not statistical equivalence':
                self.assertIn('does not establish statistical equivalence',s)
            else:self.assertIn(phrase,s)
        self.assertNotIn('Mac-only inference',s)


if __name__=='__main__':unittest.main()
