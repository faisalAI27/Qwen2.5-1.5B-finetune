"""Replay one saved GPU comparison; this command never loads a model."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

from finalize_study import ROOT, ARCHIVE_SHA, sha, require, RUN


def replay(archive, question_id, intervention):
    require(sha(archive) == ARCHIVE_SHA, 'Archive differs from the verified run')
    records = {}
    with zipfile.ZipFile(archive) as z:
        with z.open(RUN + 'evaluated.jsonl') as f:
            for line in f:
                r = json.loads(line)
                if r['question_id'] == question_id and r['condition'] in ('C', intervention):
                    records[r['condition']] = {k:r[k] for k in ('condition','is_correct','gold_answer','extracted_answer','reasoning_tokens','total_generated_tokens','forced_wait_count','extraction_status','latency_seconds')}
                    records[r['condition']]['reasoning_tail'] = '\n'.join(s['text'][-700:] for s in r['segments'] if s['phase']=='reasoning' and s['origin']=='model')[-1000:]
                    if len(records)==2: break
    require(len(records)==2, 'Question/condition pair not found')
    return {'scope':'saved_gpu_output_replay_not_new_inference','question_id':question_id,
            'archive_sha256':ARCHIVE_SHA,'records':[records[c] for c in ('C',intervention)]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=ROOT/'nlp-runpod-vllm-results (1).zip')
    p.add_argument('--question-id',default='test/algebra/170.json')
    p.add_argument('--condition',choices=['D256','D512','D1024','D2048','E1','E2'],default='D256')
    p.add_argument('--output',type=Path)
    a=p.parse_args(); result=replay(a.archive,a.question_id,a.condition)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
