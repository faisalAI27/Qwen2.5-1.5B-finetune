"""Package an explicit local research handoff; never upload or include credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from finalize_study import ROOT, ARCHIVE_SHA, EVALUATED_SHA, sha, require


def collect_paths(root, archive):
    root=Path(root).resolve(); paths=set()
    for folder in ['src','scripts','configs','manifests','docs','tests','notebooks']:
        for p in (root/folder).rglob('*'):
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.md','.txt','.yaml','.yml','.toml','.ipynb','.png','.svg','.pdf'):
                paths.add(p.relative_to(root).as_posix())
    paths.update(p.name for p in root.glob('requirements*.txt'))
    paths.update(['README.md','plan.md','pyproject.toml','NLP_Proposal.pdf','CITATION.cff','.gitignore','.gitattributes'])
    freeze=json.loads((root/'manifests/evaluation_freeze.json').read_text())
    paths.update(freeze['files_sha256'])
    with zipfile.ZipFile(archive) as z:
        export=json.loads(z.read('artifacts/vllm-export/export_manifest.json'))
    paths.update(export['inputs_sha256'])
    for folder in ['data/raw','results/protocol_dev/protocol-dev-v2','artifacts/remote-results/runpod-vllm-f816d62d7ab5']:
        paths.update(p.relative_to(root).as_posix() for p in (root/folder).rglob('*') if p.is_file() and not p.is_symlink())
    paths.update(['results/training/local-study-v1-trial-1/manifest.json',
                  'results/runtime/local_generation_retired.json','results/runtime/remote_completion_decision.json',
                  'results/runtime/PAUSE','results/demo/live-mac.json','results/demo/replay-rescued.json',
                  'results/demo/replay-spoiled.json','output/pdf/s1-small-model-report.pdf',
                  'output/source-history.bundle',archive.relative_to(root).as_posix()])
    paths.discard('docs/handoff.json')  # Added after computing the nonrecursive inventory.
    for n in paths:
        p=root/n
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root), 'Invalid handoff input: '+n)
        require(not any(x in p.relative_to(root).parts for x in ('.env','.aws','.codex','.git','.cache')), 'Unexpected private path')
    for n,h in {**freeze['files_sha256'],**export['inputs_sha256']}.items():
        require(sha(root/n)==h,'Frozen input differs: '+n)
    return sorted(paths)


def build(archive, destination):
    archive=Path(archive).resolve(); destination=Path(destination).resolve()
    require(destination.is_relative_to(ROOT/'output'),'Write the local handoff under output/')
    require(sha(archive)==ARCHIVE_SHA,'Result archive changed')
    destination.parent.mkdir(parents=True,exist_ok=True)
    history=ROOT/'output/source-history.bundle'
    subprocess.run(['git','bundle','create',str(history),'--all'],cwd=ROOT,check=True)
    paths=collect_paths(ROOT,archive)
    manifest={'status':'complete_local_handoff','scope':'private_research_handoff_not_a_public_redistribution_grant',
              'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'archive_sha256':ARCHIVE_SHA,'evaluated_sha256':EVALUATED_SHA,
              'benchmark_predictions':4500,'resolved_grades':4500,'original_mlx_adapter_included':True,
              'original_q4_base_included':True,'original_datasets_included':True,
              'reconstructed_fp16_weights_included':False,'environments_included':False,
              'source_history_bundle_included':True,'files_sha256':{n:sha(ROOT/n) for n in paths}}
    (ROOT/'docs/handoff.json').write_text(json.dumps(manifest,indent=2)+'\n')
    paths.append('docs/handoff.json')
    temp=destination.with_suffix('.zip.incomplete')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for n in paths:
            z.write(ROOT/n,n)
    # Complete CRC verification before promoting the package to its final name.
    with zipfile.ZipFile(temp) as z:
        require(z.testzip() is None,'Handoff CRC failure')
        for n,h in manifest['files_sha256'].items():
            digest=hashlib.sha256()
            with z.open(n) as f:
                for b in iter(lambda:f.read(1<<20),b''):digest.update(b)
            require(digest.hexdigest()==h,'Packaged content differs: '+n)
    temp.replace(destination)
    checksum=sha(destination)
    destination.with_suffix('.zip.sha256').write_text(checksum+'  '+destination.name+'\n')
    result={'archive':str(destination),'bytes':destination.stat().st_size,'sha256':checksum,'files':len(paths),'crc_and_all_file_hashes_verified':True}
    (ROOT/'output/handoff-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=ROOT/'nlp-runpod-vllm-results (1).zip')
    p.add_argument('--output',type=Path,default=ROOT/'output/s1-project-handoff.zip')
    a=p.parse_args();build(a.archive,a.output)
