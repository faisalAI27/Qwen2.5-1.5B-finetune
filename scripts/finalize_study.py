"""Build the final report from the audited GPU archive; never launch inference."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = 'f816d62d7ab56dbbc0da0327469c4b89f0775b35647e221e22d77c64d9edcaee'
EVALUATED_SHA = 'e392495e76ea56b4505aa82e79c3045a1ee87f18d8edd02a34fb6179ffcd5413'
RUN = 'results/test/math500-runpod-vllm-v1/'
ORDER = ['A', 'B', 'C', 'D256', 'D512', 'D1024', 'D2048', 'E1', 'E2']
LABELS = ['Base direct', 'Base reasoning', 'Adapted natural*', 'Cap 256', 'Cap 512', 'Cap 1024', 'Cap 2048', 'Wait up to 1', 'Wait up to 2']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def trace_diagnostics(rows):
    """Retain hashes, not full token arrays, across the streamed benchmark."""
    first = defaultdict(dict)
    waits, stops = defaultdict(Counter), defaultdict(Counter)
    for r in rows:
        c = r['condition']
        waits[c][str(r['forced_wait_count'])] += 1
        segments = [s for s in r['segments'] if s['origin'] == 'model' and s['phase'] == 'reasoning']
        for s in segments:
            stops[c][str(s.get('removed_stop_token_id', 'cap'))] += 1
        if segments:
            # At most 500 x 7 x 32 KiB; bounded memory, no transcripts retained.
            first[r['question_id']][c] = b''.join(int(t).to_bytes(4, 'little') for t in segments[0]['token_ids'])
    comparisons = {}
    for c in ORDER[3:]:
        pairs = [(d['C'], d[c]) for d in first.values()]
        comparisons[c] = {
            'question_pairs': len(pairs),
            'exact_first_segment_matches': sum(a == b for a, b in pairs),
            'prefix_compatible': sum(a.startswith(b) or b.startswith(a) for a, b in pairs),
        }
    return {
        'scope': 'posthoc_trace_diagnostic_no_regeneration',
        'first_reasoning_segments_vs_C': comparisons,
        'removed_reasoning_stop_ids': {c: dict(v) for c, v in stops.items()},
        'actual_wait_count_histogram': {c: dict(v) for c, v in waits.items()},
        'interpretation': 'Conditions can diverge before intervention under concurrent FP16 inference. The exact cause was not isolated; no token-identical counterfactual claim is supported.',
    }


def load_verified(archive):
    require(sha(archive) == ARCHIVE_SHA, 'Archive differs from the reviewed result archive')
    review = ROOT / 'artifacts/remote-results/runpod-vllm-f816d62d7ab5/verification.json'
    require(review.exists(), 'Run scripts/audit_runpod_archive.py first')
    verification = json.loads(review.read_text())
    require(verification['status'] == 'verified' and verification['archive_sha256'] == ARCHIVE_SHA
            and verification['analysis_recomputed_and_matched'], 'Missing successful provenance/reanalysis audit')
    with zipfile.ZipFile(archive) as z:
        analysis = json.loads(z.read(RUN + 'analysis.json'))
        summary = json.loads(z.read(RUN + 'summary.json'))
        h = hashlib.sha256()
        counts = Counter()
        seen = set()
        def rows():
            with z.open(RUN + 'evaluated.jsonl') as f:
                for line in f:
                    h.update(line)
                    r = json.loads(line)
                    key = r['question_id'], r['condition']
                    require(key not in seen, 'Duplicate prediction')
                    seen.add(key)
                    require(r['status'] == 'ok' and isinstance(r['is_correct'], bool), 'Unresolved model output/grade')
                    counts[r['condition']] += 1
                    yield r
        diagnostics = trace_diagnostics(rows())
        require(h.hexdigest() == EVALUATED_SHA == analysis['evaluated_sha256'], 'Evaluated data hash differs')
        require(counts == Counter({c: 500 for c in ORDER}), 'Incomplete benchmark')
        require(analysis['status'] == 'complete', 'Analysis is partial')
        for c in ORDER:
            s = analysis['conditions'][c]
            require(s['complete'] and s['resolved_grades'] == 500 and s['model_outputs'] == 500, 'Incomplete condition ' + c)
    return analysis, summary, diagnostics, verification


def tables(analysis):
    scores = ['| ID | Condition | Correct / 500 | Accuracy | Mean reasoning | Mean total | Mean seconds | Extraction failures |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    medians = ['| ID | Median reasoning | Median total | Median seconds | Reason cap | Safety cap | Answer cap |',
               '|---|---:|---:|---:|---:|---:|---:|']
    paired = ['| vs C | Difference (pp) | 95% paired interval (pp) | Rescued | Spoiled |', '|---|---:|---:|---:|---:|']
    for c, label in zip(ORDER, LABELS):
        s = analysis['conditions'][c]
        n = s['mean_reasoning_tokens']
        scores.append(f"| {c} | {label} | {s['correct']} | {s['accuracy']*100:.1f}% | {'N/A' if n is None else f'{n:.1f}'} | {s['mean_total_generated_tokens']:.1f} | {s['mean_latency_seconds']:.2f} | {s['extraction_failures']} |")
        n = s['median_reasoning_tokens']
        medians.append(f"| {c} | {'N/A' if n is None else f'{n:g}'} | {s['median_total_generated_tokens']:g} | {s['median_latency_seconds']:.2f} | {s['reasoning_cap_hits']} | {s['safety_cap_hits']} | {s['answer_cap_hits']} |")
        if c != 'C':
            p = analysis['paired_vs_C'][c]
            lo, hi = p['confidence_interval']
            t = p['transitions']
            paired.append(f"| {c} | {p['accuracy_difference']*100:+.1f} | [{100*lo:.1f}, {100*hi:.1f}] | {t['wrong_to_right']} | {t['right_to_wrong']} |")
    return '\n'.join(scores), '\n'.join(medians), '\n'.join(paired)


def plots(analysis, directory):
    import os
    cache = ROOT / 'tmp/matplotlib'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    s = analysis['conditions']
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.25))
    colors = ['#83909c', '#147d92'] + ['#b86725'] * 7
    axes[0].bar(ORDER, [100*s[c]['accuracy'] for c in ORDER], color=colors)
    axes[0].set(ylabel='Accuracy (%)', ylim=(0, 51), title='All 500 questions per condition')
    for i,c in enumerate(ORDER): axes[0].text(i, 100*s[c]['accuracy']+1, f"{100*s[c]['accuracy']:.1f}", ha='center', fontsize=8)
    axes[0].tick_params(axis='x', rotation=40)
    for i,c in enumerate(ORDER):
        axes[1].scatter(s[c]['mean_total_generated_tokens'], 100*s[c]['accuracy'], c=colors[i], s=34)
        offsets = {'A': (4,-12), 'B': (4,4), 'C': (-4,8), 'D256': (4,2),
                   'D512': (-29,-8), 'D1024': (-15,12), 'D2048': (3,-12),
                   'E1': (-9,-15), 'E2': (4,6)}
        axes[1].annotate(c, (s[c]['mean_total_generated_tokens'], 100*s[c]['accuracy']), xytext=offsets[c], textcoords='offset points', fontsize=9)
    axes[1].set(xlabel='Mean total generated tokens', ylabel='Accuracy (%)', title='Total output cost, including answers', ylim=(14,49), xlim=(-150,6000))
    fig.tight_layout()
    for ext in ['png','svg']: fig.savefig(directory / ('final_accuracy_cost.'+ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
    cs = [c for c in ORDER if c != 'C']
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.3))
    ys=list(range(len(cs)))
    p=analysis['paired_vs_C']
    for i,c in enumerate(cs):
        v=p[c]['accuracy_difference']*100; lo,hi=[100*x for x in p[c]['confidence_interval']]
        axes[0].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#147d92',capsize=3)
    axes[0].set(yticks=ys,yticklabels=cs,xlabel='Accuracy difference vs C (pp)',title='Paired bootstrap, 10,000 resamples')
    axes[0].axvline(0,c='#888888',lw=.8);axes[0].invert_yaxis()
    axes[1].barh([i-.17 for i in ys],[p[c]['transitions']['wrong_to_right'] for c in cs],height=.33,label='Rescued',color='#147d92')
    axes[1].barh([i+.17 for i in ys],[p[c]['transitions']['right_to_wrong'] for c in cs],height=.33,label='Spoiled',color='#b86725')
    axes[1].set(yticks=ys,yticklabels=cs,xlabel='Question count',title='Both directions of answer change');axes[1].invert_yaxis();axes[1].legend(frameon=False,fontsize=9)
    fig.tight_layout()
    for ext in ['png','svg']: fig.savefig(directory / ('final_paired_transitions.'+ext), dpi=200, bbox_inches='tight')
    plt.close(fig)
    # Retain the original five validated plots, including reasoning-only and latency.
    import shutil
    source=ROOT/'artifacts/remote-results/runpod-vllm-f816d62d7ab5'/RUN/'figures'
    for path in source.glob('*'):
        if path.suffix in ('.png','.svg'): shutil.copyfile(path,directory/path.name)


def error_appendix(analysis):
    lines=['# Qualitative audit and answer transitions', '',
           'All 24 examples below use the frozen analysis rule: lexicographically first question ID in each of four transition categories for each D/E condition. These are descriptive cases, not a representative sample. No examples were selected for favorable outcomes. Full raw segments are retained in the local result ZIP; this public summary contains IDs and extracted answers only.', '',
           '| Condition | Category | Question ID | Gold | C answer | Intervention answer | C / intervention reasoning tokens | Actual Wait |',
           '|---|---|---|---|---|---|---:|---:|']
    def cell(v): return str(v).replace('|',' / ').replace('\n',' ')[:130]
    for e in analysis['error_cases']:
        require(e['available'], 'Missing qualitative category')
        lines.append('| '+' | '.join(cell(e[k]) for k in ['condition','category','question_id','gold_answer','natural_answer','intervention_answer'])+f" | {e['natural_reasoning_tokens']} / {e['intervention_reasoning_tokens']} | {e['forced_wait_count']} |")
    lines += ['', '## Reading the cases', '',
      '- D256, algebra/170: C lists the correct integer range, then repeatedly counts overlapping endpoints and answers 21. The cap answers 11. The trace supports repetition and counting drift as a failure pattern; it does not prove shorter reasoning always helps.',
      '- D256, algebra/1031: C answers 5.5, equivalent to the gold 11/2; the capped condition answers 10. Early answer transition can also lose a correct result. Longer D512/D1024 caps spoil this ID too, and their first segment can differ from C.',
      '- D1024, algebra/1960: C answers 3; D1024 answers 9. C used only 101 reasoning tokens while D1024 used 354, both below the cap. This is not evidence that truncation alone rescued the example: independent trajectories already differ.',
      '- Algebra/1004 stays wrong in the selected cap cases: the extracted body is an equation, 8 = 2x, rather than the gold value 4. An explicit-answer extraction success does not establish a correct, fully resolved answer.',
      '- E1/E2 categories include both rescued and spoiled answers and cases with no actual Wait. The policy is conditional on a stop, so a nominal E condition is not evidence that continuation occurred. See the actual Wait column and the aggregate trace diagnostic.', '',
      '## Aggregate checks', '',
      'E1 rescues 17 and spoils 24 answers; E2 rescues 19 and spoils 18. E1 injects one Wait on 264/500 outputs. E2 injects two on 208 outputs, one on 54, and none on 238. There are no intercepted learned reasoning role-marker stops in the saved full test; intercepted stops are model EOS tokens (151645). This implementation therefore tests an EOS-based continuation policy, not proof of the original learned delimiter behavior.', '',
      'First reasoning segments match C exactly for 395/500 E1 and 393/500 E2 outputs. For cap conditions, prefix compatibility is checked because shorter output is expected. Differences before intervention are retained and reported; their numerical cause was not isolated. No outputs were regenerated or grades repaired after inspection.', '']
    return '\n'.join(lines)


def write_documents(analysis, summary, diagnostics, verification):
    docs=ROOT/'docs'; scores, medians, paired=tables(analysis)
    c=analysis['conditions']; reduction=100*(1-c['D1024']['mean_total_generated_tokens']/c['C']['mean_total_generated_tokens'])
    extra=100*(c['E2']['mean_total_generated_tokens']/c['C']['mean_total_generated_tokens']-1)
    compact={k:v for k,v in analysis.items() if k not in ('error_cases','figures')}
    compact.update(archive_sha256=ARCHIVE_SHA, generation_summary=summary, trace_diagnostics=diagnostics,
                   final_backend=verification['backend'], gpu=verification['gpu'],
                   mac_partial_outputs_excluded=1287, optional_aime_run=False, further_training=False,
                   d1024_total_token_reduction_percent=reduction, e2_extra_total_tokens_percent=extra)
    (docs/'final_results.json').write_text(json.dumps(compact,indent=2)+'\n')
    (docs/'error_cases.md').write_text(error_appendix(analysis))
    (docs/'results.md').write_text('# Final MATH-500 results\n\nThe complete, separately frozen FP16/vLLM RunPod run is the final benchmark. All 4,500 predictions and grades were verified; the partial Mac run is excluded.\n\n'+scores+'\n\n*Natural stopping has an 8,192-token reasoning safety ceiling; it is not unlimited. A/B reasoning counts are unavailable because their output is unstructured. All means include the full 500-question denominator. Seconds are concurrent request latency, not isolated latency.\n\n## Medians and cap events\n\n'+medians+'\n\nEvent categories can overlap; a reasoning cap is intentional in D conditions.\n\n## Paired uncertainty\n\n'+paired+'\n\n10,000 paired question-level bootstrap samples, seed 42, percentile 95% intervals; unadjusted and descriptive, with no equivalence margin. Failures remain in the 500-question denominator.\n\n'+f'D1024 matches the observed C accuracy with {reduction:.1f}% fewer total generated tokens. This is an observed efficiency result, not statistical equivalence or a new test-selected default. E2 uses {extra:.1f}% more tokens and gains one net correct answer. Base reasoning B is strongest, but B vs C changes both prompting/readout and adapter, so their gap is not a pure adapter effect.\n\n'+'![Accuracy and total cost](figures/final_accuracy_cost.png)\n\n![Paired intervals and transitions](figures/final_paired_transitions.png)\n\nSee [the qualitative audit](error_cases.md), [the paper](paper.md), and [machine-readable results](final_results.json).\n')
    paper = f'''# Budget Forcing on a 1.5B Model: A Resource-Constrained Study

Faisal Imran | Muhammad Ibrahim | Hamza Saeed

Research project report | 17 November 2026 | Completed single-run study

## Abstract

We study s1-style supervised adaptation and inference-time budget forcing with Qwen2.5-1.5B-Instruct under limited local resources. One LoRA run trains on 896 complete original s1K examples using an 8 GB Apple M1; 100 development examples remain held out. A separately frozen FP16/vLLM evaluation reconstructed from the trained four-bit base produces all 4,500 planned MATH-500 outputs. The reasoning-prompted base scores 43.0%, compared with 28.6% for adapted bounded natural stopping. A 1,024-token reasoning cap also scores 28.6% while using {reduction:.1f}% fewer generated tokens. Up to two Wait continuations score 28.8%, with a paired difference of +0.2 percentage points and a 95% interval of [-2.2, 2.6], at {extra:.1f}% higher token cost. We find no clear continuation benefit in this run. Frequent safety-cap hits, a shared injected answer label, and pre-intervention trajectory divergence constrain causal interpretation. The result is a documented small-model adaptation, not a reproduction of the original 32B model's performance.

## 1. Introduction

Longer reasoning is useful only if its accuracy gain justifies its cost. The original s1 work combines a curated 1,000-example dataset with budget forcing, including shorter reasoning caps and Wait-based continuation [1]. Our question is whether the same broad idea remains useful with a 1.5B model and hardware that cannot support the original training setup.

We contribute a complete constrained implementation, a source-traced Mac-to-CUDA export, and a nine-condition evaluation with per-question provenance. The important outcome is mixed: held-out teacher-target loss improves after training, yet the adapted evaluation policy does not beat the reasoning-prompted base. Shorter caps often save substantial output cost; additional continuation has uncertain benefit. We retain all outcomes, including extraction failures and spoiled answers.

## 2. Related work and scope

Chain-of-thought prompting studies intermediate reasoning as part of generation [2]; it motivates a strong reasoning-prompted base control rather than comparison only with direct answers. LoRA freezes base weights and learns low-rank updates [3]. We combine that method with affine four-bit MLX weights and FP32 activations, rather than claiming the original s1 full-training procedure. vLLM provides an inference serving system based on PagedAttention [4]. We use it to make evaluation practical, and keep its FP16 results separate from the original Mac run. Qwen2.5 [5] supplies the pretrained model family; MATH [6] supplies the underlying competition-mathematics benchmark lineage.

Original s1 contributions are dataset curation and budget forcing [1]. Our contribution is a measured small-model adaptation and failure analysis. We do not recreate teacher data, reproduce the 32B scores, or claim new scaling laws. The public s1 repository now includes later variants; this project pins the original s1K rather than silently using s1K-1.1.

<!-- pagebreak -->

## 3. Data, training and inference policy

### 3.1 Data and accepted training run

The seeded raw split is 900 training / 100 development examples (seed 42). Four conservatively identified benchmark overlaps are excluded from training, leaving 896 training and 100 development examples. One normalized duplicate pair remains wholly in training and is disclosed. Exact-match and normalized overlap checks do not rule out semantic contamination or pretraining exposure. Eligible sequences span 712-9,109 tokens and are kept complete under a 9,216-token training ceiling; no examples are dropped or truncated for length.

The formatter supervises the teacher thinking trajectory and answer while masking the user prompt and padding. All 28 attention layers receive q/k/v/o LoRA updates: rank 16, alpha 32, scale 2, dropout 0.05, 4,358,144 trainable parameters. AdamW uses learning rate 5e-5, micro-batch 1, accumulation 16 and one epoch: 56 updates over all 896 examples. The accepted MLX path uses affine Q4 weights, group size 64, and FP32 activations/adapters after numerical and memory failures on earlier PyTorch paths. Fresh-process adapter tensor and generation reload checks pass.

The user accepted this single run to limit compute. No second trial or fresh 996-example final retraining was performed; all 100 development examples remain held out. On identical 555,371 development target tokens, teacher-target NLL decreases from 0.563623 to 0.479717 (14.9% relative). This measures imitation of teacher tokens, not benchmark answer accuracy.

### 3.2 Protocol and development-only readout repair

A uses a direct-answer base prompt; B asks the base to reason. C uses the adapted structured reasoning/answer format. D256/D512/D1024/D2048 cap its reasoning phase. E1/E2 permit up to one/two Wait injections when reasoning stops. C and E share an 8,192-token reasoning ceiling, a separate 1,536-token answer allowance and a 12,288-token absolute ceiling including control tokens. A/B have a 9,728-token unstructured allowance. Greedy decoding and a 900-second timeout are fixed before testing.

The first 45 development outputs exposed failed structured answer extraction. Fifteen same-adapter development probes motivated a common injected `Final Answer: ` label after the training-format answer header. All C/D/E conditions use the same label. The raw generated body and injected label have separate provenance; an empty body remains an extraction failure. The repeated 45-output gate yields 40 extractable outputs. This is a development-only behavior check, not benchmark tuning or an accuracy score.

Stop tokens are logged and charged to generated-token budgets even when removed before continuation. Injected Wait and answer-transition tokens count toward absolute context use, separately from generated output. The final test shows EOS-based reasoning stops and no intercepted learned role-marker stops. Thus this study evaluates the declared adapted policy, not established reproduction of the original learned reasoning delimiter.

### 3.3 Separate GPU evaluation and grading

For RunPod, the frozen Q4 codes, scales and biases are reconstructed as scale * code + bias and rounded to FP16. The original dense pretrained checkpoint is not substituted. All 224 LoRA matrices are transposed exactly into PEFT format; source-adapter fingerprints and prompt token IDs are verified. Maximum base FP16 rounding error is 0.000244140625. No optimization occurs on the GPU. This export changes numerical execution, so FP16/CUDA is a separate frozen experiment.

A single RTX 4090 runs eight concurrent state machines through one vLLM engine. Answer extraction and symbolic/numeric equivalence use frozen math-verify 0.9.0 and SymPy 1.14.0. An absent extracted answer is marked incorrect; malformed and unextracted outputs remain in the 500-question denominator. Raw prompts, segments, grades and hashes are saved. Grader equivalence and extraction remain imperfect; all test gold answers were checked against the pinned source.

<!-- pagebreak -->

## 4. Results and uncertainty

{scores}

Table 1. Complete MATH-500 matrix. Each condition has 500 model outputs and 500 resolved grades. Reasoning tokens are unavailable for unstructured A/B. Total tokens include answers and emitted stop tokens but exclude injected controls. *C has a finite safety ceiling. Seconds are shared-scheduling request latency, not isolated single-request latency.

{paired}

Table 2. Paired differences relative to C, in percentage points. Intervals use 10,000 paired question-ID bootstrap samples with replacement, seed 42, percentile 95% coverage. All 500 pairs enter each comparison. Intervals are descriptive and unadjusted for multiple comparisons; no equivalence test was prespecified.

B is the strongest observed policy (215/500), ahead of C (143/500). This comparison changes both prompt/readout and adapter, so it cannot identify the causal effect of fine-tuning alone. D1024 matches C's observed accuracy while reducing mean total output from 4,178.5 to 749.1 tokens. Its interval [-1.8, 2.0] does not establish statistical equivalence. The cap was predeclared; this post-hoc efficiency observation is not permission to retune on the test set.

E1 rescues 17 answers but spoils 24, for -1.4 pp [-3.8, 1.0]. E2 rescues 19 and spoils 18, for +0.2 pp [-2.2, 2.6]. The benchmark gives no clear evidence that continuation improves the adapted model. E2 uses 5,441.0 mean tokens versus C's 4,178.5. Extraction succeeds on 4,381/4,500 outputs; the 119 failures are counted as incorrect. There are zero infrastructure failures and zero timeouts, and every output succeeds on its first recorded attempt.

<!-- pagebreak -->

## 5. Budget behavior and failure analysis

C hits its reasoning/safety ceiling on 238/500 outputs (47.6%); E1 hits it on 286 (57.2%) and E2 on 303 (60.6%). A bounded natural baseline is therefore often censored by its safety ceiling. D256/D512/D1024/D2048 hit their intended reasoning caps on 434/318/256/242 outputs. Answer caps occur on 10/2/6/3 outputs respectively, versus 10 for C, 7 for E1 and 12 for E2. Cap categories can overlap and are not infrastructure errors.

E1 actually injects Wait on 264 questions. E2 injects two Waits on 208, one on 54, and none on 238. Extra opportunities do not imply extra continuation for every question. The final trace audit also finds identical first reasoning segments versus C for only 395/500 E1 and 393/500 E2 outputs. Greedy concurrent FP16 execution is not a token-identical counterfactual control; the precise source of divergence was not isolated. Some answer changes precede the intended intervention, limiting attribution to Wait itself.

![Accuracy and total token cost](figures/final_accuracy_cost.png)

Figure 1. Full-denominator accuracy and measured output cost. The strongest observed policy is base reasoning B. C/D/E do not establish monotonic accuracy improvement with more reasoning tokens.

<!-- pagebreak -->

![Paired uncertainty and answer transitions](figures/final_paired_transitions.png)

Figure 2. Unadjusted paired 95% intervals and both directions of answer change. A/B vs C include prompt and adapter changes; D/E are declared policy comparisons with pre-intervention divergence caveats.

### 5.1 Qualitative cases

The frozen selection rule chooses the lexicographically first question ID in each rescued, spoiled, unchanged-wrong and unchanged-right category for every D/E policy: 24 cases, all retained in the appendix file. At `test/algebra/170.json`, C repeatedly counts overlapping integer-range endpoints and answers 21; D256 answers the gold 11. At `test/algebra/1031.json`, C answers the correct 5.5, while D256 answers 10. These traces illustrate both repetition drift and loss of a correct answer under a changed policy.

At `test/algebra/1960.json`, D1024 changes C's incorrect 3 to the correct 9 despite both initial reasoning lengths being below the cap. The first trajectories already differ, so the example does not isolate truncation as the cause. At `test/algebra/1004.json`, several policies retain an equation rather than the requested resolved value; explicit answer extraction alone is not correctness. These examples are descriptive, and intermediate reasoning is not treated as a faithful account of internal computation.

## 6. Conclusion

A complete small-model experiment is feasible with local adapter training and separately validated GPU inference. In this single accepted run, s1-style adaptation does not yield a policy that beats the reasoning-prompted base. A 1,024-token cap preserves observed adapted accuracy at much lower output cost, while extra Wait continuation has uncertain benefit and substantially higher cost. The central result is an efficiency observation and a limit on transfer to this setup, not a refutation of original s1 or proof that more compute never helps. Stronger causal claims would require matched prompt controls, controlled numerical execution and additional training seeds; those experiments were not run.

<!-- pagebreak -->

## Limitations

Only one training seed, one accepted recipe and one 500-question benchmark were evaluated. The original plan's second learning-rate trial and fresh all-eligible-data retraining were deliberately skipped. We cannot infer variation across training seeds, generality across domains, or performance of an optimally trained 1.5B model. AIME24 and the optional combined cap-plus-Wait condition were not run.

Model size, LoRA training, four-bit base weights, FP32 training activations, FP16 reconstructed evaluation and the answer-label repair all depart from original s1. NLL improvement does not isolate mathematical reasoning ability. A/B and C/D/E use different output protocols, so the observed B-C difference is a policy comparison, not a controlled adapter ablation. The answer label is supplied by the system, not learned evidence of spontaneous answer formatting.

Frequent safety stops censor natural reasoning, and EOS continuation is not established learned-delimiter continuation. Greedy batched trajectories can diverge before intervention. Bootstrap intervals describe question sampling conditional on this run; they do not include training, numerical, dataset contamination or grader uncertainty. We did not establish equivalence, a causal benefit from Wait, or controlled hardware-only speed ratios.

## Ethical considerations and release status

The study uses public pretrained artifacts and teacher-generated data; it creates no new human-subject dataset. Mathematical outputs can be wrong despite plausible reasoning. We publish aggregate results and source code, while raw benchmark text, traces and weights stay in a local handoff. Inspected Qwen and s1K metadata declare Apache-2.0; inspected MATH-500 metadata does not declare a license. No project-wide license grant or benchmark redistribution right is invented. Check upstream terms before further redistribution. AI assistance was used for implementation and report preparation; the named authors remain responsible for verification and any eventual submission.

## References

[1] Muennighoff et al. 2025. s1: Simple test-time scaling. Original version, arXiv:2501.19393v1. https://arxiv.org/abs/2501.19393v1

[2] Wei et al. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. arXiv:2201.11903. https://arxiv.org/abs/2201.11903

[3] Hu et al. 2021. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. https://arxiv.org/abs/2106.09685

[4] Kwon et al. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023, arXiv:2309.06180. https://arxiv.org/abs/2309.06180

[5] Qwen et al. 2025. Qwen2.5 Technical Report. arXiv:2412.15115. https://arxiv.org/abs/2412.15115

[6] Hendrycks et al. 2021. Measuring Mathematical Problem Solving With the MATH Dataset. arXiv:2103.03874. https://arxiv.org/abs/2103.03874

[7] Hugging Face. Math-Verify, version 0.9.0. https://github.com/huggingface/Math-Verify

<!-- pagebreak -->

## Appendix A. Reproducibility and detailed costs

{medians}

Table A1. Medians and event counts over 500 questions each. A/B have no separate structured reasoning phase. Reasoning caps in D are intentional, while C/E reasoning cap events are their shared safety limit. Events can overlap; totals should not be summed across categories.

The source code is at https://github.com/faisalAI27/Qwen2.5-1.5B-finetune . The GPU launcher/backend is pinned to commit 1cae805a2a29b7ba769838380f2fd71f058fa824, with source asset code at e4096450228f194e6f39963dbe8bc4dd8be19be8. All finalizer/report changes are downstream documentation and diagnostics; the frozen experiment code remains unchanged.

Model revision: 989aa7980e4cf806f80c7fef2b1adb7bc71aa306. Original s1K revision: 278d72baaa2b887a7e76a70a0ae254a5a45536e4. MATH-500 revision: 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be. The exact training recipe, prompt templates, grading policy and both distinct evaluation freezes are retained in the local handoff. See docs/reproducibility.md for commands and docs/handoff.json for file checksums.

Evaluated JSONL SHA-256: {EVALUATED_SHA}

Reviewed result ZIP SHA-256: {ARCHIVE_SHA}

Source adapter tensor fingerprint: 70ef5f920cac164fa4a2c35c3b0c670efa812ea393ee9e2def92adac042d5144

The ZIP contains all 4,500 raw outputs, grades and supporting provenance, but omits reconstructible exported dense weights. The local handoff adds the original Q4 base and accepted MLX adapter so conversion can be repeated without substituting weights. CRC, safe paths, all prediction/grade hash links, source gold answers and frozen source inputs were verified. The complete analysis, including 10,000-sample paired intervals and all qualitative selections, was recomputed locally and matched the GPU export. Original analysis figures and full machine-readable tables accompany the report.

The 1,287 partial Mac outputs are preserved for provenance but are ungraded and excluded from all final scores. After archive verification the user-authorized Mac controller was stopped on 17 November 2026 at 18:09:58 Asia/Karachi. Its PAUSE marker remains in place. No additional training or full benchmark generation is needed for this report.

<!-- pagebreak -->

## Appendix B. Hardware and measured runtime

| Item | Local training host | Final evaluation host |
|---|---|---|
| Machine / device | MacBookAir10,1; Apple M1 | RunPod; NVIDIA GeForce RTX 4090 |
| CPU / accelerator | CPU 8 cores (4P + 4E); GPU 7 cores | One CUDA GPU; compute capability 8.9 |
| Memory | 8 GB unified CPU/GPU memory | 24,564 MiB reported GPU memory |
| OS / driver | macOS 26.6.2, build 25G83 | Linux; NVIDIA driver 595.91.07 |
| Python | 3.13.3, arm64 | 3.12.3 |
| Model runtime | MLX 0.32.4; MLX-LM 0.32.2 | vLLM 0.11.2; Torch 2.9.0; CUDA 12.8 |
| Numerical path | Affine Q4, group 64; FP32 activations/adapters | Dense FP16 reconstruction; FP16 LoRA |
| Context limit | 9,216 training tokens, complete examples | 16,384 tokens; no test prompt truncation |
| Scheduling | Micro-batch 1; accumulate 16 | Concurrency 8; max sequences 8 |

The accepted training run records 64,238.27 seconds (17.84 hours) in training windows, with a 66,207.96-second attempt wall time. It processes 5,066,380 input tokens and 4,926,536 supervised targets. Peak active MLX allocation is 5,750,514,820 bytes (5.36 GiB). This allocator measurement is not dedicated VRAM or total system memory; macOS and other apps share the 8 GB pool. An approximately nine-hour low-battery sleep interruption is calendar downtime, not active GPU compute. These figures do not include every failed pilot, development run or idle period.

The complete GPU test generates 9,577,261 tokens in 8,245.26 measured generation seconds (2.29 hours), or 1,161.55 aggregate generated tokens/second across concurrent requests. Wall time including engine setup is 8,266.25 seconds. This is a measured one-GPU test-generation window, not total rented GPU time or billing. No bill, energy measurement or total project cost was collected. Per-output latency includes shared scheduling and repeated segment prefills; summing it would overcount GPU wall time.

The GPU runtime uses a 0.85 memory utilization setting, prefix caching, chunked prefill and CUDA graphs. Dense base weights occupy 3,087,428,608 bytes before caches and runtime overhead. The Mac's affine quantization and the GPU's dense FP16 reconstruction differ numerically, and request scheduling differs, so this study does not report a controlled GPU-versus-Mac hardware speedup. The final 4,500 scores come exclusively from the one frozen GPU backend.

The initial local data volume was 228 GiB with approximately 12-16 GiB free during recorded early checks; free storage varied as work progressed. Storage is not accelerator memory. The project completed local training on the user's PC and used the separately authorized cloud GPU for the final full evaluation. All hardware figures above describe observed devices rather than the proposal's provisional 16-24 GB training target.
'''
    (docs/'paper.md').write_text(paper)
    return compact


def make_pdf(markdown_path, output):
    """Render the deliberately small Markdown subset used by the paper."""
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether
    from xml.sax.saxutils import escape
    import matplotlib
    fontdir=Path(matplotlib.get_data_path())/'fonts/ttf'
    for name,file in [('Study','DejaVuSans.ttf'),('StudyBold','DejaVuSans-Bold.ttf'),('StudyItalic','DejaVuSans-Oblique.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fontdir/file)))
    pdfmetrics.registerFontFamily('Study',normal='Study',bold='StudyBold',italic='StudyItalic',boldItalic='StudyBold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyStudy',fontName='Study',fontSize=9,leading=13,spaceAfter=7,textColor=colors.HexColor('#182431'),splitLongWords=True))
    styles.add(ParagraphStyle(name='TitleStudy',fontName='StudyBold',fontSize=22,leading=27,spaceAfter=14,textColor=colors.HexColor('#133e53')))
    styles.add(ParagraphStyle(name='HeadingStudy',fontName='StudyBold',fontSize=12,leading=16,spaceBefore=9,spaceAfter=7,keepWithNext=True,textColor=colors.HexColor('#133e53')))
    styles.add(ParagraphStyle(name='SubStudy',fontName='StudyBold',fontSize=10,leading=14,spaceBefore=7,spaceAfter=5,keepWithNext=True))
    styles.add(ParagraphStyle(name='CellStudy',fontName='Study',fontSize=7,leading=10,splitLongWords=True))
    def rich(s):
        s=escape(s)
        s=re.sub(r'`([^`]+)`',r'<font color="#155b70">\1</font>',s)
        s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
        return s
    output.parent.mkdir(parents=True,exist_ok=True)
    width=A4[0]-86
    doc=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=43,rightMargin=43,topMargin=42,bottomMargin=43,
                          title='Budget Forcing on a 1.5B Model: A Resource-Constrained Study',author='Faisal Imran; Muhammad Ibrahim; Hamza Saeed')
    story=[]; lines=markdown_path.read_text().splitlines();i=0
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line: continue
        if line=='<!-- pagebreak -->': story.append(PageBreak());continue
        if line.startswith('!['):
            path=re.search(r'\]\(([^)]+)\)',line).group(1)
            img=Image(str(markdown_path.parent/path));img.drawHeight=width*img.imageHeight/img.imageWidth;img.drawWidth=width
            story.extend([img,Spacer(1,5)]);continue
        if line.startswith('|'):
            raw=[line]
            while i<len(lines) and lines[i].strip().startswith('|'):
                raw.append(lines[i].strip());i+=1
            data=[[Paragraph(rich(v.strip()),styles['CellStudy']) for v in r.strip('|').split('|')] for r in raw if not re.match(r'^\|[\s:|\-]+\|$',r)]
            n=len(data[0]); widths=([.09,.16,.12,.1,.13,.13,.12,.15] if n==8 else None)
            if widths: widths=[width*x/sum(widths) for x in widths]
            else: widths=[width/n]*n
            table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e2edf1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#397083')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d3dbe0')),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([table,Spacer(1,8)]);continue
        if line.startswith('# '): style=styles['TitleStudy'];line=line[2:]
        elif line.startswith('## '): style=styles['HeadingStudy'];line=line[3:]
        elif line.startswith('### '): style=styles['SubStudy'];line=line[4:]
        else:
            style=styles['BodyStudy']
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','<!--')):
                line+=' '+lines[i].strip();i+=1
        story.append(Paragraph(rich(line),style))
    def page(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#c6d5dc'));canvas.line(43,31,A4[0]-43,31)
        canvas.setFont('Study',7);canvas.setFillColor(colors.HexColor('#62727d'))
        canvas.drawString(43,20,'s1 small-model study | Research project report | 2026-11-17')
        canvas.drawRightString(A4[0]-43,20,str(doc.page))
    doc.build(story,onFirstPage=page,onLaterPages=page)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=ROOT/'nlp-runpod-vllm-results (1).zip')
    parser.add_argument('--pdf',action='store_true')
    args=parser.parse_args()
    a,s,d,v=load_verified(args.archive)
    plots(a,ROOT/'docs/figures')
    result=write_documents(a,s,d,v)
    if args.pdf: make_pdf(ROOT/'docs/paper.md',ROOT/'output/pdf/s1-small-model-report.pdf')
    print(json.dumps({'status':'complete','outputs':4500,'grades':4500,'pdf_created':args.pdf,'d1024_token_reduction_percent':result['d1024_total_token_reduction_percent']}))


if __name__=='__main__': main()
