"""彙總 cross_arch_v1:每個模型的 20 條件幅度比 + scene-block bootstrap CI。"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/ssd8/van/contest/RoCo')
RUN = ROOT / 'runs/cross_arch_v1'
NBOOT, SEED = 10000, 20260911
# 公開 WAFT(已提交、官方計分)的參考值,供對照
REF = json.load(open(ROOT / 'submissions/v1/LOCAL_RECOMPUTE.json'))['by_corruption']


def summarize(model_dir):
    rows = [json.loads(l) for l in (model_dir / 'metrics.jsonl').read_text().splitlines()]
    scenes = sorted({r['sequence'] for r in rows})
    sidx = {s: i for i, s in enumerate(scenes)}
    by = defaultdict(lambda: dict(q=np.zeros(len(scenes)), c=np.zeros(len(scenes)),
                                  d=[], mv=0, cl=0))
    for r in rows:
        if r['condition'] == 'clean':
            continue
        a = by[r['condition']]
        a['q'][sidx[r['sequence']]] += r['mag_sum']
        a['c'][sidx[r['sequence']]] += r['clean_mag_sum']
        a['d'].append(r['delta'])
        a['mv'] += r['moving']; a['cl'] += r['collapsed']
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(scenes), size=(NBOOT, len(scenes)))
    out = {}
    for k, a in by.items():
        r = a['q'][draws].sum(1) / a['c'][draws].sum(1)
        lo, hi = np.percentile(r, [2.5, 97.5])
        out[k] = dict(ratio=float(a['q'].sum() / a['c'].sum()),
                      ci_lo=float(lo), ci_hi=float(hi),
                      delta=float(np.mean(a['d'])),
                      collapse=float(a['cl'] / a['mv']),
                      n_records=len(a['d']))
    return out, scenes


def main():
    models = sorted(d for d in RUN.iterdir() if d.is_dir() and (d / 'DONE.json').exists())
    if not models:
        sys.exit('no completed model runs')
    all_out = {}
    for m in models:
        out, scenes = summarize(m)
        all_out[m.name] = out
        print(f'\n### {m.name}  ({len(scenes)} scenes, {next(iter(out.values()))["n_records"]} rec/cond)')
        print(f'{"condition":20s} {"ratio":>7s} {"CI95":>18s} {"delta":>8s}   {"pubWAFT":>7s}')
        for k, v in sorted(out.items(), key=lambda kv: kv[1]['ratio']):
            mark = '  <<<' if (v['ratio'] < 0.95 or v['ratio'] > 1.05) else ''
            print(f'{k:20s} {v["ratio"]:7.4f} [{v["ci_lo"]:7.4f},{v["ci_hi"]:7.4f}] '
                  f'{v["delta"]:8.3f}   {REF[k]["ratio"]:7.4f}{mark}')
        neutral = [v['ratio'] for k, v in out.items() if 0.95 < v['ratio'] < 1.05]
        print(f'  中性帶({len(neutral)} 條件): [{min(neutral):.4f}, {max(neutral):.4f}]')
    json.dump(dict(seed=SEED, nboot=NBOOT, by_model=all_out),
              open(RUN / 'CROSS_ARCH_SUMMARY.json', 'w'), ensure_ascii=False, indent=1)
    print('\nwrote', RUN / 'CROSS_ARCH_SUMMARY.json')


if __name__ == '__main__':
    main()
