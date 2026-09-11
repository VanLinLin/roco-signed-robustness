"""把 waft_resolution_v1 的兩個解析度整理成同一張表。

scene-block bootstrap 與論文其他表一致:10 個 scene 整塊重抽 10,000 次,seed 20260911。
"""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path('/ssd8/van/contest/RoCo')
RUN = ROOT / 'runs/waft_resolution_v1'
SEED, B = 20260911, 10000


def load(tag):
    rows = [json.loads(l) for l in (RUN / tag / 'metrics.jsonl').open()]
    return [r for r in rows if r['condition'] != 'clean']


def block_ci(by_scene, fn):
    scenes = sorted(by_scene)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(scenes), size=(B, len(scenes)))
    vals = []
    for d in draws:
        picked = [by_scene[scenes[i]] for i in d]
        vals.append(fn({k: sum(p[k] for p in picked) for k in picked[0]}))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


KEYS = ('mag_sum', 'clean_mag_sum', 'dot_sum', 'clean_sq_sum', 'corrupt_sq_sum')
ratio = lambda s: s['mag_sum'] / s['clean_mag_sum']
gain = lambda s: s['dot_sum'] / s['clean_sq_sum']


def summarize(tag):
    out = {}
    for cond in dict.fromkeys(r['condition'] for r in load(tag)):
        rows = [r for r in load(tag) if r['condition'] == cond]
        by_scene = {}
        for r in rows:
            acc = by_scene.setdefault(r['sequence'], {k: 0.0 for k in KEYS})
            for k in KEYS:
                acc[k] += r[k]
        tot = {k: sum(v[k] for v in by_scene.values()) for k in KEYS}
        g = gain(tot)
        orth = (tot['corrupt_sq_sum'] - 2 * g * tot['dot_sum']
                + g * g * tot['clean_sq_sum']) / tot['corrupt_sq_sum']
        per_scene = {sq: ratio(v) for sq, v in sorted(by_scene.items())}
        out[cond] = dict(
            ratio=ratio(tot), ratio_ci95=block_ci(by_scene, ratio),
            directional_gain=g, gain_ci95=block_ci(by_scene, gain),
            orthogonal_fraction=orth,
            delta=float(np.mean([r['delta'] for r in rows])),
            per_scene_ratio=per_scene,
            n_scenes_below_0p95=sum(1 for v in per_scene.values() if v < 0.95),
            fp32_fallbacks=sum(r['fp32_fallback'] for r in rows),
            n_records=len(rows), n_scenes=len(by_scene))
    return out


res = {tag: summarize(tag) for tag in ('native', 'half')}
res['provenance'] = dict(
    note='同一個 waft_dav2_a2 checkpoint、同一組 120 個 key、同一個精度;只有輸入解析度不同',
    native=json.loads((RUN / 'native/provenance.json').read_text())['resize'],
    half=json.loads((RUN / 'half/provenance.json').read_text())['resize'],
    bootstrap=f'10 scenes as blocks, {B} draws, seed {SEED}')
(RUN / 'RESOLUTION_TABLE.json').write_text(json.dumps(res, indent=2, ensure_ascii=False))

hdr = f"{'condition':<16}{'native ratio':>26}{'half ratio':>26}{'nat g':>8}{'half g':>8}"
print(hdr); print('-' * len(hdr))
for cond in res['native']:
    n, h = res['native'][cond], res['half'][cond]
    print(f"{cond:<16}{n['ratio']:>10.4f} [{n['ratio_ci95'][0]:.3f}, {n['ratio_ci95'][1]:.3f}]"
          f"{h['ratio']:>10.4f} [{h['ratio_ci95'][0]:.3f}, {h['ratio_ci95'][1]:.3f}]"
          f"{n['directional_gain']:>8.3f}{h['directional_gain']:>8.3f}")
