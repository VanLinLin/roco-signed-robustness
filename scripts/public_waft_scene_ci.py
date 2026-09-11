"""公開 WAFT 提交檔的 per-scene 統計 + scene-block bootstrap CI。

只讀 submissions/v1/flow_robustness.hdf5(唯讀)與官方取樣索引,
不寫入 submissions/v1,不做推論。取樣點與官方伺服器讀的完全相同。
"""
import json
from collections import defaultdict

import h5py
import numpy as np

ROOT = '/ssd8/van/contest/RoCo'
F = f'{ROOT}/submissions/v1/flow_robustness.hdf5'
IDX = f'{ROOT}/tools/subsampling/robust_sampling_map_v2/robust.json'
OUT = f'{ROOT}/runs/public_waft_ci_v1'
NBOOT, SEED = 10000, 20260911

idx = json.load(open(IDX))
entries = idx['entries']
assert entries[-1]['stop'] == idx['shape'][0], (entries[-1]['stop'], idx['shape'][0])

# 每筆記錄屬於哪個 scene(path 形如 0003/flow_FW_left/flow_FW_left_0001)
scenes = sorted({e['path'].split('/')[0] for e in entries})
sidx = {s: i for i, s in enumerate(scenes)}
rec_scene = np.array([sidx[e['path'].split('/')[0]] for e in entries])
starts = np.array([e['start'] for e in entries])
stops = np.array([e['stop'] for e in entries])
print(f'{len(entries)} records, {len(scenes)} scenes: {scenes}', flush=True)


def per_scene_sums(mag, weights=None):
    """把每筆記錄的量加總到它的 scene。"""
    acc = np.zeros(len(scenes), np.float64)
    cnt = np.zeros(len(scenes), np.float64)
    cs = np.concatenate([[0.0], np.cumsum(mag, dtype=np.float64)])
    tot = cs[stops] - cs[starts]
    n = (stops - starts).astype(np.float64)
    np.add.at(acc, rec_scene, tot)
    np.add.at(cnt, rec_scene, n)
    return acc, cnt


rng = np.random.default_rng(SEED)
draws = rng.integers(0, len(scenes), size=(NBOOT, len(scenes)))

with h5py.File(F, 'r') as f:
    clean = np.nan_to_num(f['clean/flow'][:]).astype(np.float64)
    cmag = np.linalg.norm(clean, axis=-1)
    c_acc, c_cnt = per_scene_sums(cmag)
    out = {}
    for k in sorted(f.keys()):
        if k == 'clean':
            continue
        cur = np.nan_to_num(f[f'{k}/flow'][:]).astype(np.float64)
        qmag = np.linalg.norm(cur, axis=-1)
        q_acc, _ = per_scene_sums(qmag)
        # bootstrap:整塊 scene 重抽,用像素和重新算比值(不是比值的平均)
        bq = q_acc[draws].sum(1)
        bc = c_acc[draws].sum(1)
        r = bq / bc
        lo, hi = np.percentile(r, [2.5, 97.5])
        out[k] = dict(
            ratio=float(q_acc.sum() / c_acc.sum()),
            ci_lo=float(lo), ci_hi=float(hi),
            per_scene={s: float(q_acc[i] / c_acc[i]) for i, s in enumerate(scenes)},
        )
        print(f'{k:20s} ratio {out[k]["ratio"]:.4f}  CI [{lo:.3f}, {hi:.3f}]', flush=True)
        del cur, qmag

json.dump(dict(seed=SEED, nboot=NBOOT, scenes=scenes,
               source_hdf5=F, sampling_index=IDX, by_corruption=out),
          open(f'{OUT}/SCENE_CI.json', 'w'), ensure_ascii=False, indent=1)
print('wrote', f'{OUT}/SCENE_CI.json')
