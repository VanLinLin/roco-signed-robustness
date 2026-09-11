"""跨架構複製:同一條規則在不同架構的光流模型上是否成立。

只讀官方 test 影像(無 GT),量 clean 與 corrupted 預測的
(a) Δ = 平均端點分歧, (b) dense 平均運動幅度比。與提交無關,不寫 submissions/。

取樣在推論前固定:每個 (scene, camera) 取 N_PAIRS 個等距的相鄰影格對,FW 與 BW 各一。
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import torch

from submission_inference import FixedPredictor, TestImages, CONDITIONS, DATA, ROOT
from run_e0 import sha256

N_PAIRS = 3  # 每個 (scene, camera) 的相鄰影格對數;推論前固定


def selection(counts):
    """固定、與模型無關的取樣:每個 (scene,cam) 在序列內等距取 N_PAIRS 個 source。"""
    keys = []
    for (seq, cam), count in sorted(counts.items()):
        # source 需滿足 1 <= source, source+1 <= count(triplet 會自行 clamp 邊界)
        lo, hi = 2, count - 2
        picks = sorted({int(round(lo + (hi - lo) * i / max(1, N_PAIRS - 1))) for i in range(N_PAIRS)})
        for source in picks:
            for direction in ('FW', 'BW'):
                keys.append((seq, cam, source if direction == 'FW' else source + 1, direction))
    return keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--precision', default='fp16')
    p.add_argument('--tag', default='')
    p.add_argument('--conditions', default='',
                   help='逗號分隔;只跑這些條件(clean 一律自動加入)')
    a = p.parse_args()

    out = ROOT / 'runs/cross_arch_v1' / f'{a.model}_{a.precision}{a.tag}'
    out.mkdir(parents=True, exist_ok=False)
    cache = out / 'clean_cache'
    cache.mkdir()

    conds = CONDITIONS if not a.conditions else \
        ['clean'] + [c for c in a.conditions.split(',') if c and c != 'clean']
    for c in conds:
        assert c in CONDITIONS, c
    model = FixedPredictor(a.model, a.precision)
    probe = TestImages('clean')
    keys = selection(probe.counts)
    probe.close()
    (out / 'provenance.json').write_text(json.dumps(dict(
        **model.provenance,
        selection=f'每個 (scene,camera) 等距 {N_PAIRS} 個相鄰對,FW 與 BW;推論前固定',
        n_keys=len(keys), n_conditions=len(conds), conditions=conds,
        metric='clean 對 corrupted 的預測分歧與 dense 平均運動幅度比;不使用 test GT',
        sweep_adapter_sha256=sha256(Path(__file__)),
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')), indent=2, default=str))
    print(f'{a.model}: {len(keys)} keys x {len(conds)} conditions = {len(keys)*len(conds)}', flush=True)

    start, n = time.monotonic(), 0
    with (out / 'metrics.jsonl').open('w') as log:
        for condition in conds:
            images = TestImages(condition)
            for seq, cam, source, direction in keys:
                key = f'{seq}_{cam}_{direction}_{source:04d}'
                inp = images.triplet(seq, cam, source, direction)
                t = time.monotonic()
                fell_back = False
                try:
                    pred = model(inp).astype(np.float64)
                except ValueError:
                    # fp16 偶爾輸出非有限值;改用 fp32 重算這一筆並記錄下來,
                    # 不跳過、不補零,讓報告能誠實說明有幾筆是這樣算的。
                    pred = model.inner(inp).astype(np.float64)
                    fell_back = True
                elapsed = time.monotonic() - t
                mag = np.linalg.norm(pred, axis=-1)
                if condition == 'clean':
                    np.save(cache / (key + '.npy'), pred.astype(np.float32))
                    delta, cmag_sum, moving, collapsed = 0.0, float(mag.sum()), 0, 0
                else:
                    ref = np.load(cache / (key + '.npy')).astype(np.float64)
                    delta = float(np.linalg.norm(pred - ref, axis=-1).mean())
                    cm = np.linalg.norm(ref, axis=-1)
                    cmag_sum = float(cm.sum())
                    mv = cm > 1.0
                    moving = int(mv.sum())
                    collapsed = int((mv & (mag < 0.5 * cm)).sum())
                row = dict(key=key, sequence=seq, camera=cam, direction=direction, source=source,
                           condition=condition, delta=delta,
                           n_pixels=int(mag.size), mag_sum=float(mag.sum()), clean_mag_sum=cmag_sum,
                           moving=moving, collapsed=collapsed, seconds=elapsed,
                           fp32_fallback=fell_back)
                log.write(json.dumps(row) + '\n'); log.flush(); n += 1
                if n % 50 == 0:
                    print(f'{n} {condition} {key} delta {delta:.3f} {elapsed:.2f}s', flush=True)
            images.close()
    assert n == len(keys) * len(conds), (n, len(keys) * len(conds))
    (out / 'DONE.json').write_text(json.dumps(dict(done=True, records=n,
                                                   wall_seconds=time.monotonic() - start)))
    print('DONE', n, time.monotonic() - start, flush=True)


if __name__ == '__main__':
    main()
