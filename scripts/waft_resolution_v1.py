"""解析度對照:同一個 WAFT checkpoint,原生 1080p 對上提交用的 2x2 box average。

R2 C4:提交欄位在送進網路前做 INTER_AREA 半解析度,對 i.i.d. 雜訊的變異砍四倍,
所以「雜訊家族的中性」同時混淆了架構與輸入解析度。這支腳本把架構、checkpoint、
精度、取樣全部固定,只動解析度,直接量那個混淆有多大。

只讀官方 test 影像(無 GT),不寫 submissions/。取樣沿用 cross_arch_ratio_v1 的規則。
"""
import argparse, json, os, shutil, time
from pathlib import Path

import numpy as np

from submission_inference import TestImages, CONDITIONS, ROOT
from scaled_inference import ScaledPredictor
from cross_arch_ratio_v1 import selection, N_PAIRS
from run_e0 import sha256

MODEL = 'waft_dav2_a2'
CONDS = ['clean', 'gaussian_noise', 'shot_noise', 'speckle_noise', 'impulse_noise', 'rain']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--precision', default='fp16')
    p.add_argument('--out', type=Path, default=ROOT / 'runs/waft_resolution_v1')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)

    for c in CONDS:
        assert c in CONDITIONS, c
    probe = TestImages('clean')
    keys = selection(probe.counts)
    probe.close()

    for scale in (1.0, 0.5):
        tag = 'native' if scale == 1.0 else 'half'
        out = a.out / tag
        out.mkdir()
        cache = out / 'clean_cache'
        cache.mkdir()
        model = ScaledPredictor(MODEL, a.precision, scale=scale)
        (out / 'provenance.json').write_text(json.dumps(dict(
            **model.provenance,
            selection=f'每個 (scene,camera) 等距 {N_PAIRS} 個相鄰對,FW 與 BW;推論前固定',
            n_keys=len(keys), conditions=CONDS,
            metric='clean 對 corrupted 的預測分歧、幅度比與 along-clean gain;不使用 test GT',
            sweep_adapter_sha256=sha256(Path(__file__)),
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')), indent=2, default=str,
            ensure_ascii=False))
        print(f'--- {tag} (scale {scale}): {len(keys)} keys x {len(CONDS)} conditions', flush=True)

        start, n, fp32 = time.monotonic(), 0, None
        with (out / 'metrics.jsonl').open('w') as log:
            for condition in CONDS:
                images = TestImages(condition)
                for seq, cam, source, direction in keys:
                    key = f'{seq}_{cam}_{direction}_{source:04d}'
                    inp = images.triplet(seq, cam, source, direction)
                    t = time.monotonic()
                    fell_back = False
                    try:
                        pred = model(inp).astype(np.float64)
                    except ValueError:
                        # fp16 偶爾輸出非有限值;那一筆改用 fp32 重算並記錄,不跳過也不補零。
                        if fp32 is None:
                            fp32 = ScaledPredictor(MODEL, 'fp32', scale=scale)
                        pred = fp32(inp).astype(np.float64)
                        fell_back = True
                    elapsed = time.monotonic() - t
                    mag = np.linalg.norm(pred, axis=-1)
                    if condition == 'clean':
                        np.save(cache / (key + '.npy'), pred.astype(np.float32))
                        row_extra = dict(delta=0.0, clean_mag_sum=float(mag.sum()),
                                         dot_sum=float((pred * pred).sum()),
                                         clean_sq_sum=float((pred * pred).sum()),
                                         corrupt_sq_sum=float((pred * pred).sum()),
                                         moving=0, collapsed=0, inflated=0)
                    else:
                        ref = np.load(cache / (key + '.npy')).astype(np.float64)
                        cm = np.linalg.norm(ref, axis=-1)
                        mv = cm > 1.0
                        row_extra = dict(
                            delta=float(np.linalg.norm(pred - ref, axis=-1).mean()),
                            clean_mag_sum=float(cm.sum()),
                            dot_sum=float((pred * ref).sum()),
                            clean_sq_sum=float((ref * ref).sum()),
                            corrupt_sq_sum=float((pred * pred).sum()),
                            moving=int(mv.sum()),
                            collapsed=int((mv & (mag < 0.5 * cm)).sum()),
                            inflated=int((mv & (mag > 2.0 * cm)).sum()))
                    log.write(json.dumps(dict(
                        key=key, sequence=seq, camera=cam, direction=direction, source=source,
                        condition=condition, scale=scale, n_pixels=int(mag.size),
                        mag_sum=float(mag.sum()), seconds=elapsed, fp32_fallback=fell_back,
                        **row_extra)) + '\n')
                    log.flush(); n += 1
                    if n % 60 == 0:
                        print(f'{tag} {n}/{len(keys)*len(CONDS)} {condition} {elapsed:.2f}s', flush=True)
                images.close()
        assert n == len(keys) * len(CONDS), (n, len(keys) * len(CONDS))
        (out / 'DONE.json').write_text(json.dumps(dict(done=True, records=n, scale=scale,
                                                       wall_seconds=time.monotonic() - start)))
        shutil.rmtree(cache)
        del model
        print(f'DONE {tag} {n} {time.monotonic() - start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
