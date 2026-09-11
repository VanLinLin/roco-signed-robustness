"""CPU-only corruption axes on official RobustSpring V1 PNGs.

Reads ZIP members into RAM and writes only this run directory.  It performs no
model inference and never opens submission artifacts for writing.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import zipfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCHIVES = Path('/ssd8/van/dataset/RoCo/archives')
CONDITIONS = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost',
              'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
              'jpeg_compression', 'motion_blur', 'pixelate', 'rain', 'saturate',
              'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']
THRESHOLD = 5
H, W = 1080, 1920


def write_json(path, value):
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def pearson(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) < 2:
        return None
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denom) if denom > 0 else None


def temporal_metrics(a, b):
    ma, mb = a > THRESHOLD, b > THRESHOLD
    inter, union = int(np.count_nonzero(ma & mb)), int(np.count_nonzero(ma | mb))
    return dict(iou=inter / union if union else None,
                residual_amplitude_pearson=pearson(a, b),
                coverage_a=float(ma.mean()), coverage_b=float(mb.mean()))


def structure_metric(amplitude):
    """Nonconstant, locally coherent residual energy at 1--8 native pixels.

    Let A=mean_RGB(abs(C-I)).  At each integer horizontal/vertical offset in
    1..8, calculate Pearson(A(x),A(x+delta)) on a fixed 1/8 spatial sample.
    S=std(A)*max(0,max rho).  It is zero for a spatially constant residual or
    for a residual with no positive local autocorrelation.  It is deliberately
    a proxy, not a proof that a neural matcher uses the residual.
    """
    sampled = amplitude[::8, ::8]
    std = float(sampled.std(dtype=np.float64))
    correlations = []
    # Original-pixel offsets, evaluated at a fixed grid to bound CPU cost.
    for offset in range(1, 9):
        correlations.extend([pearson(amplitude[::8, :-offset:8], amplitude[::8, offset::8]),
                             pearson(amplitude[:-offset:8, ::8], amplitude[offset::8, ::8])])
    finite = [x for x in correlations if x is not None and np.isfinite(x)]
    peak = max(finite) if finite else None
    return dict(amplitude_std=std, autocorr_peak_1to8=peak,
                coherent_residual_score=std * max(0., peak) if peak is not None else 0.,
                autocorrelation_sample_stride=8)


def archive_path(condition, camera):
    if condition == 'clean':
        return ARCHIVES / f'spring/test_frame_{camera}.zip'
    return ARCHIVES / f'robust_spring/{condition}.zip'


def process_scene(item):
    cv2.setNumThreads(1)
    scene, length = item
    target = OUT / 'scenes' / f'{scene}.json'
    if target.exists():
        prior = json.loads(target.read_text())
        if prior.get('done'):
            return dict(scene=scene, resumed=True)
    anchors = np.rint(np.linspace(1, length - 1, 10)).astype(int).tolist()
    frames = sorted(set(anchors + [i + 1 for i in anchors]))
    if len(frames) != 20:
        raise ValueError('Expected 20 distinct timepoints')
    start = time.time()
    rows, inputs = [], []
    for camera in ['left', 'right']:
        archives = {c: zipfile.ZipFile(archive_path(c, camera)) for c in ['clean'] + CONDITIONS}
        try:
            clean = {}
            for t in frames:
                member = f'spring/test/{scene}/frame_{camera}/frame_{camera}_{t:04d}.png'
                raw = archives['clean'].read(member)
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if image is None or image.shape != (H, W, 3):
                    raise ValueError(member)
                clean[t] = image
                inputs.append(dict(condition='clean', camera=camera, frame=t, member=member,
                                   sha256=hashlib.sha256(raw).hexdigest()))
            for condition in CONDITIONS:
                amplitude = {}
                for t in frames:
                    member = f'{condition}/test/{scene}/frame_{camera}/frame_{camera}_{t:04d}.png'
                    raw = archives[condition].read(member)
                    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if image is None or image.shape != (H, W, 3):
                        raise ValueError(member)
                    residual = image.astype(np.int16) - clean[t].astype(np.int16)
                    amplitude[t] = np.abs(residual).mean(axis=-1, dtype=np.float32)
                    inputs.append(dict(condition=condition, camera=camera, frame=t, member=member,
                                       sha256=hashlib.sha256(raw).hexdigest()))
                    rows.append(dict(scene=scene, camera=camera, condition=condition, dimension='structure',
                                     frame_a=t, frame_b=None, **structure_metric(amplitude[t])))
                for t in anchors:
                    rows.append(dict(scene=scene, camera=camera, condition=condition, dimension='time',
                                     frame_a=t, frame_b=t + 1, **temporal_metrics(amplitude[t], amplitude[t + 1])))
        finally:
            for archive in archives.values():
                archive.close()
    result = dict(done=True, scene=scene, length=length, anchors=anchors, frames=frames,
                  rows=rows, inputs=inputs, seconds=time.time() - start)
    write_json(target, result)
    write_json(OUT / 'progress' / f'{scene}.json',
               dict(scene=scene, done=True, rows=len(rows), seconds=result['seconds'], updated_at=time.time()))
    return dict(scene=scene, rows=len(rows), seconds=result['seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    for folder in ['scenes', 'progress']:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path('clean', 'left')) as archive:
        lengths = {}
        for name in archive.namelist():
            if name.endswith('.png'):
                scene = name.split('/')[2]
                lengths[scene] = lengths.get(scene, 0) + 1
    config = dict(scenes=lengths, conditions=CONDITIONS, anchors_per_scene=10,
                  timestamps_per_scene=20,
                  sampling='round(linspace(1,N-1,10)); each t is compared to t+1 in each eye',
                  temporal_amplitude='A=mean_RGB(abs(C-I)), uint8 clean/corrupt PNG promoted to integer first',
                  temporal_iou=f'M=(A>{THRESHOLD}); IoU(M_t,M_t+1)',
                  temporal_correlation='Pearson(A_t,A_t+1) over full native 1920x1080 image',
                  structure_metric='S=std(A[::8,::8])*max(0,max Pearson(A(x),A(x+d))) over d=1..8 pixels, horizontal and vertical; zero when A is spatially constant or has no positive local autocorrelation',
                  structure_sampling='fixed 1/8 grid only for autocorrelation; offsets remain native image pixels',
                  aggregation='equal scene mean; scene-block bootstrap 10000 draws, seed 20260911',
                  cpu_only=True, model_inference=False, extraction='individual ZIP PNG members decoded to RAM',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    existing = OUT / 'CONFIG.json'
    if existing.exists() and json.loads(existing.read_text()) != config:
        raise ValueError('Existing configuration differs')
    write_json(existing, config)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_scene, value) for value in sorted(lengths.items())]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    write_json(OUT / 'DONE.json', dict(done=True, scenes=len(lengths), finished_at=time.time()))


if __name__ == '__main__':
    main()
