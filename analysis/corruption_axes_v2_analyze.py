"""CPU-only all-condition template and scene-motion alignment measurement.

Reads official PNG ZIP members and already-exported clean public-WAFT flows.
It does not invoke a model and only writes beneath corruption_axes_v2.
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
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCHIVES = Path('/ssd8/van/dataset/RoCo/archives')
PREDICTIONS = ROOT / 'runs/submission_v1/predictions/clean'
CONDITIONS = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost',
              'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
              'jpeg_compression', 'motion_blur', 'pixelate', 'rain', 'saturate',
              'shot_noise', 'snow', 'spatter', 'speckle_noise', 'zoom_blur']
SCENES = {'0003': 131, '0019': 111, '0028': 39, '0029': 135, '0031': 73,
          '0034': 47, '0035': 120, '0040': 111, '0042': 116, '0046': 117}
H, W = 1080, 1920
FB_ABS = 1.0
FB_REL = 0.05
GRID_Y, GRID_X = np.indices((H, W), dtype=np.float32)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def archive_path(condition, eye):
    return ARCHIVES / ('spring/test_frame_' + eye + '.zip' if condition == 'clean'
                       else 'robust_spring/' + condition + '.zip')


def member(condition, scene, eye, frame):
    prefix = 'spring' if condition == 'clean' else condition
    return f'{prefix}/test/{scene}/frame_{eye}/frame_{eye}_{frame:04d}.png'


def read_png(archive, condition, scene, eye, frame, manifest):
    name = member(condition, scene, eye, frame)
    raw = archive.read(name)
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (H, W, 3):
        raise ValueError('invalid PNG ' + name)
    manifest.append({'condition': condition, 'scene': scene, 'eye': eye, 'frame': frame,
                     'member': name, 'sha256': hashlib.sha256(raw).hexdigest()})
    return image


def flow_path(scene, eye, direction, frame):
    return PREDICTIONS / f'test/{scene}/flow_{direction}_{eye}/flow_{direction}_{eye}_{frame:04d}.flo5'


def read_flow(scene, eye, direction, frame, manifest):
    path = flow_path(scene, eye, direction, frame)
    with h5py.File(path, 'r') as handle:
        flow = handle['flow'][()].astype(np.float32)
    if flow.shape != (H, W, 2) or not np.isfinite(flow).all():
        raise ValueError('invalid flow ' + str(path))
    manifest.append({'flow': str(path.relative_to(ROOT)),
                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return flow


def pearson(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if len(a) < 2:
        return None
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denom) if denom else None


def forward_mapping(fw, bw):
    """Make a reusable, conservative forward-splat correspondence from clean flow."""
    qx = GRID_X + fw[..., 0]
    qy = GRID_Y + fw[..., 1]
    # Linear backward-flow sampling needs the right/bottom neighbour as well.
    inside = (qx >= 0) & (qx < W - 1) & (qy >= 0) & (qy < H - 1)
    sampled_bw = cv2.remap(bw, qx, qy, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    fb_error = np.linalg.norm(fw + sampled_bw, axis=-1)
    flow_norm = np.linalg.norm(fw, axis=-1)
    fb_ok = inside & np.isfinite(sampled_bw).all(axis=-1) & (fb_error <= np.maximum(FB_ABS, FB_REL * flow_norm))
    xi = np.rint(qx).astype(np.int32)
    yi = np.rint(qy).astype(np.int32)
    good = fb_ok.ravel()
    destination = yi.ravel()[good] * W + xi.ravel()[good]
    count = np.bincount(destination, minlength=H * W)
    unique = count == 1
    source_index = np.flatnonzero(good)
    source_for_destination = np.full(H * W, -1, np.int32)
    source_for_destination[destination] = source_index
    target_index = np.flatnonzero(unique)
    return dict(source_index=source_for_destination[target_index], target_index=target_index,
                source_out_of_bounds_fraction=float((~inside).mean()),
                source_fb_inconsistent_fraction=float((inside & ~fb_ok).mean()),
                target_collision_or_hole_fraction=float((~unique).mean()),
                retained_target_fraction=float(unique.mean()),
                retained_pixel_count=int(unique.sum()))


def forward_warp_correlation(source, target, mapping):
    return {key: value for key, value in mapping.items() if key not in ['source_index', 'target_index']} | {
        'correlation': pearson(source.ravel()[mapping['source_index']], target.ravel()[mapping['target_index']])}


def anchors_for(length):
    anchors = np.rint(np.linspace(1, length - 1, 10)).astype(int).tolist()
    frames = sorted(set(anchors + [t + 1 for t in anchors]))
    if len(frames) != 20:
        raise ValueError('sampling design requires 20 distinct frames')
    return anchors, frames


def fit_templates():
    """The identical fixed-template estimator used by the old lens verification."""
    result = {}
    manifest = []
    anchors, frames = anchors_for(SCENES['0003'])
    clean_zip = zipfile.ZipFile(archive_path('clean', 'left'))
    corrupt_zips = {c: zipfile.ZipFile(archive_path(c, 'left')) for c in CONDITIONS}
    try:
        for condition in CONDITIONS:
            template = np.zeros((H, W, 3), np.int16)
            for frame in frames:
                clean = read_png(clean_zip, 'clean', '0003', 'left', frame, manifest).astype(np.int16)
                corrupt = read_png(corrupt_zips[condition], condition, '0003', 'left', frame, manifest).astype(np.int16)
                template = np.maximum(template, np.maximum(corrupt - clean, 0))
            np.save(OUT / 'templates' / (condition + '.npy'), template)
            result[condition] = {'template_maximum': int(template.max()),
                                 'template_nonzero_channel_fraction': float((template != 0).mean())}
    finally:
        clean_zip.close()
        for archive in corrupt_zips.values(): archive.close()
    write_json(OUT / 'TEMPLATE_FIT.json', {'done': True, 'source_scene': '0003', 'source_eye': 'left',
               'source_frames': frames, 'estimator': 'T=max_over_source_images(max(C-I,0)), per native pixel/channel',
               'templates': result, 'inputs': manifest})


def process_scene(item):
    cv2.setNumThreads(1)
    scene, length = item
    target = OUT / 'scenes' / (scene + '.json')
    if target.exists():
        return {'scene': scene, 'resumed': True}
    started = time.time()
    anchors, frames = anchors_for(length)
    index = {frame: i for i, frame in enumerate(frames)}
    manifest = []
    archives = {('clean', eye): zipfile.ZipFile(archive_path('clean', eye)) for eye in ['left', 'right']}
    archives.update({(condition, eye): zipfile.ZipFile(archive_path(condition, eye))
                     for condition in CONDITIONS for eye in ['left', 'right']})
    templates = {condition: np.load(OUT / 'templates' / (condition + '.npy')) for condition in CONDITIONS}
    rows = []
    try:
        clean = {eye: {frame: read_png(archives[('clean', eye)], 'clean', scene, eye, frame, manifest)
                       for frame in frames} for eye in ['left', 'right']}
        mappings = {}
        for eye in ['left', 'right']:
            for frame in anchors:
                fw = read_flow(scene, eye, 'FW', frame, manifest)
                bw = read_flow(scene, eye, 'BW', frame + 1, manifest)
                mappings[(eye, frame)] = forward_mapping(fw, bw)
                del fw, bw
        for condition in CONDITIONS:
            template = templates[condition]
            corrupted = {eye: {frame: read_png(archives[(condition, eye)], condition, scene, eye, frame, manifest)
                                for frame in frames} for eye in ['left', 'right']}
            # Test 1 on all 400 same-design images: RGB-pixel exact fraction, channel MAE and maximum.
            error_sum = bad_pixels = bad_channels = channels = 0
            maximum = 0
            for eye in ['left', 'right']:
                for frame in frames:
                    prediction = np.minimum(clean[eye][frame].astype(np.int16) + template, 255)
                    error = np.abs(prediction - corrupted[eye][frame].astype(np.int16))
                    maximum = max(maximum, int(error.max()))
                    error_sum += int(error.sum(dtype=np.int64)); channels += error.size
                    bad_channels += int(np.count_nonzero(error))
                    bad_pixels += int(np.count_nonzero(np.any(error != 0, axis=-1)))
            rows.append({'dimension': 'template', 'condition': condition, 'max_abs_error': maximum,
                         'mean_abs_error_per_channel': error_sum / channels,
                         'exact_zero_rgb_pixel_fraction': 1 - bad_pixels / (channels // 3),
                         'exact_zero_channel_fraction': 1 - bad_channels / channels,
                         'images': 40})
            # Test 2: original zero-offset Pearson and clean-flow forward warp Pearson.
            for eye in ['left', 'right']:
                for frame in anchors:
                    a = np.abs(corrupted[eye][frame].astype(np.int16) - clean[eye][frame].astype(np.int16)).mean(axis=-1, dtype=np.float32)
                    b = np.abs(corrupted[eye][frame + 1].astype(np.int16) - clean[eye][frame + 1].astype(np.int16)).mean(axis=-1, dtype=np.float32)
                    warp = forward_warp_correlation(a, b, mappings[(eye, frame)])
                    rows.append({'dimension': 'temporal', 'condition': condition, 'scene': scene, 'eye': eye,
                                 'frame': frame, 'zero_offset_correlation': pearson(a, b),
                                 'flow_warp_correlation': warp['correlation'],
                                 'delta_warp_minus_zero': warp['correlation'] - pearson(a, b), **warp})
            del corrupted
    finally:
        for archive in archives.values(): archive.close()
    write_json(target, {'done': True, 'scene': scene, 'length': length, 'anchors': anchors, 'frames': frames,
                        'rows': rows, 'inputs': manifest, 'seconds': time.time() - started})
    return {'scene': scene, 'rows': len(rows), 'seconds': time.time() - started}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--workers', type=int, default=4); args = parser.parse_args()
    for directory in ['templates', 'scenes']:
        (OUT / directory).mkdir(parents=True, exist_ok=True)
    config = {'conditions': CONDITIONS, 'scenes': SCENES, 'anchors_per_scene': 10, 'time_pairs_per_condition': 200,
              'template_validation_images_per_condition': 400, 'template_fit': '0003 left, 20 v1 sampled frames, T=max(max(C-I,0))',
              'warp': 'nearest-target forward splat using clean public-WAFT flow; reject out-of-bounds, F/B-inconsistent source pixels, and many-to-one target collisions',
              'fb_consistency': '||F(x)+B(x+F(x))|| <= max(1.0,0.05*||F(x)||)',
              'aggregation': 'equal scene mean; scene-block bootstrap 10000 draws, seed 20260911',
              'cpu_only': True, 'model_inference': False,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    config_path = OUT / 'CONFIG.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('configuration differs from existing run')
    write_json(config_path, config)
    if not (OUT / 'TEMPLATE_FIT.json').exists(): fit_templates()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_scene, item) for item in SCENES.items()]
        for future in as_completed(futures): print(json.dumps(future.result()), flush=True)
    write_json(OUT / 'DONE.json', {'done': True, 'scenes': len(SCENES), 'finished_at': time.time()})


if __name__ == '__main__': main()
