"""DPFlow E0: public-train diagnostic, not an official RoCo score."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def epe(pred, gt):
    if pred.shape != gt.shape or pred.shape[-1] != 2:
        raise ValueError(f'Flow shape mismatch: {pred.shape} vs {gt.shape}')
    if not np.isfinite(pred).all():
        raise ValueError('Non-finite model prediction')
    valid = np.isfinite(gt).all(-1)
    if not valid.any():
        raise ValueError('No valid ground-truth pixels')
    errors = np.linalg.norm(pred[valid].astype(np.float64) - gt[valid], axis=-1)
    return dict(epe=float(errors.mean()), valid_pixels=int(valid.sum()), outlier_1px_pct=float(100*(errors > 1).mean()))


def image_grid_gt(gt, shape):
    if gt.shape[:2] == (2*shape[0], 2*shape[1]):
        # PTLFlow Spring diagnostic convention: sample the supersampled grid.
        # Values already use image-pixel displacement units: DO NOT divide by 2.
        gt = gt[::2, ::2]
    if gt.shape != (*shape, 2):
        raise ValueError(f'Unexpected Spring ground truth shape {gt.shape}')
    return gt


def corrupt(images, kind, seed):
    rng = np.random.default_rng(seed)
    if kind == 'clean':
        return images
    if kind == 'noise':
        return [np.clip(im.astype(np.float32)+rng.normal(0, 12, im.shape), 0, 255).astype(np.uint8) for im in images]
    if kind == 'contrast':
        return [np.clip((im.astype(np.float32)-127.5)*0.6+127.5, 0, 255).astype(np.uint8) for im in images]
    if kind == 'blur':
        return [cv2.GaussianBlur(im, (9, 9), 2.0) for im in images]
    raise ValueError(kind)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, default=Path('/ssd8/van/dataset/RoCo/e0/manifest.json'))
    ap.add_argument('--checkpoint', type=Path, default=root/'weights/dpflow-spring-69bac7fa.ckpt')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--height', type=int, default=540)
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--max-pairs', type=int, default=4)
    ap.add_argument('--conditions', nargs='+', choices=['clean', 'noise', 'contrast', 'blur'], default=['clean', 'noise', 'contrast', 'blur'])
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    if min(args.height, args.width) < 128 or args.max_pairs < 1:
        ap.error('dimensions must be >=128; max-pairs >=1')
    if args.conditions[0] != 'clean' or len(set(args.conditions)) != len(args.conditions):
        ap.error('conditions must be unique and start with clean')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA explicitly requested but unavailable')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(20260909)
    torch.backends.cudnn.benchmark = False
    import ptlflow
    from ptlflow.utils.io_adapter import IOAdapter
    from ptlflow.utils.flow_utils import flow_read
    manifest = json.loads(args.manifest.read_text())
    data_root = Path(manifest['root'])
    for f in manifest['files']:
        if sha256(data_root/f['path']) != f['sha256']:
            raise ValueError(f"Data hash mismatch: {f['path']}")
    model = ptlflow.get_model('dpflow', ckpt_path=str(args.checkpoint)).eval().to(args.device)
    # Refuse silently unmatched weights; only a uniform optional 'model.' prefix is accepted.
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state_dict']
    if all(k.startswith('model.') for k in state):
        state = {k[6:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    records = []
    provenance = dict(purpose=manifest['purpose'], official_roco_score=False, gt_protocol='Spring [::2,::2] diagnostic; not official supersampled metric', checkpoint_sha256=sha256(args.checkpoint), manifest_sha256=sha256(args.manifest), ptlflow_commit=subprocess.check_output(['git','-C',str(root/'PTLFlow'),'rev-parse','HEAD'],text=True).strip(), python=platform.python_version(), executable=sys.executable, torch=torch.__version__, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu_name=torch.cuda.get_device_name() if args.device=='cuda' else None, settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, corruption_protocol='synthetic diagnostic, not official RobustSpring: noise sigma12/255 independent frames; contrast0.6 shared; Gaussian blur9 sigma2', timing='per-call IOAdapter + transfer + forward + unscale + output copy; first call includes warmup')
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    with torch.inference_mode(), (args.output/'metrics.jsonl').open('w') as log:
        for pair_index, pair in enumerate(manifest['pairs'][:args.max_pairs]):
            original = [cv2.imread(str(data_root/pair[k])) for k in ['image1','image2']]
            if any(x is None for x in original) or original[0].shape != original[1].shape:
                raise ValueError('Missing or inconsistent image pair')
            shape = original[0].shape[:2]
            adapter = IOAdapter(model.output_stride, shape, target_size=(args.height,args.width), cuda=args.device=='cuda')
            for direction, indices in [('fw', [0,1]), ('bw', [1,0])]:
                gt = image_grid_gt(flow_read(data_root/pair['flow_'+direction]), shape)
                clean_flow = None
                for condition in args.conditions:
                    # Generate once in chronological order so reversed direction uses the same corrupted images.
                    images = corrupt(original, condition, 20260909+pair_index)
                    if args.device == 'cuda':
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    inputs = adapter.prepare_inputs([images[i] for i in indices])
                    outputs = adapter.unscale(model(inputs))
                    pred = outputs['flows'][0,0].permute(1,2,0).float().cpu().numpy()
                    del inputs, outputs
                    if args.device == 'cuda':
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    scores = epe(pred, gt)
                    if condition == 'clean':
                        clean_flow = pred.copy()
                    delta = float(np.linalg.norm(pred-clean_flow,axis=-1).mean())
                    key = f"{pair['id']}_{direction}_{condition}"
                    np.save(args.output/(key+'.npy'),pred)
                    row = dict(id=pair['id'],sequence=pair['sequence'],direction=direction,condition=condition,**scores,delta=delta,seconds=elapsed,peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,output_bytes=int(pred.nbytes))
                    records.append(row);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    groups = {c:[r for r in records if r['condition']==c] for c in args.conditions}
    summary = {c:dict(epe_mean_over_pairs=float(np.mean([r['epe'] for r in rs])),delta_mean_over_pairs=float(np.mean([r['delta'] for r in rs])),seconds_total=sum(r['seconds'] for r in rs)) for c,rs in groups.items()}
    if len(groups)>1:
        ec=summary['clean']['epe_mean_over_pairs'];delta=float(np.mean([v['delta_mean_over_pairs'] for c,v in summary.items() if c!='clean']))
        summary['local_proxy_only']=0.5*ec/0.9925+0.5*(ec+delta)/6.030
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'DONE.json').write_text('{"done":true}\n')


if __name__ == '__main__':
    main()
