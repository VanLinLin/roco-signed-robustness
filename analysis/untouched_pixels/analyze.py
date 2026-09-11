"""Read-only dense public-WAFT predictions; write only this analysis directory."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import cv2
import h5py
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PRED=ROOT/'runs/submission_v1/predictions'
ARCHIVES=Path('/ssd8/van/dataset/RoCo/archives')
CONDITIONS=['spatter','frost','rain','snow','zoom_blur']
CONFIG_SHA='8442c435788f44ace3f3132acbe9b31df88c7269c3d837c19ab056c78de40653'
H,W=1080,1920


def write_json(path,value):
    part=path.with_suffix('.part');part.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');part.replace(path)


def sums(error,clean_mag,corrupt_mag,mask=None):
    if mask is None:
        return dict(n=int(error.size),epe_sum=float(error.sum(dtype=np.float64)),
                    clean_mag_sum=float(clean_mag.sum(dtype=np.float64)),corrupt_mag_sum=float(corrupt_mag.sum(dtype=np.float64)))
    n=int(mask.sum())
    if not n:return dict(n=0,epe_sum=0.,clean_mag_sum=0.,corrupt_mag_sum=0.)
    return dict(n=n,epe_sum=float(error[mask].sum(dtype=np.float64)),
                clean_mag_sum=float(clean_mag[mask].sum(dtype=np.float64)),corrupt_mag_sum=float(corrupt_mag[mask].sum(dtype=np.float64)))


def subtract(a,b):return {k:a[k]-b[k] for k in a}


def clean_correspondence_mask(flow,candidate,target_unchanged):
    # A stricter descriptive control using clean *prediction*, not hidden GT.
    out=np.zeros(candidate.shape,bool);indices=np.flatnonzero(candidate)
    if not len(indices):return out
    yy,xx=np.divmod(indices,W)
    tx=xx+flow.reshape(-1,2)[indices,0];ty=yy+flow.reshape(-1,2)[indices,1]
    valid=(tx>=0)&(tx<W-1)&(ty>=0)&(ty<H-1)
    indices=indices[valid];xi=np.floor(tx[valid]).astype(np.int32);yi=np.floor(ty[valid]).astype(np.int32)
    ok=target_unchanged[yi,xi]&target_unchanged[yi,xi+1]&target_unchanged[yi+1,xi]&target_unchanged[yi+1,xi+1]
    out.ravel()[indices[ok]]=True
    return out


class Images:
    def __init__(self,scene,camera):
        self.scene=scene;self.camera=camera;self.zips={};self.cache={};self.manifest={}
        for c in ['clean']+CONDITIONS:
            path=ARCHIVES/(f'spring/test_frame_{camera}.zip' if c=='clean' else f'robust_spring/{c}.zip')
            self.zips[c]=zipfile.ZipFile(path);self.cache[c]=OrderedDict()
    def read(self,c,t):
        if t in self.cache[c]:return self.cache[c][t]
        member=f'{"spring" if c=="clean" else c}/test/{self.scene}/frame_{self.camera}/frame_{self.camera}_{t:04d}.png'
        raw=self.zips[c].read(member);image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if image is None or image.shape!=(H,W,3):raise ValueError(member)
        self.manifest[member]=dict(member=member,sha256=hashlib.sha256(raw).hexdigest())
        small=cv2.resize(image,(960,540),interpolation=cv2.INTER_AREA)
        self.cache[c][t]=(image,small)
        while len(self.cache[c])>2:self.cache[c].popitem(last=False)
        return image,small
    def close(self):
        for z in self.zips.values():z.close()


def worker(task):
    cv2.setNumThreads(1)
    scene,camera,length,records=task
    target=OUT/'workers'/f'{scene}_{camera}.json'
    if target.exists():
        saved=json.loads(target.read_text())
        assert saved['done'] and saved['config_sha256']==CONFIG_SHA
        return dict(scene=scene,camera=camera,resumed=True)
    started=time.time();images=Images(scene,camera);verified={};rows=[];mask_cache={c:OrderedDict() for c in CONDITIONS}
    def flow(c,direction,t):
        relative=f'{c}/test/{scene}/flow_{direction}_{camera}/flow_{direction}_{camera}_{t:04d}.flo5'
        row=records[relative];raw=(PRED/relative).read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if digest!=row['sha256'] or len(raw)!=row['bytes']:raise ValueError('Prediction integrity: '+relative)
        with h5py.File(io.BytesIO(raw),'r') as f:
            dataset=f['flow']
            if dataset.shape!=(H,W,2) or dataset.dtype!=np.float16:raise ValueError('Prediction format: '+relative)
            result=dataset[()].astype(np.float64)
        if not np.isfinite(result).all():raise ValueError('Nonfinite prediction: '+relative)
        verified[relative]=dict(path=relative,sha256=digest,bytes=len(raw))
        return result
    def masks(c,t):
        if t in mask_cache[c]:return mask_cache[c][t]
        clean,small_clean=images.read('clean',t);corrupt,small_corrupt=images.read(c,t)
        same=np.all(clean==corrupt,axis=-1)
        model_same=np.all(small_clean==small_corrupt,axis=-1)
        local=cv2.erode(model_same.astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        local=cv2.resize(local,(W,H),interpolation=cv2.INTER_NEAREST).astype(bool)
        mask_cache[c][t]=(same,local)
        while len(mask_cache[c])>2:mask_cache[c].popitem(last=False)
        return same,local
    for t in range(1,length):
        pair_masks={}
        for c in CONDITIONS:
            ma,la=masks(c,t);mb,lb=masks(c,t+1)
            pair_masks[c]=(ma,mb,la,lb,ma&mb)
        for direction,source,target_frame in [('FW',t,t+1),('BW',t+1,t)]:
            clean_flow=flow('clean',direction,source)
            clean_mag=np.hypot(clean_flow[...,0],clean_flow[...,1])
            for c in CONDITIONS:
                corrupt_flow=flow(c,direction,source)
                delta=clean_flow-corrupt_flow
                error=np.hypot(delta[...,0],delta[...,1]);corrupt_mag=np.hypot(corrupt_flow[...,0],corrupt_flow[...,1])
                ma,mb,la,lb,both=pair_masks[c]
                source_mask,target_mask,local_source,local_target=(ma,mb,la,lb) if direction=='FW' else (mb,ma,lb,la)
                local_both=both&la&lb
                correspondence=clean_correspondence_mask(clean_flow,both&local_source,target_mask&local_target)
                stats={'all':sums(error,clean_mag,corrupt_mag),
                       'source_untouched':sums(error,clean_mag,corrupt_mag,source_mask),
                       'both_untouched':sums(error,clean_mag,corrupt_mag,both),
                       'local5_untouched':sums(error,clean_mag,corrupt_mag,local_both),
                       'correspondence_local5_untouched':sums(error,clean_mag,corrupt_mag,correspondence)}
                stats['source_touched']=subtract(stats['all'],stats['source_untouched'])
                stats['both_touched']=subtract(stats['all'],stats['both_untouched'])
                rows.append(dict(scene=scene,camera=camera,direction=direction,source=source,target=target_frame,condition=c,stats=stats))
                del corrupt_flow,delta,error,corrupt_mag
            del clean_flow,clean_mag
        if t%5==0 or t==length-1:
            elapsed=time.time()-started
            write_json(OUT/'progress'/f'{scene}_{camera}.json',dict(scene=scene,camera=camera,
                pairs_done=len(rows),pairs_expected=(length-1)*10,seconds=elapsed,
                comparisons_per_second=len(rows)/elapsed,updated_at=time.time()))
    images.close()
    assert len(verified)==(length-1)*12
    result=dict(done=True,config_sha256=CONFIG_SHA,scene=scene,camera=camera,length=length,
                rows=rows,prediction_files=list(verified.values()),inputs=list(images.manifest.values()),seconds=time.time()-started)
    write_json(OUT/'workers'/f'{scene}_{camera}.json',result)
    return dict(scene=scene,camera=camera,comparisons=len(rows),seconds=time.time()-started)


def main():
    for sub in ['workers','progress']:(OUT/sub).mkdir(exist_ok=True)
    raw_config=(PRED/'inference_config.json').read_bytes();assert hashlib.sha256(raw_config).hexdigest()==CONFIG_SHA
    config=json.loads(raw_config);assert config['spatial_scale']==.5 and config['checkpoint_sha256']=='04a4560e834020bbf0da9b605c0cbdbc63020b507ef165152c3bc344386866e4'
    records={};journal_hashes={}
    for camera in ['left','right']:
        journal=(PRED/f'journal_{camera}.jsonl').read_bytes();digest=hashlib.sha256(journal).hexdigest()
        done=json.loads((PRED/f'DONE_{camera}.json').read_text())
        assert done['done'] and done['config_sha256']==CONFIG_SHA and done['journal_sha256']==digest
        journal_hashes[camera]=digest
        for line in journal.splitlines():
            row=json.loads(line)
            if row['condition'] in ['clean']+CONDITIONS:
                assert row['path'] not in records;records[row['path']]=row
    assert len(records)==23760
    with zipfile.ZipFile(ARCHIVES/'spring/test_frame_left.zip') as archive:
        lengths={}
        for member in archive.namelist():
            if member.endswith('.png'):
                scene=member.split('/')[2];lengths[scene]=lengths.get(scene,0)+1
    protocol=dict(config_sha256=CONFIG_SHA,prediction_config=config,prediction_root=str(PRED),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),journal_sha256=journal_hashes,
        scenes=lengths,conditions=CONDITIONS,cpu_only=True,model_inference=False,
        primary_untouched='native RGB bit-identical to corresponding clean at the SAME raster position in BOTH input frames',
        secondary_source='native RGB bit-identical in the SOURCE frame only',
        local5='primary mask AND a 5x5 neighborhood in both actual 960x540 uint8 INTER_AREA inputs is bit-identical; nearest block association to native output',
        correspondence_local5='primary mask AND source local5 unchanged AND four pixels supporting the clean-predicted target correspondence are native+local5 unchanged; in-bounds only; predicted correspondence, not GT',
        epe='float64 L2(clean predicted flow - corrupt predicted flow), native image pixels; not GT EPE or stereo disparity',
        motion_ratio='sum(||corrupt flow||)/sum(||clean flow||) on the same subset; not mean individual ratios',
        empty_subset='N=0: EPE/mean magnitudes/ratio are null, not zero',
        aggregation='pixel-pooled within each subset across 3960 directed flow fields per corruption; scene-block bootstrap 10000 draws seed20260910')
    p=OUT/'CONFIG.json'
    if p.exists():assert json.loads(p.read_text())==protocol,'Existing analysis configuration differs'
    write_json(p,protocol)
    tasks=[]
    for scene,length in sorted(lengths.items()):
        for camera in ['left','right']:
            selected={k:v for k,v in records.items() if v['sequence']==scene and v['camera']==camera}
            assert len(selected)==(length-1)*12
            tasks.append((scene,camera,length,selected))
    with ProcessPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(worker,t) for t in tasks]):print(json.dumps(future.result()),flush=True)
    write_json(OUT/'DONE.json',dict(done=True,workers=20,comparisons=19800,finished_at=time.time()))


if __name__=='__main__':main()
