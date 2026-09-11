"""Shared fixed inference and ZIP input decoding for diagnostics and submissions."""
import contextlib,functools,json,zipfile
from pathlib import Path
import cv2,numpy as np,torch
from strong_models import Predictor,ROOT
DATA=Path('/ssd8/van/dataset/RoCo/archives')
CONDITIONS=['clean','frost','fog','brightness','contrast','defocus_blur','elastic_transform','gaussian_blur','glass_blur','jpeg_compression','pixelate','motion_blur','zoom_blur','saturate','rain','shot_noise','snow','spatter','impulse_noise','gaussian_noise','speckle_noise']
class FixedPredictor:
    def __init__(self,name,precision='fp16'):
        self.inner=Predictor(name);self.name=name;self.precision=precision
        if precision not in ['fp32','fp16','bf16']:raise ValueError(precision)
        self.provenance=dict(self.inner.provenance)
        self.provenance.update(precision=precision,autocast=precision!='fp32',tf32=False)
    def __call__(self,images):
        dtype={'fp32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[self.precision]
        with torch.autocast('cuda',dtype=dtype,enabled=self.precision!='fp32'):
            return self.inner(images)
class TestImages:
    def __init__(self,condition):
        self.condition=condition
        self.zips={cam:zipfile.ZipFile(DATA/'spring'/f'test_frame_{cam}.zip') for cam in ['left','right']} if condition=='clean' else {'both':zipfile.ZipFile(DATA/'robust_spring'/f'{condition}.zip')}
        self.counts={}
        for z in self.zips.values():
            for p in z.namelist():
                if p.endswith('.png'):
                    parts=p.split('/');seq,cam=parts[2],parts[3].removeprefix('frame_');idx=int(Path(p).stem.split('_')[-1]);self.counts[(seq,cam)]=max(idx,self.counts.get((seq,cam),0))
        assert len(self.counts)==20 and sum(self.counts.values())==2000
    @functools.lru_cache(maxsize=8)
    def image(self,seq,cam,idx):
        idx=max(1,min(self.counts[(seq,cam)],idx))
        name=f'{"spring" if self.condition=="clean" else self.condition}/test/{seq}/frame_{cam}/frame_{cam}_{idx:04d}.png'
        z=self.zips[cam if self.condition=='clean' else 'both']
        im=cv2.imdecode(np.frombuffer(z.read(name),np.uint8),cv2.IMREAD_COLOR)
        if im is None or im.shape!=(1080,1920,3):raise ValueError(name)
        return im
    def triplet(self,seq,cam,source,direction):
        offsets=[-1,0,1] if direction=='FW' else [1,0,-1]
        return [self.image(seq,cam,source+i) for i in offsets]
    def close(self):
        self.image.cache_clear()
        for z in self.zips.values():z.close()
