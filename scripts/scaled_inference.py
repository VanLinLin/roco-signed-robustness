"""One uniform anti-aliased half-resolution ablation; never condition-dependent."""
import cv2,numpy as np
from submission_inference import FixedPredictor
class ScaledPredictor:
    def __init__(self,name,precision='fp16',scale=.5):
        if scale not in [1.,.5]:raise ValueError('Only predeclared native/half settings')
        self.inner=FixedPredictor(name,precision);self.name=name;self.scale=scale
        self.provenance=dict(self.inner.provenance,resize=('native 1080x1920; model padding only' if scale==1. else 'uniform 960x540 input, restored 1920x1080 flow'),spatial_scale=scale,downsampling='OpenCV INTER_AREA',flow_upsampling='OpenCV INTER_LINEAR; x/y displacements scaled to original pixels')
    def __call__(self,images):
        if self.scale==1.:return self.inner(images)
        h,w=images[-1].shape[:2];sh,sw=round(h*self.scale),round(w*self.scale)
        small=[cv2.resize(im,(sw,sh),interpolation=cv2.INTER_AREA) for im in images]
        pred=self.inner(small);full=cv2.resize(pred,(w,h),interpolation=cv2.INTER_LINEAR)
        full[:,:,0]*=w/sw;full[:,:,1]*=h/sh
        if full.shape!=(h,w,2) or not np.isfinite(full).all():raise ValueError('Invalid restored flow')
        return full
