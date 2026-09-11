"""Fixed full-resolution Spring checkpoint adapters. BGR uint8 input -> xy flow."""
import json
from pathlib import Path
import numpy as np
import torch
from run_e0 import sha256
ROOT=Path(__file__).resolve().parents[1]
CHECKPOINTS={'dpflow':ROOT/'weights/dpflow-spring-69bac7fa.ckpt','waft_dav2_a2':ROOT/'weights/waft_dav2_a2-spring-04a4560e.ckpt','memfof':ROOT/'weights/MEMFOF-spring/model.safetensors'}
class Predictor:
    def __init__(self,name):
        self.name=name
        torch.set_num_threads(4);torch.manual_seed(20260909);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        path=CHECKPOINTS[name]
        if name=='memfof':
            from memfof import MEMFOF
            from safetensors.torch import load_file
            config=json.loads(path.with_name('config.json').read_text())
            self.model=MEMFOF(**config,backbone_weights=None)
            self.model.load_state_dict(load_file(str(path)),strict=True)
            self.settings=dict(**config,iters=8,scale=0,output='flow[-1][:,1]',triplet='[previous,source,target]; previous clamped at sequence boundary; reversed chronology for BW')
        else:
            import ptlflow
            self.model=ptlflow.get_model_reference(name)()
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            state=checkpoint['state_dict']
            if all(k.startswith('model.') for k in state):state={k[6:]:v for k,v in state.items()}
            self.model.load_state_dict(state,strict=True)
            self.settings=dict(self.model.hparams)
            self.settings['iters']=getattr(self.model,'iters',None)
        self.model.eval().cuda()
        self.provenance=dict(model=name,checkpoint=str(path),checkpoint_sha256=sha256(path),settings=self.settings,precision='FP32; TF32 disabled',resize='native 1080x1920; model padding only',adapter_sha256=sha256(Path(__file__)))
    @torch.inference_mode()
    def __call__(self,images):
        if self.name=='memfof':
            assert len(images)==3
            arr=np.stack([im[:,:,::-1] for im in images]).transpose(0,3,1,2).copy()
            inp=torch.from_numpy(arr).float().unsqueeze(0).cuda()
            out=self.model(inp,iters=8)['flow'][-1][0,1]
        else:
            from ptlflow.utils.io_adapter import IOAdapter
            adapter=IOAdapter(self.model.output_stride,images[-1].shape[:2],cuda=True)
            inp=adapter.prepare_inputs(images[-2:])
            out=adapter.unscale(self.model(inp))['flows'][0,0]
        pred=out.permute(1,2,0).float().cpu().numpy()
        if pred.shape!=(*images[-1].shape[:2],2) or not np.isfinite(pred).all():raise ValueError('Bad flow output')
        return pred
