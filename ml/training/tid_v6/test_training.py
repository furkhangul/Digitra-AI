import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from augment import CAMERA_EFFECTS, TrainingTransform, camera_effect, evaluation_transform, stress_transform
from train import Images, calibration, save_torch


def gradient_image():
    pixels=np.zeros((150,230,3),dtype=np.uint8)
    pixels[:,:,0]=np.arange(230,dtype=np.uint8)[None,:]
    pixels[:,:,1]=np.arange(150,dtype=np.uint8)[:,None]
    pixels[:,:,2]=120
    return Image.fromarray(pixels)


def test_camera_effects_preserve_shape_and_valid_pixels():
    im=gradient_image()
    for effect in CAMERA_EFFECTS:
        result=camera_effect(im,effect)
        assert result.size==im.size and result.mode=='RGB'
        assert np.asarray(result).std()>1


def test_augmentation_seed_reproducible_and_eval_independent():
    im=gradient_image();aug=TrainingTransform()
    def sample():
        random.seed(3);np.random.seed(3);torch.manual_seed(3)
        return aug(im)
    first=sample();second=sample()
    torch.testing.assert_close(first,second)
    assert first.shape==(3,224,224) and torch.isfinite(first).all()
    baseline=evaluation_transform(im)
    for _ in range(3):aug(im)
    torch.testing.assert_close(baseline,evaluation_transform(im))
    for variant in ['mirror','far','shift','rotation','low_light','blur','jpeg']:
        assert torch.isfinite(stress_transform(im,variant)).all()


def test_synthetic_supervision_weight_and_rgba_compositing(tmp_path):
    p=tmp_path/'hand.png';Image.new('RGBA',(128,128),(200,150,100,180)).save(p)
    df=pd.DataFrame([{'path':str(p),'label':'A','source':'digitra_3d'}])
    image,target,weight=Images(df,['A'],TrainingTransform(),synthetic_weight=.2)[0]
    assert weight==.2 and target==0 and image.shape==(3,224,224)


def test_frozen_split_and_synthetic_held_out():
    root=Path(__file__).resolve().parents[3]
    df=pd.read_csv(root/'artifacts/tid-v6/data/manifest.csv',encoding='utf-8-sig')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not set(df[df.split==a].sha256)&set(df[df.split==b].sha256)
        assert not set(df[df.split==a].pseudo_session)&set(df[df.split==b].pseudo_session)
    synthetic=pd.read_csv(root/'artifacts/tid-v6/synthetic/manifest.csv',encoding='utf-8-sig')
    assert set(synthetic.split)<= {'train','auxiliary'}


def test_low_confidence_calibration_has_valid_fallback():
    logits=np.zeros((60,29),dtype=np.float32);truth=np.arange(60)%29
    temp,threshold=calibration(logits,truth)
    assert .4<=temp<=3. and threshold['coverage']>=.5


def test_ema_batchnorm_and_safe_resume_serialization(tmp_path):
    model=nn.Sequential(nn.Linear(3,3),nn.BatchNorm1d(3))
    ema=AveragedModel(model,multi_avg_fn=get_ema_multi_avg_fn(.98),use_buffers=True)
    model(torch.rand(4,3));ema.update_parameters(model)
    model(torch.rand(4,3));ema.update_parameters(model)
    p=tmp_path/'last.pt'
    save_torch(p,{'ema':ema.state_dict(),'numpy_rng':torch.from_numpy(np.random.get_state()[1].astype(np.int64)),'python_rng':random.getstate()})
    restored=torch.load(p,weights_only=True)
    ema.load_state_dict(restored['ema'])
    assert restored['numpy_rng'].shape==(624,)
