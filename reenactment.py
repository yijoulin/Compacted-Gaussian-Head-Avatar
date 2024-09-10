'''
@inproceedings{xu2023gaussianheadavatar,
  title={Gaussian Head Avatar: Ultra High-fidelity Head Avatar via Dynamic Gaussians},
  author={Xu, Yuelang and Chen, Benwang and Li, Zhe and Zhang, Hongwen and Wang, Lizhen and Zheng, Zerong and Liu, Yebin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
'''
import os
import torch
import argparse
from torch import nn

from config.config import config_reenactment

from lib.dataset.Dataset import ReenactmentDataset
from lib.dataset.DataLoaderX import DataLoaderX
from lib.module.GaussianHeadModule import GaussianHeadModule
from lib.module.SuperResolutionModule import SuperResolutionModule
from lib.module.CameraModule import CameraModule
from lib.recorder.Recorder import ReenactmentRecorder
from lib.apps.Reenactment import Reenactment

def restore(state_dict):
    sampled_centers = torch.gather(state_dict['scale_rot_codebook'], 0, state_dict['scale_rot_id'].unsqueeze(-1).repeat(1, 7))
    state_dict['scales'] = nn.Parameter(sampled_centers[:, :3])
    state_dict['rotation'] = nn.Parameter(sampled_centers[:, 3:])
    del state_dict['scale_rot_codebook']
    del state_dict['scale_rot_id']

    sampled_centers_feature = torch.gather(state_dict['feature_codebook'], 0, state_dict['feature_id'].unsqueeze(-1).repeat(1, 128))
    state_dict['feature'] = nn.Parameter(sampled_centers_feature)
    del state_dict['feature_codebook']
    del state_dict['feature_id']
    return state_dict

if __name__ == '__main__':
    compact = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/reenactment_N031.yaml')
    arg = parser.parse_args()

    cfg = config_reenactment()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    dataset = ReenactmentDataset(cfg.dataset)
    dataloader = DataLoaderX(dataset, batch_size=1, shuffle=False, pin_memory=True) 

    device = torch.device('cuda:%d' % cfg.gpu_id)

    gaussianhead_state_dict = torch.load(cfg.load_gaussianhead_checkpoint, map_location=lambda storage, loc: storage)

    if 'scale_rot_codebook' in gaussianhead_state_dict:
        gaussianhead_state_dict = restore(gaussianhead_state_dict)

    gaussianhead = GaussianHeadModule(cfg.gaussianheadmodule, 
                                          xyz=gaussianhead_state_dict['xyz'], 
                                          feature=gaussianhead_state_dict['feature'],
                                          landmarks_3d_neutral=gaussianhead_state_dict['landmarks_3d_neutral']).to(device)
    gaussianhead.load_state_dict(gaussianhead_state_dict)

    supres = SuperResolutionModule(cfg.supresmodule).to(device)
    supres.load_state_dict(torch.load(cfg.load_supres_checkpoint, map_location=lambda storage, loc: storage))

    camera = CameraModule()
    recorder = ReenactmentRecorder(cfg.recorder)

    app = Reenactment(dataloader, gaussianhead, supres, camera, recorder, cfg.gpu_id, dataset.freeview)
    app.run()
