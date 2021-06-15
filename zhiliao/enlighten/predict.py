from .models.single_model import SingleModel

import torch
from PIL import Image
import cv2 as cv

import argparse
opt_dict = {
    'gpu_ids':[0],
    'checkpoints_dir':'./checkpoints',
    'which_model_netG':'sid_unet_resize',
    'name':'enlightening',
    'resize_or_crop':'no',
    'isTrain':False,
    'no_dropout':True,
    'no_flip':True,
    'loadSize':286,
    'fineSize':256,
    'batchSize':1,
    'input_nc':3,
    'output_nc':3,
    'vgg':0,
    'vgg_mean':None,
    'IN_vgg':None,
    'fcn':0,
    'skip':1,
    'ngf':64,
    'use_norm':1,
    'norm':'instance',
    'which_epoch':200,
    'patchD':None,
    'patchD_3':0,
    'vary':1,
    'low_times':200,
    'high_times':400,
    'noise':0,
    'input_linear':None,
    'use_wgan':0,
    'use_ragan':None,
    'hybrid_loss':None,
    'D_P_times2':None,
    'new_lr':None,
    'lr':None,
    'niter_decay':None,
    'self_attention':True,
    'syn_norm':False,
    'use_avgpool':0,
    'tanh':False,
    'times_residual':True,
    'linear_add':False,
    'linear':False,
    'latent_threshold':False,
    'latent_norm':False,
    }
opt = argparse.Namespace(**opt_dict)


if __name__ == '__main__':
    img = Image.open("test.jpg").convert('RGB')
    with torch.no_grad():

        model = SingleModel()
        model.initialize(opt)

        fake_B = model.predict(img)
        cv.imshow('debug',fake_B)
        cv.waitKey(0)
