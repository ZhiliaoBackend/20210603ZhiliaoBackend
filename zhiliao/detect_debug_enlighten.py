import random

def init_enlightenGAN():
    from .enlighten.models.single_model import SingleModel
    import argparse
    opt_dict = {
        'gpu_ids':[0],
        'checkpoints_dir':'./zhiliao/enlighten/checkpoints',
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
    model = SingleModel()
    model.initialize(opt)
    return model

class PredictRes(object):

    __slots__ = ['eye_abnormal','mouth_abnormal']

    def __init__(self):
        self.eye_abnormal = None  # True if eye is closed
        self.mouth_abnormal = None  # True if mouth is opened

class Detection(object):

    __slots__ = []

    def __init__(self, weights, img_size=640):
        pass

    def detect(self, img):
        assert len(img.shape) == 3,"Image must have 3 dimensions"

        pred = PredictRes()
        pred.eye_abnormal = bool(random.randint(0,1))
        pred.mouth_abnormal = bool(random.randint(0,1))

        return pred