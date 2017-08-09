#!/usr/bin/env python
# -*- coding: utf-8 -*- 
#
# dump_pytorch_features.py - compute pytorch for comparison vs. mcn
#
# Since matlab does not support calling recent versions of python, 
# the process now consists of:
#   1. Dumping all intermediate pytorch features to disk
#   2. Loading the imported matconvnet network in matlab and comparing
#      each computed feature
# 
# This script simply performs step (1) above.
#
# --------------------------------------------------------
# mcnPyTorch
# Licensed under The MIT License [see LICENSE.md for details]
# Copyright (C) 2017 Samuel Albanie 
# --------------------------------------------------------

import sys
import numpy as np
import scipy.io
from PIL import Image
from ast import literal_eval as make_tuple
from torch.autograd import Variable
import torchvision.transforms as transforms
from pathlib import Path

sys.path.insert(0, 'python')
import pytorch_utils as pl

# set path for the feature dump and set sample image
sample_im_path = 'test/peppers.png'
feature_path = Path('../../data/mcnPyTorch/pyt-feats.mat')
if not feature_path.parent.exists(): feature_path.parent.mkdir()

# parse args
parser = pl.set_conversion_kwargs()
p = parser.parse_args(sys.argv[1:])

# load pytorch model
if p.model_def and p.model_weights:
    paths = {'def': p.model_def, 'weights': p.model_weights}
else:
    paths = {}
net,flatten_loc = pl.load_pytorch_model(p.pytorch_model, paths=paths)

# compute activations for a sample image, using standard pytorch normalisation
im_orig = Image.open(sample_im_path).convert('RGB')
image_size = tuple(make_tuple(p.image_size))
im = np.array(im_orig.resize(image_size))
im_mean = [0.485, 0.456, 0.406] ; im_std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=im_mean, std=im_std)
transform = transforms.Compose([transforms.ToTensor(),normalize])
x = Variable(transform(im).unsqueeze(0))
py_feats_tensors = pl.compute_intermediate_feats(net.eval(), x, flatten_loc)

# form dict for dumping
feat_dump = {}
for ii, feat in enumerate(py_feats_tensors): 
    featKay = 'x{}'.format(ii)
    feat_dump[featKay] = feat.data.numpy()

# save to disk
scipy.io.savemat(str(feature_path), feat_dump)
print('pytorch features have been saved to {}'.format(feature_path))
