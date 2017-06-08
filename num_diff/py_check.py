import sys
import matplotlib, os
matplotlib.use('Agg')

import matplotlib.pyplot as plt
path = os.path.expanduser('~/coding/src/zsvision/python')
sys.path.insert(0, path) # lazy
from zsvision.zs_iterm import zs_dispFig

sys.path.insert(0, '../python') 

import copy
import torch
from pathlib import Path
import numpy as np
import scipy.io
import cv2
import ipdb
import torchvision
from collections import OrderedDict
import pytorch_layers as pl
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as transforms

# compare against matconvnet
import matlab.engine
eng = matlab.engine.start_matlab()

net = torchvision.models.alexnet(pretrained=True)

def get_inter_feats(net, x, feats=[]):
   if len(list(net.children())) == 0:
       return [net(x)]
   trunk = torch.nn.Sequential(*list(net.children())[:-1])
   sizes = [*get_inter_feats(trunk, x, feats), net(x)]
   return sizes

dropout_removed = True

# generate image and convert to var
im = Image.open('sample.jpg')
im = np.array(im.resize((224,224)))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(),normalize])
x = Variable(transform(im).unsqueeze(0))

feature_feats = get_inter_feats(net.features.eval(), x)
last = feature_feats[-1]
last = last.view(last.size(0), -1)
classifier_feats = get_inter_feats(net.classifier.eval(), last)
py_feats_tensors = feature_feats + classifier_feats

# create image to pass to MATLAB and compute the feature maps
im_np = np.array(torch.squeeze(x.data,0).numpy())
#im_np = np.transpose(im_np, (1,2,0)) # return from CxHxW to HxWxC
mcn_im = im_np.flatten().tolist() # no numpy support
mcn_feats_ = [np.array(x) for x in eng.get_mcn_features(mcn_im, im_np.shape)]
py_feats = [np.squeeze(x.data.numpy()) for x in py_feats_tensors]
mcn_feats = [np.squeeze(np.transpose(x, (2,0,1))) for x in mcn_feats_] # to CxHxW

print('num mcn feature maps: {}'.format(len(mcn_feats)))
print('num py feature maps: {}'.format(len(py_feats)))

# determine feature pairing (accounts for the extra layers created to 
# match the flattening performed before the classifier in pytorch, as 
# well as the removal of dropout layers)
def module_execution_order(module):
    modules = []
    children = list(module.children())
    if len(children) == 0:
        modules.append(module)
    else:
        for module in children:
            modules.extend(module_execution_order(module))
    return modules

def get_feature_pairs(net):
    feat_modules = module_execution_order(net.features)
    classifier_modules = module_execution_order(net.classifier)
    modules = feat_modules + classifier_modules
    py_feat_idx = list(range(len(modules) + 2))
    dropout_idx = [i + 1 for i,x in enumerate(modules)  # +1 for input im
            if isinstance(x, torch.nn.modules.dropout.Dropout)]
    pairs = [] 
    cursor = 0
    for py_idx in py_feat_idx:
        if py_idx == len(feat_modules) + 1:
            cursor += 1 # mcn flattening procedure uses an extra layer
        if py_idx in dropout_idx and dropout_removed:
            continue
        print(py_idx, cursor)
        pairs.append([py_idx, cursor])
        cursor += 1
    return pairs

pairs = get_feature_pairs(net)

for py_idx, mcn_idx in pairs:
    py_feat = py_feats[py_idx]
    mcn_feat = mcn_feats[mcn_idx]
    print('{}v{}: size py: {} vs size mcn: {}'.format(py_idx,mcn_idx,
                      py_feat.shape, mcn_feat.shape))
    diff = np.absolute(py_feat - mcn_feat).mean()
    if diff > 1e-5:
        print('diff: {}'.format(diff))
        print('py mean: {}'.format(py_feat.mean()))
        print('mcn mean: {}'.format(mcn_feat.mean()))
