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

# generate image and convert to var
im = Image.open('sample.jpg')
im = im.resize((224,224))
im = np.array(im) 
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(),normalize])
x = Variable(transform(im).unsqueeze(0))

feature_feats = get_inter_feats(net.features, x)

# create image to pass to MATLAB and compute the feature maps
im_np = np.array(torch.squeeze(x.data,0).numpy())
im_np = np.transpose(im_np, (1,2,0)) # return from CxHxW to HxWxC
mcn_im = im_np.flatten().tolist() # no numpy support
mcn_feats = eng.mcn_check(mcn_im, im_np.shape)
print('num feature maps: {}'.format(len(mcn_feats)))
