import sys, os
sys.path.insert(0, '../python') 

import ipdb
debug = 0

if debug:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = os.path.expanduser('~/coding/src/zsvision/python')
    sys.path.insert(0, path) # lazy
    from zsvision.zs_iterm import zs_dispFig

import torch
import argparse
import numpy as np
import scipy.io
import torchvision
from PIL import Image
from ast import literal_eval as make_tuple
import pathlib
from torch.autograd import Variable
import torchvision.transforms as transforms

if 1:
    sys.path.insert(0, os.path.expanduser('~/local/matlab-engine/lib'))
    sys.path.insert(0, 'python')

import pytorch_utils as pl

# compare against matconvnet
import matlab.engine
eng = matlab.engine.start_matlab()
cwd = pathlib.Path.cwd()

parser = argparse.ArgumentParser(
   description='Check activations of MatConvNet model imported from PyTorch.')
parser.add_argument('py_model',
                    type=str,
                    help='The input should be the name of a pytorch model \
                      (if present in pytorch.visionmodels), otherwise it \
                      should be a path to its .pth file')
parser.add_argument('mcn_model',
                    type=str,
                    help='Output MATLAB file')
parser.add_argument('--image-size',
                    type=str,
                    nargs='?',
                    default='[224,224]',
                    help='Size of the input image')
parser.add_argument('--remove-dropout', #TODO(sam): clean up, determine automatically
                    dest='remove_dropout',
                    action='store_true',
                    default=False,
                    help='Remove dropout layers') 
parser.add_argument('--is-torchvision-model',
                    type=bool,
                    nargs='?',
                    default=True,
                    help='is the model part of the torchvision.models')
args = parser.parse_args()

# params = torch.load(str(vgg))
if args.is_torchvision_model:
    net = pl.load_valid_pytorch_model(args.py_model)
else:
    raise ValueError('not yet supported')

def get_inter_feats(net, x, feats=[]):
   if len(list(net.children())) == 0:
       return [net(x)]
   trunk = torch.nn.Sequential(*list(net.children())[:-1])
   feats = [*get_inter_feats(trunk, x, feats), net(x)]
   return feats

# generate image and convert to var
im_orig = Image.open(str(cwd / 'test/peppers.png')).convert('RGB')
image_size = tuple(make_tuple(args.image_size))
im = np.array(im_orig.resize(image_size))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(),normalize])
x = Variable(transform(im).unsqueeze(0))

# feature_feats = get_inter_feats(net.features.eval(), x)
# last = feature_feats[-1]
# last = last.view(last.size(0), -1)
# classifier_feats = get_inter_feats(net.classifier.eval(), last)
# py_feats_tensors = feature_feats + classifier_feats
py_feats_tensors = pl.compute_intermediate_feats(net, x)

if 0:
    # sanity check
    # 1. Define the appropriate image pre-processing function.
    preprocessFn = transforms.Compose([transforms.Scale(256), 
                                       transforms.CenterCrop(227), 
                                       transforms.ToTensor(), 
                                       transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])])
    inputVar =  Variable(preprocessFn(im_orig).unsqueeze(0))
    out = net.eval()(inputVar)
    # 2. Load the imagenet class names.
    import json
    imagenetClasses = {int(idx): entry[1] for (idx, entry) in 
                        json.load(open('imagenet_class_index.json')).items()}
    #preds = py_feats_tensors[-1]
    probs, indices = (-torch.nn.Softmax()(out).data).sort()
    probs = (-probs).numpy()[0][:10]; indices = indices.numpy()[0][:10]
    preds = [imagenetClasses[idx] + ': ' + str(prob) for (prob, idx) in zip(probs, indices)]
    m = torch.nn.Softmax()
    probs = m(preds)
    best = probs.max().data.numpy()[0]
    print('top prediction {0:.2f}'.format(best))

# create image to pass to MATLAB and compute the feature maps
im_np = np.array(torch.squeeze(x.data,0).numpy())
mcn_im = im_np.flatten().tolist() # no numpy support
eng.addpath(str(cwd/'test'),nargout=0)
mcn_feats_ = [np.array(x) for x in 
                   eng.get_mcn_features(args.mcn_model, mcn_im, im_np.shape)]
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
        if py_idx in dropout_idx and args.remove_dropout:
            print('drop zone')
            continue
        if debug: print(py_idx, cursor)
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
        raise ValueError('numerical checks failed')

print('Success! the imported mcn-model is numerically equivalent to \
       its PyTorch counterpart')
