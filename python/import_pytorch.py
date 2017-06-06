from IPython import get_ipython
ipython = get_ipython()
ipython.magic('load_ext autoreload')
ipython.magic('autoreload 2')

# breakpoint on exception
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
     color_scheme='Linux', call_pdb=1)

import matplotlib 
matplotlib.use('Agg')

import sys

import matplotlib.pyplot as plt
sys.path.insert(0, '/users/albanie/coding/src/zsvision/python') # lazy
from zsvision.zs_iterm import zs_dispFig

import torch
from pathlib import Path
import numpy as np
import scipy.io
import cv2
import ipdb
import torchvision
from collections import OrderedDict
import pytorch_layers as pl
from torch.autograd import Variable

# forward pass to compute sizes
im = scipy.misc.face()
im = scipy.misc.imresize(im, [227,227])
plt.imshow(im) 
zs_dispFig()


# model_dir = Path('/users/albanie/coding/libs/convert_torch_to_pytorch/models')
# vgg = model_dir / 'vgg16-397923af.pth'

# params = torch.load(str(vgg))

# N,C,H,W
# alexnet = models.alexnet(pretrained=True)
# vgg = models.vgg16(pretrained=True)
# params = alexnet.state_dict()
keepDropout = False
target = 'alexnet'
save_path = '../models/{}-mcn.mat'.format(target)

class ImTransform(object):
    """
    resize (int): input dims
    rgb ((int,int,int)): average RGB of the dataset (104,117,123)
    """
    def __init__(self, imsz, rgb, swap=(2, 0, 1)):
        self.mean_im = rgb
        self.imsz = imsz
        self.swap = swap

    def __call__(self, im):
        im = cv2.resize(np.array(im), self.imsz).astype(np.float32)
        im -= self.mean_im
        print(im.shape)
        im = im.transpose(self.swap)
        return torch.from_numpy(im)

# --------------------------------------------------------------------
#                                                          Load layers
# --------------------------------------------------------------------
supported_models = [torchvision.models.resnet.ResNet]

if target == 'alexnet':
    net = torchvision.models.alexnet(pretrained=True)
    imsz = (227, 227)
elif target == 'vgg16':
    net = torchvision.models.vgg16(pretrained=True)
elif target == 'resnet50':
    net = torchvision.models.resnet50(pretrained=True)
else:
    raise ValueError('target not recognised')

params = net.state_dict()

transform = ImTransform(imsz, (104, 117, 123), (2, 0, 1))
x = Variable(transform(im).unsqueeze(0))
y = net(x)

# this is probably silly, but I don't know enough about how pyTorch
# works to do it sensibly
def get_feat_sizes(net, x, sizes=[]):
   if len(list(net.children())) == 0:
       return [pl.tolist(net(x).size())]
   trunk = torch.nn.Sequential(*list(net.children())[:-1])
   sizes = [*get_feat_sizes(trunk, x, sizes), pl.tolist(net(x).size())]
   return sizes
   
feat_sizes = get_feat_sizes(net.features, x, sizes=[])
rand_feats = np.random.random(feat_sizes[-1])

x = Variable(torch.from_numpy(rand_feats)).float()
x = x.view(x.size(0), -1)
cls_sizes = get_feat_sizes(net.classifier, x, sizes=[])

# TODO: Insert flatten operation
# TODO: refactor messy connectors into state dict

sizes = feat_sizes + cls_sizes

for sz in sizes:
    print(sz)
   

# rename keys to make compatible (duplicates params)
tmp = OrderedDict()
for key in params:
    new_name = key.replace('.', '_')
    tmp[new_name] = params[key]
params = tmp 

def process_custom_module(name, module, in_vars, sizes):
    layers = []
    if isinstance(module, torchvision.models.resnet.Bottleneck):
        id_var = in_vars
        downsample = hasattr(module, 'downsample') and bool(module.downsample)
        children = list(module.named_children())
        assert len(children) == 7 + downsample, 'unexpected bottleneck size'
        block = construct_layers(children[:6], in_vars=in_vars, prefix=name, sizes=sizes)
        layers.extend(block)
        in_vars = block[-1].outputs

        if downsample:
            down_block = construct_layers([children[-1]], id_var, prefix=name, sizes=sizes)
            layers.extend(down_block)
            id_var = down_block[-1].outputs

        cat_name = '{}_cat'.format(name)
        cat_layer = pl.PTConcat(cat_name, [*id_var, *in_vars], [cat_name], 3)
        layers.append(cat_layer)
        in_vars = cat_layer.outputs

        relu_idx = [child[0] for child in children].index('relu')
        assert relu_idx > 0, 'relu not found'
        relu = construct_layers([children[relu_idx],], in_vars=in_vars, prefix=name, sizes=sizes)
        layers.extend(relu) # note that relu is a "one-layer" list
    else:
        raise ValueError('unrecognised module {}'.format(type(module)))
    return layers

def print_size_info(name, module, sizes):
    """
    print size summary and perform some sanity checks
    """
    print('{}: {}'.format(name, module))
    in_sz, out_sz = sizes[0], sizes[1]
    print(' +  size: {} -> {}'.format(in_sz, out_sz))
    if module in ['ReLU', 'BatchNorm']:
        assert in_sz == out_sz, 'sizes should match for {}'.format(module)

def construct_layers(graph, in_vars, prefix='', sizes=[]):
    layers = [] 
    for name, module in graph:
        name = name.replace('.', '_') # make comatible with MATLAB
        if prefix: name = '{}_{}'.format(prefix, name)
        opts, out_vars, rec_out = {}, [name,], None

        if isinstance(module, torch.nn.modules.conv.Conv2d):
            print_size_info(name, 'Conv2d', sizes)
            sizes.pop(0)
            opts['bias_term'] = bool(module.bias)
            opts['num_output'] = module.out_channels
            opts['kernel_size'] = module.kernel_size
            opts['pad'] = module.padding
            opts['stride'] = module.stride
            opts['dilation'] = module.dilation
            opts['group'] = module.groups
            layer = pl.PTConv(name, in_vars, out_vars, **opts)
            layers.append(layer)

        elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
            print_size_info(name, 'BatchNorm', sizes)
            sizes.pop(0)
            opts['eps'] = module.eps
            opts['use_global_stats'] = module.affine
            opts['moving_average_fraction'] = module.momentum
            layer = pl.PTBatchNorm(name, in_vars, out_vars, **opts)
            layers.append(layer)

        elif isinstance(module, torch.nn.modules.activation.ReLU):
            print_size_info(name, 'ReLU', sizes)
            sizes.pop(0)
            layer = pl.PTReLU(name, in_vars, out_vars)
            layers.append(layer)

        elif isinstance(module, torch.nn.modules.dropout.Dropout):
            print_size_info(name, 'Dropout', sizes)
            sizes.pop(0)
            if keepDropout:
                opts['ratio'] = module.p # TODO: check that this shouldn't be 1 - p
                layer = pl.PTDropout(name, in_vars, out_vars, **opts)
                layers.append(layer)
            else:
                out_vars = in_vars

        elif isinstance(module, torch.nn.modules.pooling.MaxPool2d):
            print_size_info(name, 'MaxPool', sizes)
            sizes.pop(0)
            opts['method'] = 'max'
            opts['pad'] = module.padding
            opts['stride'] = module.stride
            opts['kernel_size'] = module.kernel_size
            layer = pl.PTPooling(name, in_vars, out_vars, **opts)
            layers.append(layer)

        elif isinstance(module, torch.nn.modules.pooling.AvgPool2d):
            print_size_info(name, 'AvgPool', sizes)
            sizes.pop(0)
            opts['method'] = 'avg'
            opts['pad'] = module.padding
            opts['stride'] = module.stride
            opts['kernel_size'] = module.kernel_size
            layer = pl.PTPooling(name, in_vars, out_vars, **opts)
            layers.append(layer)

        elif isinstance(module, torch.nn.modules.linear.Linear):
            ipdb.set_trace()
            print('{}: Linear, output size: {}'.format(name, sizes.pop(0)))
            opts['bias_term'] = bool(module.bias)
            opts['num_output'] = module.out_features
            opts['kernel_size'] = [1,1]
            opts['pad'] = [0,0]
            opts['stride'] = [1,1]
            opts['dilation'] = [1,1]
            opts['group'] = 1
            layer = pl.PTConv(name, in_vars, out_vars, **opts)
            layers.append(layer)
            
        elif isinstance(module, torchvision.models.resnet.Bottleneck):
            #print('processing bottleneck: {}'.format(name))
            layers_ = process_custom_module(name, module, in_vars, sizes) # soz Guido
            layers.extend(layers_)

        elif isinstance(module, torch.nn.modules.container.Sequential):
            print('flattening sequence: {}'.format(name))
            children = list(module.named_children())
            layers_,sizes = construct_layers(children, in_vars=in_vars, 
                                             prefix=name, sizes=sizes)
            out_vars = layers_[-1].outputs
            layers.extend(layers_)
        elif type(module) in supported_models:
            pass
        else:
            raise ValueError('unrecognised module {}'.format(type(module)))
        in_vars = out_vars
    return layers, sizes

mods = list(net.modules())
graph = mods[0].named_children()
in_vars = ['data',]
layers = construct_layers(graph, in_vars=in_vars, sizes=sizes)

# load layers into model and set params
# ptmodel = pl.PTModel()
# for layer in layers:
    # name = layer.name
    # ptmodel.add_layer(layer)
    # layer.setTensor(ptmodel, params)
    # # load parameter values
    # print('{} has type: {}'.format(name, type(layer)))

# for l in layers:
   # print('---------')
   # print('name', l.name)
   # print('inputs', l.inputs)
   # print('outputs', l.outputs)

# # --------------------------------------------------------------------
# #                                                        Normalization
# # --------------------------------------------------------------------

# # standard normalization for vision models in pytorch
# average_image = np.array((0.485, 0.456, 0.406),dtype='float')
# image_std = np.array((0.229, 0.224, 0.225), dtype='float') 

# minputs = np.empty(shape=[0,], dtype=pl.minputdt)
# dataShape = [224,224,3] # hardcode as a temp fix
# fullImageSize = [256, 256]
# print('Input image data tensor shape:', dataShape)
# print('Full input image size:', fullImageSize)

# mnormalization = {
  # 'imageSize': pl.row(dataShape),
  # 'averageImage': average_image,
  # 'imageStd': image_std, 
  # 'interpolation': 'bilinear',
  # 'keepAspect': True,
  # 'border': pl.row([0,0]),
  # 'cropSize': 1.0}

# # --------------------------------------------------------------------
# #                                                    Convert to MATLAB
# # --------------------------------------------------------------------

# # net.meta
# meta_dict = {'inputs': minputs.reshape(1,-1), 'normalization': mnormalization}
# mmeta = pl.dictToMatlabStruct(meta_dict)

# # This object should stay a dictionary and not a NumPy array due to
# # how NumPy saves to MATLAB
# mnet = {'layers': np.empty(shape=[0,], dtype=pl.mlayerdt),
        # 'params': np.empty(shape=[0,], dtype=pl.mparamdt),
        # 'meta': mmeta}

# for layer in ptmodel.layers.values():
    # mnet['layers'] = np.append(mnet['layers'], layer.toMatlab(), axis=0)
 
# for param in ptmodel.params.values():
    # mnet['params'] = np.append(mnet['params'], param.toMatlab(), axis=0)

# # to row
# mnet['layers'] = mnet['layers'].reshape(1,-1)
# mnet['params'] = mnet['params'].reshape(1,-1)

# # --------------------------------------------------------------------
# #                                                          Save output
# # --------------------------------------------------------------------

# print('Saving network to {}'.format(save_path))
# scipy.io.savemat(save_path, mnet, oned_as='column')
