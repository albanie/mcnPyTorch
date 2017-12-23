# Layers and helper methods for pytorch model conversion
# based on Andrea Vedaldi's caffe-importer scripts
#
# --------------------------------------------------------
# mcnPyTorch
# Licensed under The MIT License [see LICENSE.md for details]
# Copyright (C) 2017 Samuel Albanie
# --------------------------------------------------------

import os
import cv2
import math
import ipdb
import sys
import argparse
import importlib
import pathlib
import copy
import torch
from torch import nn
import skeletons.inception
from torch import autograd

import torchvision
import numpy as np
from collections import OrderedDict

sys.path.append(os.path.expanduser('~/coding/libs/pretrained-models.pytorch'))
import pretrainedmodels

# --------------------------------------------------------------------
#                                                     Helpers Functions
# --------------------------------------------------------------------

def set_conversion_kwargs():
    """configure parser for shared keyword arguments
    """
    parser = argparse.ArgumentParser(
	    description='Convert model from PyTorch to MatConvNet.')
    parser.add_argument('pytorch_model',
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
    parser.add_argument('--full-image-size',
                        type=str,
                        nargs='?',
                        default='[256,256]',
                        help='Size of the full image (from which crops are taken)')
    parser.add_argument('--average-image',
                        type=argparse.FileType('rb'),
                        nargs='?',
                        help='Average image')
    parser.add_argument('--average-value',
                        type=str,
                        nargs='?',
                        default=None,
                        help='Average image value')
    parser.add_argument('--remove-dropout',
                        dest='remove_dropout',
                        action='store_true',
                        help='Remove dropout layers')
    parser.add_argument('--remove-loss',
                        dest='remove_loss',
                        action='store_true',
                        help='Remove loss layers')
    parser.add_argument('--model-def',
                        type=str,
                        nargs='?',
                        default=None,
                        help='path to the model definition (a `.py` file)')
    parser.add_argument('--model-weights',
                        type=str,
                        nargs='?',
                        default=None,
                        help='path to the model weights ( a `.pth` file)')
    parser.add_argument('--append-softmax',
                        dest='append_softmax',
                        action='append',
                        default=[],
                        help='Add a softmax layer after the specified layer')
    parser.set_defaults(remove_dropout=False)
    parser.set_defaults(remove_loss=False)
    return parser

# --------------------------------------------------------------------
#                                                     Helpers Functions
# --------------------------------------------------------------------

numerics = (int,float)

mlayerdt = [('name',object),
            ('type',object),
            ('inputs',object),
            ('outputs',object),
            ('params',object),
            ('block',object)]

mparamdt = [('name',object),
            ('value',object)]

minputdt = [('name',object),
            ('size',object)]

def row(x):
    return np.array(x,dtype=float).reshape(1,-1)

def rowcell(x):
    return np.array(x,dtype=object).reshape(1,-1)

def tolist(x):
    """Convert x to a Python list. x can be a Torch size tensor, a list, tuple
    or scalar
    """
    if isinstance(x, torch.Size):
      return [z for z in x]
    elif isinstance(x, (list,tuple)):
      return [z for z in x]
    else:
      return [x]

def pt_tensor_to_array(tensor):
    """Convert a PyTorch Tensor to a numpy array.

    (changes the order of dimensions to [width, height, channels, instance])
    """
    dims = tolist(tensor.size())
    raw = tensor.numpy()
    if len(dims) == 4:
        return raw.transpose((2,3,1,0))
    else:
        return raw.transpose() 

def dictToMatlabStruct(d): 
    if not d: return np.zeros((0,))
    dt = []
    for x in d.keys():
        pair = (x,object)
        if isinstance(d[x], np.ndarray): pair = (x, type(d[x]))
        dt.append(pair)
    y = np.empty((1,), dtype=dt)
    for x in d.keys():
        y[x][0] = d[x]
    return y

class ImTransform(object):
    """Create image transformation

    Args:
        resize (int): input dims
        rgb ((int,int,int)): average RGB values of the dataset
    """
    def __init__(self, imsz, rgb, swap=(2, 0, 1)):
        self.mean_im = rgb
        self.imsz = imsz
        self.swap = swap

    def __call__(self, im):
        im = cv2.resize(np.array(im), self.imsz).astype(np.float32)
        im -= self.mean_im
        im = im.transpose(self.swap)
        return torch.from_numpy(im)

# --------------------------------------------------------------------
#                                                               Models
# --------------------------------------------------------------------

class CanonicalNet(nn.Module):
    """
    """

    def __init__(self, features, classifier, flatten_loc):
        super().__init__()
        self.features = features
        self.classifier = classifier
        self.flatten_loc = flatten_loc

    def forward(self, x):
        # mini = list(self.features.children())[:4]
        # mini_f = torch.nn.modules.Sequential(*mini) ;
        # y = mini_f(x)
        # ipdb.set_trace()
        # mini = list(self.features.children())

        x = self.features(x)
        if self.flatten_loc == 'classifier':
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        elif self.flatten_loc == 'end':
            x = self.classifier(x)
            x = x.view(x.size(0), -1)
        else:
            msg = 'unrecognised flatten_loc: {}'.format(self.flatten_loc)
            raise ValueError(msg)
        return x

def canonical_net(net, name, flatten_loc='classifier', remove_aux=True):
    """
    restructure models to be consistent for easier processing
    """
    is_resnet = isinstance(net, torchvision.models.resnet.ResNet)
    is_densenet = isinstance(net, torchvision.models.densenet.DenseNet)
    is_inception = isinstance(net, torchvision.models.inception.Inception3)
    is_resnext = name in ['resnext_50_32x4d',
                          'resnext_101_32x4d',
                          'resnext_101_64x4d']
    if is_resnet:
        feats_srcs = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1',
                     'layer2', 'layer3', 'layer4', 'avgpool']
        feat_layers = [getattr(net, attr) for attr in feats_srcs]
        features = torch.nn.modules.Sequential(*feat_layers)
        classifier = torch.nn.modules.Sequential(net.fc)
    elif is_densenet:
        feats = net.features
        # insert additional
        # out = F.relu(features, inplace=True)
        # out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        feats_ = [feats, nn.ReLU(inplace=True), nn.AvgPool2d(7)]
        features = nn.modules.Sequential(*feats_)
        classifier = nn.modules.Sequential(net.classifier)
    elif is_inception:
        # skeleton = skeletons.inception.Inception3(aux_logits=False)
        skeleton = skeletons.inception.inception_v3(pretrained=True)
        children = list(skeleton.children())
        tail = children[-1]
        feat_layers = copy.deepcopy(children[:-1])

        # TODO: clean up
        # skeleton.input_space = net.input_space
        # skeleton.input_size = net.input_size
        # skeleton.std = net.std
        # skeleton.mean = net.mean

        # # resort to transferring from skeleton definition
        # # TODO transfer modules

        # # debug
        # model = skeleton
        # path_img = '/users/albanie/coding/libs/pretrained-models.pytorch/data/cat.jpg'
        # with open(path_img, 'rb') as f:
            # with Image.open(f) as img:
                # input_data = img.convert(model.input_space)

        # tf = transforms.Compose([
            # transforms.Scale(round(max(model.input_size)*1.143)),
            # transforms.CenterCrop(max(model.input_size)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=model.mean, std=model.std)
        # ])

        # input_data = tf(input_data)          # 3x400x225 -> 3x299x299
        # input_data = input_data.unsqueeze(0) # 3x299x299 -> 1x3x299x299
        # input = torch.autograd.Variable(input_data)
        # output = model(input) # size(1, 1000)

        if remove_aux: # remove auxiliary classifiers if desired
            feat_layers = [l for l in feat_layers if not
                    isinstance(l, skeletons.inception.InceptionAux)]
        classifier = torch.nn.modules.Sequential(tail)
        features = torch.nn.modules.Sequential(*feat_layers)
    elif is_resnext:
        children = list(net.children())
        feat_layers = copy.deepcopy(children[:-2])
        tail = copy.deepcopy(net)
        while not isinstance(tail, nn.modules.linear.Linear):
            tail = list(tail.children())[-1]
        classifier = torch.nn.modules.Sequential(tail)
        features = torch.nn.modules.Sequential(*feat_layers)
    else:
        raise ValueError('{} unrecognised torchvision model'.format(name))
    return CanonicalNet(features, classifier, flatten_loc)



def load_pytorch_model(name, paths=None):
    flatten_loc = 'classifier' # by default, flattening occurs before classifier
    if paths: # load custom net params and defs
        def_path = pathlib.Path(paths['def'])
        sys.path.insert(0, str(def_path.parent))
        try:
          model_def = importlib.import_module(str(def_path.stem))
        except ImportError:
            msg = 'Failed to import the specified custom model definition ' \
                   + 'module `{}`. Is it on the PYTHONPATH?'
            print(msg.format(def_path.stem))
            raise

    if name == 'alexnet':
        net = torchvision.models.alexnet(pretrained=True)
    elif name == 'vgg11':
        net = torchvision.models.vgg11(pretrained=True)
    elif name == 'vgg13':
        net = torchvision.models.vgg13(pretrained=True)
    elif name == 'vgg13_bn':
        net = torchvision.models.vgg13_bn(pretrained=True)
    elif name == 'vgg16':
        net = torchvision.models.vgg16(pretrained=True)
    elif name == 'vgg19':
        net = torchvision.models.vgg19(pretrained=True)
    elif name == 'inception_v3':
        net = pretrainedmodels.__dict__['inceptionv3'](pretrained='imagenet')
        net = canonical_net(net, name, flatten_loc)
    elif name == 'squeezenet1_0':
        net = torchvision.models.squeezenet1_0(pretrained=True)
        flatten_loc = 'end' # delay flattening
    elif name == 'squeezenet1_1':
        net = torchvision.models.squeezenet1_1(pretrained=True)
        flatten_loc = 'end' # delay flattening
    elif name == 'resnet18':
        net = torchvision.models.resnet18(pretrained=True)
        net = canonical_net(net, name)
    elif name == 'resnet34':
        net = torchvision.models.resnet34(pretrained=True)
        net = canonical_net(net, name)
    elif name == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
        net = canonical_net(net, name)
    elif name == 'resnet101':
        net = torchvision.models.resnet101(pretrained=True)
        net = canonical_net(net, name)
    elif name == 'resnet152':
        net = torchvision.models.resnet152(pretrained=True)
        net = canonical_net(net, name)
    elif name == 'densenet121':
        net = torchvision.models.densenet121(pretrained=True)
        net = canonical_net(net, name)
    elif name == 'densenet161':
        net = torchvision.models.densenet161(pretrained=True)
        net = canonical_net(net, name)
    elif name == 'densenet169':
        net = torchvision.models.densenet169(pretrained=True)
        net = canonical_net(net, name)
    elif name == 'densenet201':
        net = torchvision.models.densenet201(pretrained=True)
        net = canonical_net(net, name)
    elif name == 'resnext_50_32x4d':
        net = model_def.resnext_50_32x4d
        net.load_state_dict(torch.load(paths['weights']))
        net = canonical_net(net, name, flatten_loc=flatten_loc)
    elif name == 'resnext_101_32x4d':
        net = model_def.resnext_101_32x4d
        net.load_state_dict(torch.load(paths['weights']))
        net = canonical_net(net, name, flatten_loc=flatten_loc)
    elif name == 'resnext_101_64x4d':
        net = model_def.resnext_101_64x4d
        net.load_state_dict(torch.load(paths['weights']))
        net = canonical_net(net, name, flatten_loc=flatten_loc)
    else:
        raise ValueError('{} unrecognised torchvision model'.format(name))
    return net, flatten_loc

# --------------------------------------------------------------------
#                                                   Feature Extraction
# --------------------------------------------------------------------

class MapReducePair(object):
    def __init__(self, map, reduce):
        self.map = map
        self.reduce = reduce

    def children(self): return [] # maintain interface

def in_place_replica(x):
    """ Returns a deep copy of an autograd variable's data.
    A number of PyTorch operations are perfomed in-place, which
    makes graph comparison non-trivial. This function enables the
    state of the variable to be stored before it is overwritten"""
    return copy.deepcopy(autograd.Variable(x.data))

def get_custom_feats(net, x):
    children = list(net.children())
    child_warning = 'unexpected number of children'
    if isinstance(net, torchvision.models.squeezenet.Fire):
        assert len(children) == 6, 'unexpected number of children'
        squeeze = torch.nn.Sequential(*children[:2])
        out1x1 = torch.nn.Sequential(*children[:4])
        out3x3 = torch.nn.Sequential(*(children[:2] + children[4:6]))
        feats = []
        for subnet in [squeeze, out1x1, out3x3]:
            raw = torch.nn.Sequential(*list(subnet.children())[:-1])(x)
            feats.append(raw) # without relu
            feats.append(subnet(x))
        feats.append(net(x))
    elif isinstance(net, torchvision.models.densenet._DenseBlock):
        assert len(children) % 2 == 0, 'unexpected number of children'
        feats = []
        current_in = x
        for dense_layer in children:
            # every dense block has the same form
            bn1 = getattr(dense_layer, 'norm.1')(current_in)
            feats.append(in_place_replica(bn1))
            r1 = getattr(dense_layer, 'relu.1')(bn1) ; feats.append(r1)
            c1 = getattr(dense_layer, 'conv.1')(r1) ; feats.append(c1)
            bn2 = getattr(dense_layer, 'norm.2')(c1)
            feats.append(in_place_replica(bn2))
            r2 = getattr(dense_layer, 'relu.2')(bn2) ; feats.append(r2)
            c2 = getattr(dense_layer, 'conv.2')(r2) ; feats.append(c2)
            cat_out = torch.cat([current_in, c2], 1)
            feats.append(cat_out)
            current_in = cat_out
    elif isinstance(net, torchvision.models.densenet._Transition):
        assert len(children) == 4, 'unexpected number of children'
        feats = []
        bn = net.norm(x) ; feats.append(in_place_replica(bn))
        r = net.relu(bn) ; feats.append(r)
        c = net.conv(r) ; feats.append(c)
        p = net.pool(c) ; feats.append(p)
    elif isinstance(net, torchvision.models.resnet.BasicBlock):
        assert len(children) == 5 + bool(net.downsample), child_warning
        feats = []
        residual = x
        c1 = net.conv1(x) ; feats.append(c1)
        bn1 = net.bn1(c1) ; feats.append(in_place_replica(bn1))
        r1 = net.relu(bn1) ; feats.append(r1)
        c2 = net.conv2(r1) ; feats.append(c2)
        out = net.bn2(c2) ; feats.append(in_place_replica(out))
        if net.downsample:
            projection = list(net.downsample.children())[0]
            proj = projection(residual) ; feats.append(in_place_replica(proj))
            residual = net.downsample(residual) # apply sequence
            feats.append(in_place_replica(residual))
        out += residual ; feats.append(in_place_replica(out))
        out = net.relu(out) ; feats.append(out)
    elif isinstance(net, torchvision.models.resnet.Bottleneck):
        assert len(children) == 7 + bool(net.downsample), child_warning
        feats = []
        residual = x
        c1 = net.conv1(x) ; feats.append(c1)
        bn1 = net.bn1(c1) ; feats.append(in_place_replica(bn1))
        r1 = net.relu(bn1) ; feats.append(r1)
        c2 = net.conv2(r1) ; feats.append(c2)
        bn2 = net.bn2(c2) ; feats.append(in_place_replica(bn2))
        r2 = net.relu(bn2) ; feats.append(r2)
        c3 = net.conv3(r2) ; feats.append(c3)
        out = net.bn3(c3) ; feats.append(in_place_replica(out))
        if net.downsample:
            projection = list(net.downsample.children())[0]
            proj = projection(residual) ; feats.append(in_place_replica(proj))
            residual = net.downsample(residual) # apply sequence
            feats.append(in_place_replica(residual))
        out += residual ; feats.append(in_place_replica(out))
        out = net.relu(out) ; feats.append(out)
    elif isinstance(net, skeletons.inception.BasicConv2d):
        # Inception Basic Building Block
        # ------------------------------
        assert len(children) == 3, child_warning
        c1 = net.conv(x) ; feats = [c1]
        bn1 = net.bn(c1) ; feats.append(bn1)
        out = net.relu(bn1) ; feats.append(out)
    elif isinstance(net, skeletons.inception.InceptionA):
        # Inception A style module
        assert len(children) == 8, child_warning
        # for each call to get_feats, we skip over the first retuned feature
        # (since it is a duplicate of the input)
        br1x1 = get_feats(net.branch1x1, x)[1:] ; b1 = br1x1[-1]
        br5x5_1 = get_feats(net.branch5x5_1, x)[1:] ; z = br5x5_1[-1]
        br5x5_2 = get_feats(net.branch5x5_2, z)[1:] ; b5 = br5x5_2[-1]
        br3x3dbl_1 = get_feats(net.branch3x3dbl_1, x)[1:] ; z = br3x3dbl_1[-1]
        br3x3dbl_2 = get_feats(net.branch3x3dbl_2, z)[1:] ; z = br3x3dbl_2[-1]
        br3x3dbl_3 = get_feats(net.branch3x3dbl_3, z)[1:] ; b3 = br3x3dbl_3[-1]
        avg = net.branch_avgpool(x) ; # branch_avgpool is an nn.Module
        brpool = get_feats(net.branch_pool, avg)[1:] ; bp = brpool[-1]
        out = torch.cat([b1, b5, b3, bp], 1)
        feats = (br1x1 + br5x5_1 + br5x5_2 + br3x3dbl_1 +
                    br3x3dbl_2 + br3x3dbl_3 + [avg] + brpool + [out])
    elif isinstance(net, skeletons.inception.InceptionB):
        # Inception B style module
        assert len(children) == 5, child_warning
        br3x3 = get_feats(net.branch3x3, x)[1:] ; b3_1 = br3x3[-1]
        br3x3dbl_1 = get_feats(net.branch3x3dbl_1, x)[1:] ; z = br3x3dbl_1[-1]
        br3x3dbl_2 = get_feats(net.branch3x3dbl_2, z)[1:] ; z = br3x3dbl_2[-1]
        br3x3dbl_3 = get_feats(net.branch3x3dbl_3, z)[1:] ; b3_2 = br3x3dbl_3[-1]
        brpool = net.branch_pool(x)
        out = torch.cat([b3_1, b3_2, brpool], 1)
        feats = (br3x3 + br3x3dbl_1 + br3x3dbl_2 + br3x3dbl_3 + [brpool] + [out])
    elif isinstance(net, skeletons.inception.InceptionC):
        # Inception C style module
        assert len(children) == 11, child_warning
        br1x1 = get_feats(net.branch1x1, x)[1:] ; b1 = br1x1[-1]
        br7x7_1 = get_feats(net.branch7x7_1, x)[1:] ; z = br7x7_1[-1]
        br7x7_2 = get_feats(net.branch7x7_2, z)[1:] ; z = br7x7_2[-1]
        br7x7_3 = get_feats(net.branch7x7_3, z)[1:] ; b7_1 = br7x7_3[-1]
        br7x7dbl_1 = get_feats(net.branch7x7dbl_1, x)[1:] ; z = br7x7dbl_1[-1]
        br7x7dbl_2 = get_feats(net.branch7x7dbl_2, z)[1:] ; z = br7x7dbl_2[-1]
        br7x7dbl_3 = get_feats(net.branch7x7dbl_3, z)[1:] ; z = br7x7dbl_3[-1]
        br7x7dbl_4 = get_feats(net.branch7x7dbl_4, z)[1:] ; z = br7x7dbl_4[-1]
        br7x7dbl_5 = get_feats(net.branch7x7dbl_5, z)[1:] ; b7_2 = br7x7dbl_5[-1]

        avg = net.branch_avgpool(x)
        brpool = get_feats(net.branch_pool, avg)[1:] ; bp = brpool[-1]
        out = torch.cat([b1, b7_1, b7_2, bp], 1)
        feats = (br1x1 + br7x7_1 + br7x7_2 + br7x7_3 + br7x7dbl_1 +
                  br7x7dbl_2 + br7x7dbl_3 + br7x7dbl_4 + br7x7dbl_5 +
                  [avg] + brpool + [out])
    elif isinstance(net, skeletons.inception.InceptionD):
        assert len(children) == 7, child_warning
        br3x3_1 = get_feats(net.branch3x3_1, x)[1:] ; z = br3x3_1[-1]
        br3x3_2 = get_feats(net.branch3x3_2, z)[1:] ; b3 = br3x3_2[-1]
        br7x7x3_1 = get_feats(net.branch7x7x3_1, x)[1:] ; z = br7x7x3_1[-1]
        br7x7x3_2 = get_feats(net.branch7x7x3_2, z)[1:] ; z = br7x7x3_2[-1]
        br7x7x3_3 = get_feats(net.branch7x7x3_3, z)[1:] ; z = br7x7x3_3[-1]
        br7x7x3_4 = get_feats(net.branch7x7x3_4, z)[1:] ; b7 = br7x7x3_4[-1]

        brpool = net.branch_maxpool(x)
        out = torch.cat([b3, b7,  brpool], 1)
        feats = (br3x3_1 + br3x3_2 + br7x7x3_1 + br7x7x3_2 + br7x7x3_3
                      + br7x7x3_4 + [brpool] + [out])
    elif isinstance(net, skeletons.inception.InceptionE):
        assert len(children) == 10, child_warning
        br1x1 = get_feats(net.branch1x1, x)[1:] ; b1 = br1x1[-1]
        br3x3_1 = get_feats(net.branch3x3_1, x)[1:] ; z = br3x3_1[-1]
        br3x3_2a = get_feats(net.branch3x3_2a, z)[1:] ; z1 = br3x3_2a[-1]
        br3x3_2b = get_feats(net.branch3x3_2b, z)[1:] ; z2 = br3x3_2b[-1]
        br3x3 = torch.cat([z1, z2], 1)
        br3x3dbl_1 = get_feats(net.branch3x3dbl_1, x)[1:] ; z = br3x3dbl_1[-1]
        br3x3dbl_2 = get_feats(net.branch3x3dbl_2, z)[1:] ; z = br3x3dbl_2[-1]
        br3x3dbl_3a = get_feats(net.branch3x3dbl_3a, z)[1:] ; z1 = br3x3dbl_3a[-1]
        br3x3dbl_3b = get_feats(net.branch3x3dbl_3b, z)[1:] ; z2 = br3x3dbl_3b[-1]
        br3x3dbl = torch.cat([z1, z2], 1)
        avg = net.branch_avgpool(x) ;
        brpool = get_feats(net.branch_pool, avg)[1:] ; bp = brpool[-1]

        out = torch.cat([b1, br3x3, br3x3dbl, bp], 1)
        feats = (br1x1 + br3x3_1 + br3x3_2a + br3x3_2b + [br3x3]
              + br3x3dbl_1 + br3x3dbl_2 + br3x3dbl_3a + br3x3dbl_3b + [br3x3dbl]
              + [avg] + brpool + [out])

    elif isinstance(net, MapReducePair):
        _map, _reduce = net.map.lambda_func, net.reduce.lambda_func
        if _map(1) != 1: raise ValueError('only identity map supported')
        if _reduce(1,1) != 2: raise ValueError('only summation reduce supported')
        feats = []
        id_map = x
        base = nn.Sequential(*net.map[0])
        base_feats = get_feats(base, id_map, []) ; feats.extend(base_feats[1:])
        if not hasattr(net.map[1], 'lambda_func'): # projection exists
            projection = nn.Sequential(net.map[1])
            proj = get_feats(projection, id_map, []) ; feats.extend(proj[1:])
            id_map = proj[-1]
        else:
            assert net.map[1].lambda_func(1) == 1, 'non projection should be identity'
        out = id_map + base_feats[-1] ; feats.append(out)
    else:
        raise ValueError('{} unrecognised custom module'.format(type(net)))
    return feats

def has_lambda_child(module):
    has_children = len(list(module.children())) > 0
    return  has_children and is_lambda_map(list(module.children())[0])

def is_lambda_map(mod):
    """check for the presence of the LambdaMap block used by the
    torch -> pytorch converter
    """
    return 'LambdaMap' in mod.__repr__().split('\n')[0]

def is_lambda_reduce(mod):
    """check for the presence of the LambdReduce block used by the
    torch -> pytorch converter
    """
    return 'LambdaReduce' in mod.__repr__().split('\n')[0]

def is_plain_lambda(mod):
    """check for the presence of the LambdReduce block used by the
    torch -> pytorch converter
    """
    return 'Lambda ' in mod.__repr__().split('\n')[0]

def get_feats(net, x, feats=[]):
   children = list(net.children())
   if len(children) == 0:
       return [net(x)]
   head, tail = children[:-1], children[-1]

   # handle chunking for models imported from torch (lua)
   # use a string check hack to avoid adding a module import
   # path dependency
   if is_lambda_reduce(tail):
       assert is_lambda_map(head[-1]), 'invalid map reduce pair'
       tail = MapReducePair(head[-1], tail)
       head = head[:-1] # adjust head
   elif isinstance(tail, (torchvision.models.densenet._DenseBlock,
                          torchvision.models.densenet._Transition)):
       pass # handel in the custom block
   else: # standard model structure
       while isinstance(tail, torch.nn.Sequential):
           children = list(tail.children())
           head_, tail = children[:-1], children[-1]
           head = head + head_

   trunk = torch.nn.Sequential(*head)
   trunk_feats = get_feats(trunk, x, feats)
   if type(tail) in [torchvision.models.squeezenet.Fire,
                     torchvision.models.resnet.BasicBlock,
                     torchvision.models.resnet.Bottleneck,
                     torchvision.models.densenet._DenseBlock,
                     torchvision.models.densenet._Transition,
                     skeletons.inception.BasicConv2d,
                     skeletons.inception.InceptionA,
                     skeletons.inception.InceptionB,
                     skeletons.inception.InceptionC,
                     skeletons.inception.InceptionD,
                     skeletons.inception.InceptionE,
                     MapReducePair]:
       tail_feats = get_custom_feats(tail, trunk_feats[-1])
   else:
       tail_feats = [net(x)]
   sizes = trunk_feats + tail_feats
   return sizes

def compute_intermediate_feats(net, x, flatten_loc):
    feature_feats = get_feats(net.features, x, feats=[])
    x = feature_feats[-1]
    if flatten_loc == 'classifier':
        x = x.view(x.size(0), -1)
        cls_feats = get_feats(net.classifier, x, feats=[])
    elif flatten_loc == 'end':
        cls_feats = get_feats(net.classifier, x, feats=[])
        x = x.view(x.size(0), -1)
    else:
        raise ValueError('flatten_loc: {} not recognised'.format(flatten_loc))
    #offset = len(list(net.classifier.children())) > 0
    offset = len(cls_feats) > 2
    return feature_feats + cls_feats[offset:] # drop duplicate feature if needed

# --------------------------------------------------------------------
#                                              PyTorch Aggregator class
# --------------------------------------------------------------------

class TFTensor(object):
    def __init__(self, name):
        self.name = name
        self.shape = None
        self.value = np.zeros(shape=(0,0), dtype='float32')
        self.bgrInput = False
        self.transposable = True # first two dimensions are spatial

    def transpose(self):
        if self.shape: self.shape = [self.shape[k] for k in [1,0,2,3]]

    def toMatlab(self):
        mparam = np.empty(shape=[1,], dtype=mparamdt)
        mparam['name'][0] = self.name
        mparam['value'][0] = self.value
        return mparam

    def hasValue(self):
        return reduce(mul, self.value.shape, 1) > 0

class PTModel(object):
    def __init__(self):
        self.layers = OrderedDict()
        self.vars = OrderedDict()
        self.params = OrderedDict()

    def add_layer(self, layer):
        ename = layer.name
        while ename in self.layers:
            ipdb.set_trace()
            ename = ename + 'x'
        if layer.name != ename:
            print('Warning: a layer with name {} already found' \
                  ', using {} instead'.format(layer.name, ename))
            layer.name = ename
        for v in layer.inputs:  self.add_var(v)
        for v in layer.outputs: self.add_var(v)
        for p in layer.params: self.add_param(p)
        self.layers[layer.name] = layer

    def add_var(self, name):
        if name not in self.vars:
            self.vars[name] = TFTensor(name)

    def add_param(self, name):
        if name not in self.params:
            self.params[name] = TFTensor(name)

    def renameLayer(self, old, new):
        self.layers[old].name = new
        # reinsert layer with new name -- this mess is to preserve the order
        layers = OrderedDict([(new,v) if k==old else (k,v)
                              for k,v in self.layers.items()])
        self.layers = layers

# --------------------------------------------------------------------
#                                                         Basic Layers
# --------------------------------------------------------------------

class PTLayer(object):
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.params = []
        self.model = None

    def reshape(self, model):
        pass

    def display(self):
        print("Layer \'{}\'".format(self.name))
        print("  +- type: %s" % (self.__class__.__name__))
        print("  +- inputs: %s" % (self.inputs,))
        print("  +- outputs: %s" % (self.outputs,))
        print("  +- params: %s" % (self.params,))

    def transpose(self, model):
        pass

    def setTensor(self, model, all_params):
        print(self.name) # useful sanity check
        ipdb.set_trace()
        assert(False)

    def toMatlab(self):
        mlayer = np.empty(shape=[1,],dtype=mlayerdt)
        mlayer['name'][0] = self.name
        mlayer['type'][0] = None
        mlayer['inputs'][0] = rowcell(self.inputs)
        mlayer['outputs'][0] = rowcell(self.outputs)
        mlayer['params'][0] = rowcell(self.params)
        mlayer['block'][0] = dictToMatlabStruct({})
        return mlayer

class PTElementWise(PTLayer):
    def reshape(self, model):
        for i in range(len(self.inputs)):
            model.vars[self.outputs[i]].shape = \
                model.vars[self.inputs[i]].shape

class PTReLU(PTElementWise):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)

    def setTensor(self, model, all_params):
        pass

    def toMatlab(self):
        mlayer = super().toMatlab()
        mlayer['type'][0] = u'dagnn.ReLU'
        mlayer['block'][0] = dictToMatlabStruct(
            {'leak': float(0.0) })
        # todo: leak factor
        return mlayer

class PTConv(PTLayer):
    def __init__(self, name, inputs, outputs,
                 num_output,
                 bias_term,
                 pad,
                 kernel_size,
                 stride,
                 dilation,
                 group, 
         filter_depth=None):

        super().__init__(name, inputs, outputs)

        if len(kernel_size) == 1 : kernel_size = kernel_size * 2
        if len(stride) == 1 : stride = stride * 2
        if len(dilation) == 1 : dilation = dilation * 2
        if len(pad) == 1 : pad = pad * 4
        elif len(pad) == 2 : pad = [pad[0], pad[0], pad[1], pad[1]]

        self.num_output = num_output
        self.bias_term = bias_term
        self.pad = pad
        self.kernel_size = tolist(kernel_size)
        self.stride = stride
        self.dilation = dilation
        self.group = group

        self.params = [name + '_filter']
        if bias_term: self.params.append(name + '_bias')
        self.filter_depth = filter_depth

    def display(self):
        super(PTConv, self).display()
        print("  +- filter dimension:", self.filter_depth)
        print("  c- num_output (num filters): %s" % self.num_output)
        print("  c- bias_term: %s" % self.bias_term)
        print("  c- pad: %s" % (self.pad,))
        print("  c- kernel_size: %s" % self.kernel_size)
        print("  c- stride: %s" % (self.stride,))
        print("  c- dilation: %s" % (self.dilation,))
        print("  c- group: %s" % (self.group,))

    def reshape(self, model):
        varin = model.vars[self.inputs[0]]
        varout = model.vars[self.outputs[0]]
        if not varin.shape: return
        varout.shape = getFilterOutputSize(varin.shape[0:2],
                                           self.kernel_size,
                                           self.stride,
                                           self.pad) \
                                           + [self.num_output, varin.shape[3]]
        self.filter_depth = varin.shape[2] / self.group

    def setTensor(self, model, all_params):
        filters = pt_tensor_to_array(all_params['{}_weight'.format(self.name)])
        while len(filters.shape) < 4 : # handle torch FC tensors
            filters = np.expand_dims(filters,axis=0)
        assert(filters.shape[0] == self.kernel_size[0])
        assert(filters.shape[1] == self.kernel_size[1])
        assert(filters.shape[3] == self.num_output)
        self.filter_depth = filters.shape[2]
        tensors = [filters]
        if self.bias_term:
            bias = pt_tensor_to_array(all_params['{}_bias'.format(self.name)])
            tensors.append(bias)
        for ii, tensor in enumerate(tensors):
            model.params[self.params[ii]].value = tensor
            model.params[self.params[ii]].shape = tensor.shape

    def toMatlab(self):
        size = self.kernel_size + [self.filter_depth, self.num_output]
        mlayer = super().toMatlab()
        mlayer['type'][0] = u'dagnn.Conv'
        mlayer['block'][0] = dictToMatlabStruct(
            {'hasBias': self.bias_term,
             'size': row(size),
             'pad': row(self.pad),
             'stride': row(self.stride),
             'dilate': row(self.dilation)})
        return mlayer

# --------------------------------------------------------------------
#                                                            BatchNorm
# --------------------------------------------------------------------

class PTBatchNorm(PTLayer):
    def __init__(self, name, inputs, outputs, use_global_stats,
                                             momentum, eps):
        super().__init__(name, inputs, outputs)

        self.use_global_stats = use_global_stats
        self.momentum = momentum
        self.eps = eps

        self.params = [name + u'_mult',
                       name + u'_bias',
                       name + u'_moments']

    def display(self):
        super().display()
        print('  c- use_global_stats: %s'.format(self.use_global_stats,))
        print('  c- momentum: %s'.format(self.momentum,))
        print('  c- eps: %s'.format(self.eps))

    def setTensor(self, model, all_params):
        # Note: PyTorch stores variances rather than sigmas
        gamma = pt_tensor_to_array(all_params['{}_weight'.format(self.name)])
        beta = pt_tensor_to_array(all_params['{}_bias'.format(self.name)])
        mean = pt_tensor_to_array(all_params['{}_running_mean'.format(self.name)])
        var = pt_tensor_to_array(all_params['{}_running_var'.format(self.name)])

        # note: PyTorch uses a slightly different formula for batch norm than
        # matconvnet at *test* time:
        # pytorch:
        #    y = ( x - mean(x)) / (sqrt(var(x)) + eps) * gamma + beta
        # mcn:
        #    y = ( x - mean(x)) / sqrt(var(x)) * gamma + beta
        sigma = np.sqrt(var + self.eps)
        moments = np.vstack((mean, sigma)).T
        tensors = [gamma, beta, moments]
        for ii, tensor in enumerate(tensors):
            model.params[self.params[ii]].value = tensor
            model.params[self.params[ii]].shape = tensor.shape

    def toMatlab(self):
        mlayer = super().toMatlab()
        mlayer['type'][0] = u'dagnn.BatchNorm'
        mlayer['block'][0] = dictToMatlabStruct(
            {'epsilon': self.eps})
        return mlayer


class PTPooling(PTLayer):
    def __init__(self, name, inputs, outputs, method, pad, kernel_size,
                 stride, ceil_mode, sizes, dilation=[1,1]):

        super().__init__(name, inputs, outputs)

        if isinstance(kernel_size, numerics): kernel_size = [kernel_size,]
        if isinstance(stride, numerics): stride = [stride,]
        if isinstance(pad, numerics): pad = [pad,]

        if len(kernel_size) == 1 : kernel_size = kernel_size * 2
        if len(stride) == 1 : stride = stride * 2
        if len(pad) == 1 : pad = pad * 4
        elif len(pad) == 2 : pad = [pad[0], pad[0], pad[1], pad[1]]

        if ceil_mode:
            # if ceil mode is engaged, we need to handle padding more
            # carefully pad to compensate for discrepancy
            in_sz, out_sz = sizes
            h_out = ((in_sz[2] + 2*pad[0] - dilation[0]*(kernel_size[0] - 1) -1)
                                                            / stride[0] + 1)
            w_out = ((in_sz[3] + 2*pad[1] - dilation[1]*(kernel_size[1] - 1) -1)
                                                             / stride[1] + 1)
            if math.ceil(h_out) > math.floor(h_out): pad[1] += 1
            if math.ceil(w_out) > math.floor(w_out): pad[3] += 1

        self.method = method
        self.pad = tolist(pad)
        self.kernel_size = tolist(kernel_size)
        self.stride = tolist(stride)
        self.pad_corrected = None

    def setTensor(self, model, all_params):
        pass

    def display(self):
        super(PTPooling, self).display()
        print('  +- pad_corrected: %s' % (self.pad_corrected,))
        print('  c- method: ', self.method)
        print('  c- pad: %s' % (self.pad,))
        print('  c- kernel_size: %s' % (self.kernel_size,))
        print('  c- stride: %s' % (self.stride,))

    def toMatlab(self):
        mlayer = super().toMatlab()
        self.pad_corrected = self.pad # TODO(sam): This may require a fix
        mlayer['type'][0] = u'dagnn.Pooling'
        mlayer['block'][0] = dictToMatlabStruct(
            {'method': self.method,
             'poolSize': row(self.kernel_size),
             'stride': row(self.stride),
             'pad': row(self.pad_corrected)})
        return mlayer

class PTConcat(PTLayer):
    def __init__(self, name, inputs, outputs, concatDim):
        super(PTConcat, self).__init__(name, inputs, outputs)
        self.concatDim = concatDim

    def display(self):
        super().display()
        print('  Concat Dim: ', self.concatDim)

    def setTensor(self, model, all_params):
        pass

    def toMatlab(self):
        mlayer = super(PTConcat, self).toMatlab()
        mlayer['type'][0] = u'dagnn.Concat'
        mlayer['block'][0] = dictToMatlabStruct({'dim': float(self.concatDim)})
        return mlayer


class PTSum(PTElementWise):
    def __init__(self, name, inputs, outputs,):
        super().__init__(name, inputs, outputs)

    def toMatlab(self):
        mlayer = super().toMatlab()
        mlayer['type'][0] = u'dagnn.Sum'
        return mlayer

    def setTensor(self, model, all_params):
        pass

    def display(self):
        super().display()
        print('  c- operation: ', self.operation)
        print('  c- coeff: {}'.format(self.coeff))
        print('  c- stable_prod_grad: {}'.format(self.stable_prod_grad))

class PTDropout(PTElementWise):
    def __init__(self, name, inputs, outputs, ratio):
        super().__init__(name, inputs, outputs)
        self.ratio = ratio

    def toMatlab(self):
        mlayer = super(PTDropout, self).toMatlab()
        mlayer['type'][0] = u'dagnn.DropOut'
        mlayer['block'][0] = dictToMatlabStruct({'rate': float(self.ratio)})
        return mlayer

    def setTensor(self, model, all_params):
        pass

    def display(self):
        super(PTDropout, self).display()

class PTFlatten(PTLayer):
    def __init__(self, name, inputs, outputs, axis):
        super().__init__(name, inputs, outputs)
        self.axis = 3

    def display(self):
        super(PTFlatten, self).display()
        print('  c- axis: {}'.format(self.axis))

    def setTensor(self, model, all_params):
        pass

    def toMatlab(self):
        mlayer = super().toMatlab()
        mlayer['type'][0] = u'dagnn.Flatten'
        mlayer['block'][0] = dictToMatlabStruct(
            {'axis': self.axis})
        return mlayer

class PTPermute(PTLayer):
    def __init__(self, name, inputs, outputs, order):
        super().__init__(name, inputs, outputs)
        self.order = order

    def display(self):
        super().display()
        print('  c- order %s'.format(self.order))

    def reshape(self, model):
        varin = model.vars[self.inputs[0]]
        varout = model.vars[self.outputs[0]]
        return
        if not varin.shape: return
        varout.shape = getFilterOutputSize(varin.shape[0:2],
                                           self.kernel_size,
                                           self.stride,
                                           self.pad) \
                                           + [self.num_output, varin.shape[3]]
        self.filter_depth = varin.shape[2] / self.group

    def setTensor(self, model, all_params):
        pass

    def toMatlab(self):
        mlayer = super().toMatlab()
        mlayer['type'][0] = u'dagnn.Permute'
        mlayer['block'][0] = dictToMatlabStruct(
            {'order': self.order})
        return mlayer
