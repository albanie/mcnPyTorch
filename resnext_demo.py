import io,os
import requests
from collections import OrderedDict
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
import sys
from torch import nn
import scipy.io
from torch.autograd import Variable
sys.path.insert(0, 'python')

from IPython import get_ipython
ipython = get_ipython()
ipython.magic('load_ext autoreload')
ipython.magic('autoreload 2')

import pytorch_utils as pl

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
IMG_URL = 'https://s3.amazonaws.com/outcome-blog/wp-content/uploads/2017/02/25192225/cat.jpg'

# vision models
# model = 'resnet50'
# net,_ = pl.load_pytorch_model(model)

model = 'resnext_101_64x4d'
model = 'resnext_101_64x4d'
paths = {}
paths['def'] = os.path.expanduser('~/.torch/models/{}.pth'.format(model))
paths['weights'] = os.path.expanduser('~/.torch/models/{}.pth'.format(model))
net,_ = pl.load_pytorch_model(model,paths)


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Scale(224),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])
# response = requests.get(IMG_URL)
# img_pil = Image.open(io.BytesIO(response.content))
img_pil = Image.open('cat.jpg')

img_tensor = preprocess(img_pil)
img_tensor.unsqueeze_(0)
img_variable = Variable(img_tensor)

py_feats_tensors = pl.compute_intermediate_feats(net.eval(), img_variable, 'classifier')
py_feats = [np.squeeze(x.data.numpy()) for x in py_feats_tensors]

py_store = {'x{}'.format(i):x for i, x in enumerate(py_feats)}
scipy.io.savemat('{}-cat.mat'.format(model), py_store, oned_as='column')

# rename keys to make compatible (duplicates params)
params = net.state_dict()
tmp = OrderedDict()
for key in params:
    new_name = key.replace('.', '_')
    tmp[new_name] = params[key].numpy()
params = tmp 

scipy.io.savemat('{}-params.mat'.format(model), params, oned_as='column')

# if 1:
    # xx = img_variable.data.numpy() 
    # x = {} ; x['im'] = xx

fc_out = net.eval()(img_variable)
labels = {int(key):value for (key, value)
          in requests.get(LABELS_URL).json().items()}
print(labels[fc_out.data.numpy().argmax()])

probs = nn.Softmax()(fc_out).data
print(probs.max())
