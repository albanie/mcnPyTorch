import io
import requests
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
IMG_URL = 'https://s3.amazonaws.com/outcome-blog/wp-content/uploads/2017/02/25192225/cat.jpg'


net = models.vgg16(pretrained=True)
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
response = requests.get(IMG_URL)
img_pil = Image.open(io.BytesIO(response.content))

img_tensor = preprocess(img_pil)
img_tensor.unsqueeze_(0)
img_variable = Variable(img_tensor)
fc_out = net.eval()(img_variable)
labels = {int(key):value for (key, value)
          in requests.get(LABELS_URL).json().items()}
print(labels[fc_out.data.numpy().argmax()])

probs = torch.nn.Softmax()(fc_out).data
print(probs.max())

