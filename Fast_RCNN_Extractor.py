device = 'cuda'
import torch.nn as nn
from collections import OrderedDict
class Fast_RCNN_VGG16(nn.Module):
  def __init__(self):
    super(Fast_RCNN_VGG16, self).__init__()
    self.Convolution_Layer = nn.Sequential(
        OrderedDict([('Conv_1', nn.Conv2d(3, 64, 3, stride = 1, padding = 1)),
                     ('ReLU_1', nn.ReLU(inplace = True)),
                     ('Conv_2', nn.Conv2d(64, 64, 3, stride = 1, padding = 1)),
                     ('ReLU_2', nn.ReLU(inplace = True)),
                     ('MaxPool_1', nn.MaxPool2d(2, stride = 2)),

                     ('Conv_3', nn.Conv2d(64, 128, 3, stride = 1, padding = 1)),
                     ('ReLU_3', nn.ReLU(inplace = True)),
                     ('Conv_4', nn.Conv2d(128, 128, 3, stride = 1, padding = 1)),
                     ('ReLU_4', nn.ReLU(inplace = True)),
                     ('MaxPool_2', nn.MaxPool2d(2, stride = 2)),

                     ('Conv_5', nn.Conv2d(128, 256, 3, stride = 1, padding = 1)),
                     ('ReLU_5', nn.ReLU(inplace = True)),
                     ('Conv_6', nn.Conv2d(256, 256, 3, stride = 1, padding = 1)),
                     ('ReLU_6', nn.ReLU(inplace = True)),
                     ('Conv_7', nn.Conv2d(256, 256, 3, stride = 1, padding = 1)),
                     ('ReLU_7', nn.ReLU(inplace = True)),
                     ('MaxPool_3', nn.MaxPool2d(2, stride = 2)),

                     ('Conv_8', nn.Conv2d(256, 512, 3, stride = 1, padding = 1)),
                     ('ReLU_8', nn.ReLU(inplace = True)),
                     ('Conv_9', nn.Conv2d(512, 512, 3, stride = 1, padding = 1)),
                     ('ReLU_9', nn.ReLU(inplace = True)),
                     ('Conv_10', nn.Conv2d(512, 512, 3, stride = 1, padding = 1)),
                     ('ReLU_10', nn.ReLU(inplace = True)),
                     ('Maxpool_4', nn.MaxPool2d(2, stride = 2)),

                     ('Conv_11', nn.Conv2d(512, 512, 3, stride = 1, padding = 1)),
                     ('ReLU_11', nn.ReLU(inplace = True)),
                     ('Conv_12', nn.Conv2d(512, 512, 3, stride = 1, padding = 1)),
                     ('ReLU_12', nn.ReLU (inplace = True)),
                     ('Conv_13', nn.Conv2d(512, 512, 3, stride = 1, padding = 1)),
                     ('ReLU_13', nn.ReLU(inplace = True)),
                     ('MaxPool_5', nn.MaxPool2d(2, stride = 2))]))
  
  def forward(self, Image):
    Conv_Image = self.Convolution_Layer(Image)
    return Conv_Image

Fast_RCNN_Extractor = Fast_RCNN_VGG16().to(device)
