import torchvision
from torchvision import transforms
Train_data = torchvision.datasets.VOCDetection(root = './VOCDetection',
                                               year = '2012',
                                               image_set = 'train',
                                               download = True,
                                               transform = None,
                                               target_transform = None,
                                               transforms = None)

#Validation_data = torchvision.datasets.VOCDetection(root = './data',
#                                                    year = '2012',
#                                                   image_set = 'val',
#                                                   download = True,
#                                                   transform = None,
#                                                   target_transform = None,
#                                                   transforms = None)
