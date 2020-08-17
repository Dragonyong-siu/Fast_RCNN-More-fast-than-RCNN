import numpy as np
import selectivesearch
class Fast_RCNN_Dataset(torch.utils.data.Dataset):
  def __init__(self, Data):
    self.Data = Data

  def __len__(self):
    return len(self.Data)

  def __getitem__(self, index):
    Dictionary = {}
    PIL_Image = self.Data[index][0]
    Annotation = self.Data[index][1]['annotation']
     
    # Full_Image_Array (= Model's Input)
    # Ground_Truths
    # Names 
    Full_Image_Array = np.asarray(PIL_Image)
    Objects = Annotation['object']
    Size_Tuple = Annotation['size']
    Original_Size = (int(Size_Tuple['depth']),
                     int(Size_Tuple['height']),
                     int(Size_Tuple['width']))

    Ground_Truths = []
    Names = []
    for i in range(len(Objects)):
      bndbox = Objects[i]['bndbox']
      xmax = int(bndbox['xmax'])
      xmin = int(bndbox['xmin'])
      ymax = int(bndbox['ymax'])
      ymin = int(bndbox['ymin'])
      ground_truth = (xmax, xmin, ymax, ymin)
      name = Objects[i]['name']
      Ground_Truths.append(ground_truth)
      Names.append(name)
    
    # Feature_Map
    Input_Tensor = torch.Tensor(Full_Image_Array).to(device)
    Input_View = Input_Tensor.view(-1, 3, Original_Size[1], Original_Size[2])
    Feature_Map = Fast_RCNN_Extractor(Input_View)
    
    # Regions_of_Interests
    _, Regions = selectivesearch.selective_search(Full_Image_Array, 
                                                  scale = 10, 
                                                  min_size = 1000) 
    # Feature_Map_length
    Green_box = (125, 255, 51)
    Regions_of_Interests = []
    _, Orig_Height, Orig_Width = Original_Size
    _, Conv_Height, Conv_Width = Feature_Map.squeeze(0).shape
    for candidate in Regions:
      left = (candidate['rect'][0] * Conv_Width) / Orig_Width
      bottom = (candidate['rect'][1] * Conv_Height) / Orig_Height
      right = left + (candidate['rect'][2] * Conv_Width) / Orig_Width
      top = bottom + (candidate['rect'][3] * Conv_Height) / Orig_Height
      roi = (int(right), int(left), int(top), int(bottom))
      if int(right) > int(left) and int(top) > int(bottom):
        Regions_of_Interests.append(roi)
    
    # Model_INPUT
    # CT_BOX
    Model_Ids = RoI_Pooling(Feature_Map, Regions_of_Interests, (7, 7))
    T_INPUT = []
    F_INPUT = []
    T_Boxes = []
    F_Boxes = []
    for i in range(len(Ground_Truths)):
      Xmax, Xmin, Ymax, Ymin = Ground_Truths[i]
      Resized_GT = (int((Xmax * Conv_Width) / Orig_Width),
                    int((Xmin * Conv_Width) / Orig_Width),
                    int((Ymax * Conv_Height) / Orig_Height),
                    int((Ymin * Conv_Height) / Orig_Height))
      for bi, RoI in enumerate(Regions_of_Interests):
        if Compute_IoU(RoI, Resized_GT) > 0.3:
          model_label = Encoding(Names[i])
          T_INPUT.append((Model_Ids[bi], model_label, torch.Tensor(Resized_GT)))
          T_Boxes.append(RoI)

        elif Compute_IoU(RoI, Resized_GT) >= 0.0 and Compute_IoU(RoI, Resized_GT) < 0.2:
          model_label = Encoding('background')
          F_INPUT.append((Model_Ids[bi], model_label, torch.Tensor(Resized_GT)))
          F_Boxes.append(RoI)
    # Make num to 40
    F_num = 40 - len(T_Boxes)
    F_Boxes = F_Boxes[:F_num]
    CT_Boxes = T_Boxes + F_Boxes

    F_num = 40 - len(T_INPUT)
    F_INPUT = F_INPUT[:F_num]
    Model_INPUT = T_INPUT + F_INPUT
    
    #Dictionary['Full_Image_Array'] = Full_Image_Array
    #Dictionary['Ground_Truths'] = Ground_Truths
    #Dictionary['Names'] = Names
    #Dictionary['Feature_Map'] = Feature_Map
    #Dictionary['Feature_Map_length'] = (Conv_Height, Conv_Width)
    #Dictionary['Regions_of_Interests'] = Regions_of_Interests
    Dictionary['Model_INPUT'] = Model_INPUT
    Dictionary['CT_Boxes'] = CT_Boxes
    
    return Dictionary

# 2.RoI(Regions of Interests) Pooling
import torch.nn.functional as F
def RoI_Pooling(feature_map, rois, size):
  Pooled_RoI = []
  rois_num = len(rois)
  for i in range(rois_num):
    roi = rois[i]
    Right, Left, Top, Bottom = roi
    Cut_Feature_Map = feature_map[:, :, Bottom:Top, Left:Right]
    Fixed_Feature_Map = F.adaptive_max_pool2d(Cut_Feature_Map, size)
    Pooled_RoI.append(Fixed_Feature_Map)

  return torch.cat(Pooled_RoI)

def Compute_IoU(CD_box, GT_box):
  X_1 = np.maximum(CD_box[1], GT_box[1])
  X_2 = np.minimum(CD_box[0], GT_box[0])
  Y_1 = np.maximum(CD_box[3], GT_box[3])
  Y_2 = np.minimum(CD_box[2], GT_box[2])

  Intersection = np.maximum(X_2 - X_1, 0) * np.maximum(Y_2 - Y_1, 0)
  CD_area = (CD_box[0] - CD_box[1]) * (CD_box[2] - CD_box[3])
  GT_area = (GT_box[0] - GT_box[1]) * (GT_box[2] - GT_box[3])
  Union = CD_area + GT_area - Intersection
  IoU = Intersection / Union

  return IoU

from torch.utils.data import DataLoader
Train_dataloader = DataLoader(Fast_RCNN_Dataset(Train_data),
                              batch_size = 1,
                              shuffle = True,
                              drop_last = True)
