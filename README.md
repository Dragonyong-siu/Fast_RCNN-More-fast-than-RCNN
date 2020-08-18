# Fast_RCNN-More-fast-than-RCNN

 Base
  Data : PASCAL_VOC_2012

 0.Fast_RCNN_VGG16 : Feature_Map Extraction

 1.Fast_RCNN_Dataset

 2.RoI(Regions of Interests) Pooling

 3.Fast_RCNN_Main_Model : Fully_Conection_Model

 4.Fast_RCNN_Loss : L_cls(p, u) + lamda * [u >= 1] * L_loc(t_u, v)

 5.Train(Classification + Bbox_Regression)

 #didn't code the nms part, because it was all part of other work, faster rcnn. you can check them in code of faster rcnn.
