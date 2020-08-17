class Fully_Conection_Model(nn.Module):
  def __init__(self):
    super(Fully_Conection_Model, self).__init__()
    self.First_Sibling_Layer = nn.Sequential(
        OrderedDict([('Dropout_1', nn.Dropout()),
                     ('Linear_1', nn.Linear(512 * 7 * 7, 4096)),
                     ('ReLU_1', nn.ReLU(inplace = True)),
    
                     ('Dropout_2', nn.Dropout()),
                     ('Linear_2', nn.Linear(4096, 4096)),
                     ('ReLU_2', nn.ReLU(inplace = True)),
    
                     ('Linear_3', nn.Linear(4096, 21))]))
    
    self.Second_Sibling_Layer = nn.Sequential(
        OrderedDict([('Dropout_1', nn.Dropout()),
                     ('Linear_1', nn.Linear(512 * 7 * 7, 4096)),
                     ('ReLU_1', nn.ReLU(inplace = True)),
    
                     ('Dropout_2', nn.Dropout()),
                     ('Linear_2', nn.Linear(4096, 4096)),
                     ('ReLU_2', nn.ReLU(inplace = True)),
    
                     ('Linear_3', nn.Linear(4096, 4))]))
    
  def forward(self, Model_INPUT):
    Flatten_INPUT = Model_INPUT.view(-1, 512 * 7 * 7)
    Classification_Output = self.First_Sibling_Layer(Flatten_INPUT)
    Bbox_Regression_Output = self.Second_Sibling_Layer(Flatten_INPUT)
    return Classification_Output, Bbox_Regression_Output

Fast_RCNN_Model = Fully_Conection_Model().to(device) 
