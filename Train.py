def Make_Forms(Tuple):
  x_2, x_1, y_2, y_1 = Tuple
  T_x = (x_1 + x_2) * 0.5
  T_y = (y_1 + y_2) * 0.5
  T_w = (x_2 - x_1)
  T_h = (y_2 - y_1)

  return (T_x, T_y, T_w, T_h)

def To_Device(Tuple):
  t1, t2, t3, t4 = Tuple
  t1 = t1.to(device)
  t2 = t2.to(device)
  t3 = t3.to(device)
  t4 = t4.to(device)
  
  return (t1, t2, t3, t4)

from tqdm import tqdm
def Train_Epoch(dataloader, model, optimizer, device):
  model.train()
  Book = tqdm(dataloader, total = len(dataloader))
  total_loss = 0.0
  for bi, Dictionary in enumerate(Book):
    Model_INPUT = Dictionary['Model_INPUT']
    CT_Boxes = Dictionary['CT_Boxes']
    for i in range(len(Model_INPUT)):
      input_ids = Model_INPUT[i][0].to(device)
      cls_label = Model_INPUT[i][1].to(device)
      loc_label = torch.Tensor(Model_INPUT[i][2]).unsqueeze(0).to(device)

      model.zero_grad()
      Logits = model(input_ids) 
      
      (d_x, d_y, d_w, d_h) = Logits[1].squeeze(0)
      (p_x, p_y, p_w, p_h) = Make_Forms(CT_Boxes[i])
      (p_x, p_y, p_w, p_h) = To_Device((p_x, p_y, p_w, p_h))
      T_u = (p_w * d_x + p_x, p_h * d_y + p_y, p_w * torch.exp(d_w), p_h * torch.exp(d_h))
      T_u = torch.Tensor(T_u).to(device)

      Loss = Fast_RCNN_Loss(Logits[0], cls_label, T_u, loc_label)
      Loss.backward(retain_graph = True)

      optimizer.step()
      optimizer.zero_grad()
      total_loss += Loss.item()
  Average_Train_Loss = total_loss / len(dataloader)
  print(" Average Train Loss: {0:.2f}".format(Average_Train_Loss))

def FIT(dataloader, model, Epochs, Learning_Rate):
  optimizer = torch.optim.AdamW(model.parameters(), lr = Learning_Rate)
  for i in range(Epochs):
    print(f"EPOCHS:{i+1}")
    print('TRAIN')
    Train_Epoch(dataloader, model, optimizer, device)
    torch.save(model, '/content/gdrive/My Drive/' + f'Fast_RCNN_Model')

FIT(Train_dataloader, Fast_RCNN_Model, Epochs = 3, Learning_Rate = 0.002)
