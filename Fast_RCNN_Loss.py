def Fast_RCNN_Loss(P, U, T_u, V, lamda = 10, Ncls = 1, Nreg = 1 * 9):
  CLS_Function = nn.CrossEntropyLoss()
  LOC_Function = nn.SmoothL1Loss()
  CLS_Loss = CLS_Function(P, U)
  LOC_Loss = LOC_Function(T_u, V)
  if U == 20:
    IBI = 0
  else:
    IBI = 1
  Loss = (CLS_Loss / Ncls) + (lamda * IBI * LOC_Loss / Nreg)
  return Loss
