import torch
import torch.nn.functional as F
from torch import nn

class MixUpLoss(nn.Module):
    def __init__(self):
        super(MixUpLoss,self).__init__()
    def forward(self,output_x,trg_x,output_u,trg_u):
        """
        loss function:eq.(2)-(4)
        :param output_x: mixuped x output:[N,10]
        :param trg_x: mixuped targ:[N,10]
        :param output_u: mixuped u output [2*N,10]
        :param trg_u: mixuped target u output:[2*N,10]
        :return: Lx,Lu
        """
        Lx = -torch.mean(torch.sum(F.log_softmax(output_x,dim=-1)*trg_x,dim=-1)) # cross-entropy,supervised loss
        Lu = F.mse_loss(output_u,trg_u) # consistency reg
        return Lx,Lu