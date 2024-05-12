import torch
import numpy as np
class linear_sti(torch.nn.Module):
    def __init__(self, 
                 in_feature:int,
                 out_size:int,
                 ):
        super(linear_sti,self).__init__()
        self.register_buffer('A',torch.zeros((out_size, in_feature)))
    def forward(self,data):
        x, y = data
        d, m = x.shape
        XX = x[:,:-1]@x[:,:-1].T+np.eye(d)
        XX_inv = np.linalg.inv(XX)
        l = len(y)
        y =torch.from_numpy(y)
        XX_inv = torch.from_numpy(x[:,:-1].T@XX_inv)
        for i in range(l-m):
            A_temp= y[i+1:m+i]@XX_inv
            self.A[i,:] = A_temp
            y[m+i] = A_temp@x[:,-1]
        return y, self.A