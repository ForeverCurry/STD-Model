import torch
import numpy as np
from scipy.linalg import toeplitz
from common.losses import nrmse_loss
from common.ops import hankel_to_scalar, soft_threshold
  
class STD(torch.nn.Module):
    '''
    STD Model
    '''
    def __init__(self,
                 in_feature: int,
                 in_size: int,
                 out_size: int,
                 lambda_1: float = 0.1,
                 lambda_2:float = 0.1,
                 max_iter: int = 1e5,
                 eps: float = 1e-5):
        super(STD, self).__init__()
        '''
        :param in_feature: The spatial dimension of data;
        :param in_size: The length of the input data;
        :param out_size: Prediction steps;
        :param lambda_1 and lambda_2: Values of regularization parameter. Defaults to 0.1;
        :param max_iter: Maximum number of iterations. Defaluts to 1e5;
        :param eps: Value of difference when training will be stopped. Defaults to 1e-5;
        '''
        self.epsilon = eps
        self.max_iter = max_iter
        self.in_size = in_size
        self.out_size = out_size
        
        self.register_buffer("lambda_2",torch.tensor(
            lambda_2, dtype=torch.float32))
        self.register_buffer('lambda_1',torch.tensor(
            lambda_1, dtype=torch.float32))
        self.register_buffer('tau',torch.tensor(0
            , dtype=torch.float32))
        self.register_buffer('a',torch.tensor(
            1, dtype=torch.float32))
        self.register_buffer('A',torch.zeros((out_size+1, in_feature)))
        self.register_buffer('B',torch.zeros((out_size+1, in_feature)))
        self.register_buffer('Delta', self.delta()) # the second-order difference matrix
        self.register_buffer('eig',torch.clone(self.maxeig(self.delta())))
  
    def delta(self,):
        '''
        Generate the second difference matrix
        '''
        T = [1, -2, 1]
        T2 = np.concatenate((T,np.zeros(self.out_size-2)),axis=0)
        T2 = np.tril(toeplitz(T2))
        T2[0,0] = 0
        T2[1,0] = -1
        return torch.from_numpy(T2)
    
    def maxeig(self, X):
        '''
        Compute the the square of the max eigenvalue of X;
        '''
        S = torch.linalg.svdvals(X)
        max_S = S[0]
        return torch.pow(max_S,2)

    def forward(self, data, mask:torch.Tensor, initA=None):
        '''
        optimize model
        
        :param data: the input data pairs (X, Y);
        :param mask: the index of known elements of Y;
        :param initA: Values to initialize weights A and B. If None, weights A and B will be initialized with 0. 
                    Defaults to None;
        :returns: predictions and coefficient.
        '''

        x, part_y = data
        d, m = x.shape
        l, _ = part_y.shape
        
        #### Initialize A and B
        if initA==None:
            self.A = torch.zeros_like(self.A)
            self.B = torch.zeros_like(self.B)
        else:
            self.A = initA
            self.B = initA
            
        #### initialize a
        self.a = torch.ones_like(self.a)
        
        #### initialize the step size tau
        self.tau = 1/(2*(self.maxeig(x)+self.lambda_1*self.eig))
        if self.tau*self.lambda_2>0.1:
            self.lambda_2 = torch.clone(0.1 / self.tau)
             
        y_k = torch.rand(l+m-1, device=x.device)
        
        #### run optimization
        for _ in range(int(self.max_iter)): 
            self.update_A(x, part_y, mask)
            ax_temp = hankel_to_scalar(self.A@x)
            convergence = nrmse_loss(y_k[-l+1:],ax_temp[-l+1:])
            if convergence <= self.epsilon :
                return self.A@x, self.A
            else:
                y_k = ax_temp.clone()
        return self.A@x, self.A

    def update_A(self, x, y, mask):
        '''
        update variable A by FISTA-based algorithm.
        '''
        A_ = self.lambda_1*self.Delta.T@self.Delta@self.A+(mask*(self.A@x-y))@x.T
        A_ = self.B - self.tau*A_
        A_ = soft_threshold(A_, self.tau*self.lambda_2)
        a_ = (1+torch.sqrt(1+4*torch.pow(self.a,2)))/2
        self.B.data = self.A+((self.a-1)/a_)*(A_-self.A)
        self.A.data = A_
        self.a.data = a_


