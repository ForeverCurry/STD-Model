import torch
import numpy as np
from scipy.linalg import toeplitz
import torch.nn.functional
from common.losses import nrmse_loss
from common.ops import hankel_to_scalar, soft_threshold
import pandas as pd
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
                 lambda_3:float = 0.01,
                 order: int = 2,
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
        self.order = order
        self.register_buffer("lambda_2",torch.tensor(
            lambda_2, dtype=torch.float32))
        self.register_buffer('lambda_1',torch.tensor(
            lambda_1, dtype=torch.float32))
        self.register_buffer('lambda_3',torch.tensor(
            lambda_3, dtype=torch.float32))
        self.register_buffer('tau',torch.tensor(0
            , dtype=torch.float32))
        self.register_buffer('a',torch.tensor(
            1, dtype=torch.float32))
        self.register_buffer('A',torch.zeros((out_size+1, in_feature)))
        self.register_buffer('B',torch.zeros((out_size+1, in_feature)))
        self.register_buffer('Delta', self.delta()) # the second-order difference matrix
        self.register_buffer('eig',torch.clone(self.maxeig(self.delta())))
        # M1, M2, M3, M4 = self.linear_mask()
        # self.register_buffer('M1',M1)
        # self.register_buffer('M2',M2)
        # self.register_buffer('M3',M3)
        # self.register_buffer('M4',M4)
        self.k=0
    def diff_coef(self,):
        coef = np.zeros(self.order+1)
        ids = np.arange(0,self.order+1)
        if self.order == 2:
            values = [1,-2,1]
        elif self.order == 1:
            values = [1/2,-2,3/2]
        elif self.order ==3 :
            values = [1,-3,3,-1]
        coef[ids] = values
        return coef
    def linear_mask(self):
        '''
        Generate the linear mask matrix
        '''
        M1 = torch.zeros((self.out_size+1,self.out_size+1))
        M2 = torch.zeros((self.in_size, self.in_size))
        M3 = torch.zeros((self.out_size+1,self.out_size+1))
        M4 = torch.zeros((self.in_size, self.in_size))
        for i in range(self.out_size):
            M1 += torch.diag(torch.cat((torch.ones(self.out_size-i),torch.zeros(1+i))))
            M2 += torch.cat((torch.zeros((1+i,self.in_size)),torch.cat((torch.eye(self.in_size-1-i),torch.zeros((self.in_size-1-i,1+i))),dim=1)),dim=0)
            M4 += torch.diag(torch.cat((torch.ones(self.in_size-1-i),torch.zeros(1+i))))
            M3 += torch.cat((torch.cat((torch.zeros((self.out_size-i,1+i)),torch.eye(self.out_size-i)), dim=1), torch.zeros((1+i,self.out_size+1))),dim=0)
        M1 = M1/self.out_size
        M2 = M2/self.out_size
        M3 = M3/self.out_size
        M4 = M4/self.out_size
        return M1, M2, M3, M4
    def delta(self):
        '''
        Generate the difference matrix
        '''
        T = np.zeros((self.out_size+1, self.out_size+1))
        coef = self.diff_coef()
        T2 = np.concatenate((coef,np.zeros(self.out_size-len(coef)+1)),axis=0)
        T2 = np.tril(toeplitz(T2))
        T += T2
        T[0,0] = 0
        T[1,0] = -1
        return torch.from_numpy(T)
    
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
        temp = []
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
             
        y_k = torch.ones(l+m-1, device=x.device)
        
        #### run optimization
        for i in range(int(self.max_iter)): 
            self.update_A(x, part_y, mask)
            hankel_pre = self.A@x
            ax_temp = hankel_to_scalar(hankel_pre)
            convergence = nrmse_loss(y_k[-l+1:],ax_temp[-l+1:])
            if (i+1)%50 == 0:
            #     print(f'iteration {i+1}: train NRMSE {nrmse_loss(x[20,:m],ax_temp[:m]).item():.4f}')
                temp.append(ax_temp.clone().numpy().squeeze())
            if convergence <= self.epsilon :
                temp_data = pd.DataFrame(data=temp)
                temp_data.to_csv(rf'temp\{self.k}_data.csv',index=False)
                self.k+=1
                # print(f'iteration {i+1} converged with NRMSE {convergence:.4f}')
                return self.A@x, self.A
            else:
                y_k = ax_temp.clone()
        temp_data = pd.DataFrame(data=temp)
        temp_data.to_csv(rf'temp\{self.k}_data.csv',index=False)
        self.k+=1
        return self.A@x, self.A

    def update_A(self, x, y, mask):
        '''
        update variable A by FISTA-based algorithm.
        '''
        grad1_ = ((mask*(self.A@x-y))*mask)@x.T
        A_ = self.lambda_1*self.Delta.T@self.Delta@self.A+ grad1_ 
        A_ = self.B - 2*self.tau*A_
        A_ = soft_threshold(A_, self.tau*self.lambda_2)
        a_ = (1+torch.sqrt(1+4*torch.pow(self.a,2)))/2
        self.B.data = self.A+((self.a-1)/a_)*(A_-self.A)
        self.A.data = A_
        self.a.data = a_
    # def update_A(self, x, y, mask):
    
    #     '''
    #     update variable A by FISTA-based algorithm.
    #     '''
    #     # print(self.X1@self.A@x[:,1:]-self.X2@self.A@x[:,:-1])
    #     res = self.M1@self.A@x@self.M2-self.M3@self.A@x@self.M4
    #     grad1_ = ((mask*(self.A@x-y))*mask)@x.T
    #     grad2_ = self.M1.T@res@self.M2.T@x.T + self.M3.T@res@self.M4.T@x.T
    #     A_ = grad1_ + self.lambda_1*self.Delta.T@self.Delta@self.A + self.lambda_3*grad2_
    #     A_ = self.B - self.tau*A_
    #     A_ = soft_threshold(A_, self.tau*self.lambda_2)
    #     a_ = (1+torch.sqrt(1+4*torch.pow(self.a,2)))/2
    #     self.B.data = self.A+((self.a-1)/a_)*(A_-self.A)
    #     self.A.data = A_
    #     self.a.data = a_


class STD_test(torch.nn.Module):
    '''
    STD Model
    '''
    def __init__(self,
                 in_feature: int,
                 out_size: int,
                 hidden_size: int = 512):
        super(STD_test, self).__init__()
        '''
        :param in_feature: The spatial dimension of data;
        :param out_size: Prediction steps;

        '''
        self.in_feature = in_feature
        self.out_size = out_size
        self.linear = torch.nn.Linear(in_feature, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, out_size+1)
    def forward(self, x):
        y = self.linear(x)
        # y = torch.tanh(self.linear2(y))
        y = self.linear3(y)
        return y, self.linear3.weight.data
    
class train():
    def __init__(self,
                 model,
                 out_size: int,
                 lambda_1: float = 0.1,
                 lambda_2:float = 0.1,
                 order: int = 2,
                 periods: list = [1],
                 max_iter: int = 10000,
                 eps: float = 1e-4,
                 device: str = 'cpu'):
        self.out_size = out_size
        self.model = model.to(device)
        self.lambda_1 = torch.tensor(lambda_1, dtype=torch.float16, device=device)
        self.lambda_2 = torch.tensor(lambda_2, dtype=torch.float16, device=device)
        self.order = order
        self.periods = periods  
        self.max_iter = max_iter
        self.eps = eps
        self.device = device
        self.diff = self.delta()
        self.criterion = torch.nn.MSELoss()
        self.lasso = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    def diff_coef(self, period):
        coef = np.zeros(self.order*period+1)
        ids = np.arange(0,self.order*period+1,period)
        if self.order == 2:
            values = [1,-2,1]
        elif self.order == 1:
            values = [1/2,-2,3/2]
        elif self.order ==3 :
            values = [1,-3,3,-1]
        coef[ids] = values
        return coef
    def delta(self):
        '''
        Generate the difference matrix
        '''
        T = np.zeros((self.out_size+1, self.out_size+1))
        for period in self.periods:
            if period > 0:
                coef = self.diff_coef(period)
                T2 = np.concatenate((coef,np.zeros(self.out_size-len(coef)+1)),axis=0)
                T2 = np.tril(toeplitz(T2))
                T += T2
            else:
                raise ValueError("Period must be greater than 0")
        T[0,0] = 0
        T[1,0] = -1
        return torch.from_numpy(T/len(self.periods)).float()
    def train_model(self, data, mask, initA=None):
        self.model.train()
        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)
        d, m = x.shape
        l, _ = y.shape
        mask = mask.to(self.device)
        if initA is not None:
            initA = initA.to(self.device)
        loss_ = torch.tensor(float('inf'), device=self.device)
        y_k = torch.rand(l+m-1, device=x.device)
        for i in range(self.max_iter):
            pred, coef = self.model(x.T)
            loss = self.criterion(mask*(pred.T), mask*y) + self.lambda_2*self.lasso(coef, torch.zeros_like(coef)) + self.lambda_1*self.criterion(torch.matmul(self.diff,coef), torch.zeros_like(torch.matmul(self.diff,coef)))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            ax_temp = hankel_to_scalar(pred)
            convergence = nrmse_loss(y_k[-l+1:],ax_temp[-l+1:])
            if i%100 == 0:
                print(f'iteration {i+1}: train NRMSE {nrmse_loss(y,pred.T).item():.4f}  | convergence {convergence:.4f}')
            if convergence <= self.eps:
                return pred, coef
            else:
                y_k = ax_temp.clone()
        return pred, coef
        