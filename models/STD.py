import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.linalg import toeplitz
from common.losses import nrmse_loss, pearson
from common.ops import hankel_to_scalar, hank, soft_threshold
from statsforecast.models import AutoETS, AutoTheta, AutoARIMA
from models.RDE import linear_sti
from models.ARNN import ARNN
class Experiment(ABC):
    '''Experiment base class.'''
    def __init__(self,
                dataset,
                target: int,
                in_feature: int,
                input_size: int,
                output_size: int, ) -> None:
        '''
        :param target: the index of target variable;
        :param in_feature: number of spatial dimension;
        :param input_size: number of series length;
        :param output_size: step size of prediction;
        '''
        self.dataset = dataset
        self.target = target
        self.in_feature = in_feature
        self.input_size = input_size
        self.output_size = output_size
        
    @abstractmethod
    def test(self):
        '''
        test the performance of the model on the test dataset.
        '''
        pass
    
    def run(self, model, mask, hankel_y, X, initA=None ):
        '''
        :return: prediction result and coefficient;
        '''
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).type(torch.float32)
        if not isinstance( hankel_y, torch.Tensor):
            hankel_y = torch.from_numpy(hankel_y).type(torch.float32)
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X[:,:self.input_size]).type(torch.float32)
        y, A = model(data=(X, hankel_y),
                    mask=mask, initA=initA)
        y = hankel_to_scalar(y)
        return y, A
    

    
class pre_exp(Experiment):
    def __init__(self, 
                 dataset,
                 target: int,
                 in_feature: int,
                 input_size: int,
                 output_size: int, 
                 warm: bool=True,
                 ) -> None:
        '''
        Construct STD experiments.
        :param dataset: data for validation and test. Type shoule be iteration;
        :param warm: If true, use last coefficient to initialize STD model. Defaults to True;
        '''
        super().__init__(dataset, target, in_feature, input_size, output_size)
        self.warm = warm 
    
    def con_Y(self, y):
        '''
        Construct the target hankel matrix and mask;
        '''
        y = y[-self.input_size-self.output_size:]
        test_y = torch.from_numpy(y[-self.output_size:]).type(torch.float32)
        mask, hankel_y = hank(y, self.input_size, self.output_size)
        return mask, hankel_y, test_y
        
    def val(self, hyper_lists: list, size: int):
        '''
        :param hyperlist: the hyperparameter of STD model;
        :param size: sample size of the test;
        :return: the best hyperparamter;
        '''
        val_loss_ = float('inf')
        best_par = []
        for hyper_list in hyper_lists:
            self.dataset.reset()
            model = STD(in_feature = self.in_feature,in_size = self.input_size,
                               out_size = self.output_size, lambda_1=hyper_list[0],
                                lambda_2 = hyper_list[1]).float()
            val_loss = 0
            A = None
            print(f'current hyperparameter:{hyper_list}')
            for i, (X, y) in enumerate(iter(self.dataset)):
                
                mask, hankel_y, test_y = self.con_Y(y)
                hat_y, A = self.run(model, mask, hankel_y, X, A)
                
                if i<size:
                    loss_ = nrmse_loss(hat_y[-self.output_size:],test_y)
                    if loss_<float('inf'):
                        val_loss += loss_
                else:
                    break
            print(f'current val loss:{val_loss}')
            
            if val_loss < val_loss_:
                best_par = hyper_list
                val_loss_ = val_loss
        print(f'best hyperparameter:{best_par} | val loss:{val_loss_}')
        return best_par
    
    def test(self, hyper_list: list, size: int, save: bool=False):
        '''
        :param hyperlist: the hyperparameter of STD model;
        :param size: sample size of the test;
        :param save: whether to save prediction results. Defaults to False;
        :return: test sample's NRMSE and PCC.
        '''
        model = STD(in_feature = self.in_feature,in_size = self.input_size,
                    out_size = self.output_size, lambda_1=hyper_list[0],
                    lambda_2 = hyper_list[1]).float()
        pres = []
        gts = []
        losses = []
        # pccs = []
        self.dataset.reset()
        A = None
        for i, (X, y) in enumerate(iter(self.dataset)):
            mask, hankel_y, test_y = self.con_Y(y)
            hat_y, A = self.run(model, mask, hankel_y, X, A)
            
            gts.extend(y)
            pres.extend(hat_y.detach().numpy())
            loss = nrmse_loss(hat_y[-self.output_size:],test_y)
            # pcc = pearson(hat_y[-self.output_size:],test_y)
            losses.append(loss)
            # pccs.append(pcc)
            print(f'Iteration:{i}   |   Test loss:{loss:.4f}')
            if i >= size:
                pcc = pearson(np.array(pres), np.array(gts))
                result = pd.DataFrame(data=result)
                if save:
                    result.to_csv(f'{save}.csv',index=False)
                return losses, pcc
            

class refine_exp(Experiment):
    '''
    Run refinement experiment;
    '''
    def __init__(self, dataset, target: int, in_feature: int, input_size: int, output_size: int, base_model: str) -> None:
        super().__init__(dataset, target, in_feature, input_size, output_size)    
        self.base_model = base_model
    def refine(self, X, temp_y, model):
        '''
        Refine predictions by STD model;
        :returns: refined predictions;
        '''
        mask = np.ones_like(temp_y)
        y, A = self.run(model, mask, temp_y, X)
        return y, A
    
    def base_pre(self, X, y):
        '''
        :returns: preditions by base model.
        '''
        test_y = torch.from_numpy(y[-self.output_size:]).type(torch.float32)
        if self.base_model == 'ETS':
            temp_pre = AutoETS().forecast(y=X[self.target, :self.input_size],
                                X=np.delete(X[:,:self.input_size].T, self.target, axis=1),
                                h=self.output_size, fitted=True)
            temp_pre = temp_pre['mean']
        elif self.base_model == 'Theta':
            temp_pre = AutoTheta().forecast(y=X[self.target, :self.input_size],
                                            X=np.delete(X[:,:self.input_size].T, 
                                                        self.target, axis=1),
                                            h=self.output_size, fitted=True)
            temp_pre = temp_pre['mean']
        elif self.base_model == 'Arima':
            temp_pre = AutoARIMA(biasadj=True).forecast(y=X[self.target, :self.input_size],
                                                        h=self.output_size, fitted=True)
            temp_pre = temp_pre['mean']
        elif self.base_model == 'RDE':
            y_rde = np.zeros_like(y)
            y_rde[:self.input_size] = y[:self.input_size]
            temp_pre, _ = linear_sti(self.in_feature, self.output_size)(data=(X[:, :self.input_size], y_rde))
            temp_pre = temp_pre[-self.output_size:]
        elif self.base_model == 'ARNN':
            temp_pre, _ = ARNN(x=X, target=self.target, output_size=self.output_size).train()
            temp_pre = temp_pre[-self.output_size:].detach().numpy()
            
        y_temp = np.concatenate((y[:self.input_size], temp_pre))
        mask, Y =hank(y_temp, self.input_size, self.output_size, refine=True)
        temp_loss = nrmse_loss(temp_pre, test_y)
        return mask, Y, y_temp, test_y, temp_loss
    
    def test(self, size: int, save: bool=False, hyper_list: list=[0.1,0.1],):
        '''
        :param hyperlist: the hyperparameter of STD model;
        :param size: sample size of the test;
        :param save: whether to save prediction results. Defaults to False;
        :returns: test sample's NRMSE and PCC.
        '''
        model = STD(in_feature = self.in_feature,in_size = self.input_size,
                                out_size = self.output_size, lambda_1=hyper_list[0],
                                lambda_2 = hyper_list[1]).float()
        ref_pres = []
        temp_pres = []
        gts = []
        losses = []
        temp_losses = []
        self.dataset.reset()
        for i, (X, y) in enumerate(iter(self.dataset)):
            
            mask, hankel_y, y_temp,test_y, temp_loss = self.base_pre(X, y)
            hat_y, A = self.run(model, mask, hankel_y, X)
            
            gts.extend(y)
            ref_pres.extend(hat_y)
            temp_pres.extend(y_temp)
            loss = nrmse_loss(hat_y[-self.output_size:],test_y)
            losses.append(loss)
            temp_losses.append(temp_loss)
            print(f'Refined loss:{loss:.4f}   |   Original loss:{temp_loss:.4f}')
            if i >= size: 
                temp_pcc = pearson(np.array(temp_pres),np.array(gts))
                ref_pcc = pearson(np.array(ref_pres),np.array(gts))
                # result = pd.DataFrame(data=result)
                # if save:
                #     result.to_csv(f'{save}.csv',index=False)
                return losses, temp_losses, ref_pcc, temp_pcc
            
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
        A_ = soft_threshold(A_,self.tau*self.lambda_2)
        a_ = (1+torch.sqrt(1+4*torch.pow(self.a,2)))/2
        self.B.data = self.A+((self.a-1)/a_)*(A_-self.A)
        self.A.data = A_
        self.a.data = a_


