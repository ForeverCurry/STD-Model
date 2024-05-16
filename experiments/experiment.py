import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from common.losses import nrmse_loss, pearson
from common.ops import hankel_to_scalar, hank
from statsforecast.models import AutoETS, AutoTheta, AutoARIMA
import pyEDM
from models.RDE import linear_sti
from models.ARNN import ARNN
from models.STD import STD
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
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
    
    def test(self, hyper_list: list, size: int, save: str=None):
        '''
        :param hyperlist: the hyperparameter of STD model;
        :param size: sample size of the test;
        :param save: whether to save prediction results. Defaults to False;
        :return: test sample's NRMSE and PCC.
        '''
        model = STD(in_feature = self.in_feature,in_size = self.input_size,
                    out_size = self.output_size, lambda_1=hyper_list[0],
                    lambda_2 = hyper_list[1]).float()
        results = []
        losses = []
        pccs = []
        self.dataset.reset()
        A = None
        for i, (X, y) in enumerate(iter(self.dataset)):
            mask, hankel_y, test_y = self.con_Y(y)
            if self.warm:
                hat_y, A = self.run(model, mask, hankel_y, X, A)
            else:
                hat_y, A = self.run(model, mask, hankel_y, X, None)
            results.append(y)
            results.append(hat_y.detach().numpy())
            loss = nrmse_loss(hat_y[-self.output_size:],test_y)
            pcc = pearson(hat_y[-self.output_size:],test_y)
            losses.append(loss)
            pccs.append(pcc)
            print(f'Iteration:{i}   |   Test loss:{loss:.4f}')
            if i+1 >= size:
                # pcc = pearson(np.array(pres[-size*self.output_size:]), np.array(gts[-size*self.output_size:]))
                if save != None:
                    results = pd.DataFrame(data=results)
                    results.to_csv(f'./results/STD_{save}.csv',index=False)
                return losses, pccs
            

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
        elif self.base_model == 'MVE':
            time = np.arange(X.shape[1]).reshape(1,-1)
            X = np.concatenate((time,X),axis=0)
            columns = np.concatenate((['time'],np.arange(self.in_feature)),axis=0)
            X = pd.DataFrame(X.T, columns=columns).rename(columns =str)
            
            MV = pyEDM.Multiview(dataFrame = X.copy(), lib = [1, self.input_size], 
                                 pred = [self.input_size,self.input_size+self.output_size-1], E = 1, multiview=1,
                                columns = np.arange(self.in_feature), target = f"{self.target}", numThreads=4,)
            
            temp_pre = MV['Predictions']['Predictions'].values[1:]
            
        y_temp = np.concatenate((y[:self.input_size], temp_pre))
        mask, Y =hank(y_temp, self.input_size, self.output_size, refine=True)
        temp_loss = nrmse_loss(temp_pre, test_y)
        return mask, Y, y_temp, test_y, temp_loss
    
    def test(self, size: int, save: str=None, hyper_list: list=[0.1,0.1],):
        '''
        :param hyperlist: the hyperparameter of STD model;
        :param size: sample size of the test;
        :param save: Model to save. if None, the prediction results will not be saved. Defaults to None;
        :returns: test sample's NRMSE and PCC.
        '''
        model = STD(in_feature = self.in_feature,in_size = self.input_size,
                                out_size = self.output_size, lambda_1=hyper_list[0],
                                lambda_2 = hyper_list[1]).float()
        ref_pres = []
        # temp_pres = []
        losses = []
        temp_losses = []
        self.dataset.reset()
        for i, (X, y) in enumerate(iter(self.dataset)):        
            mask, hankel_y, y_temp, test_y, temp_loss = self.base_pre(X, y)
            hat_y, A = self.run(model, mask, hankel_y, X)
            ref_pres.append(y_temp)
            ref_pres.append(hat_y.detach().numpy())
            loss = nrmse_loss(hat_y[-self.output_size:],test_y)
            losses.append(loss)
            temp_losses.append(temp_loss)
            print(f'Refined loss:{loss:.4f}   |   Original loss:{temp_loss:.4f}')
            if i >= size: 
                if save != None:
                    ref_pres = pd.DataFrame(data=ref_pres)
                    ref_pres.to_csv(f'./results/{self.base_model}_ref_{save}.csv',index=False)
                return losses, temp_losses
            