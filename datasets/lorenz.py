# 生成洛伦兹动力系统的仿真数据

import numpy as np
from scipy.integrate import solve_ivp
import gin

def lorenz(vars, t, pars=[10.,28.,8/3]):

    '''
    洛伦兹动力系统：
    dself.x/dt = p(y-self.x)
    dy/dt = self.x(q-z)-y
    dz/dt = self.xy-rz
    
    Parameters：
    ----------------------
    vars：表示变量初始值;
        类型：array
    pars：表示超参；
        类型：array
    '''
    x, y, z = vars
    p, q, r = pars
    dx_dt = p*(y-x)
    dy_dt = x*(q-z)-y
    dz_dt = x*y-r*z
    return np.array([dx_dt,dy_dt,dz_dt]) 

# 定义96维耦合洛伦兹系统的演化函数


@gin.configurable
class lorenz_coupled(): 
    def __init__(self,
                x:np.array, 
                t:list,
                start:int,
                stop:int,
                # target:int,
                input_size:int,
                output_size:int,
                label_size:int=None,
                # test_length:float=0.1,
                batch_size:int=1024,
                sigma:float=10.0, 
                beta:float=8.0/3.0, 
                rho:float=28.0, 
                gamma:float=0.1, 
                delta:float=0.1):
        self.n = x.shape[0]
        # self.target = target
        self.batch_size = batch_size
        self.input_size = input_size
        if label_size == None:
            self.label_size = int(self.input_size/2)
        else:
            self.label_size = label_size
        self.output_size = output_size
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.gamma = gamma
        self.delta = delta
        self.timeseries = self._generate_data(start=start, stop=stop, t=t, x=x).y
        self.test_length = 5*output_size
        self.train_length = self.timeseries.shape[1]-self.test_length
    def _lorenz_equation(self, t, y):
        dxdt = np.zeros_like(y)
        end = int(self.n/3)
        for i in range(0, end, 3):
            dxdt[i] = self.sigma*(y[i+1] - y[i])
            dxdt[i+1] = self.rho*y[i] - y[i+1] - y[i]*y[i+2]
            dxdt[i+2] = y[i]*y[i+1] - self.beta*y[i+2] + self.gamma*(y[(i+3)%96]-y[(i-3)%96]) + \
                        self.delta*(y[(i+6)%96]-2*y[i]+y[(i-6)%96])
        return dxdt
    def _generate_data(self, start, stop, t, x):
        return solve_ivp(self._lorenz_equation, t_span=[start, stop], y0=x, t_eval=t)
    def __iter__(self):
        '''
        size:[batch size, d]
        '''
        while True:
            insample = np.zeros((self.batch_size, self.input_size,self.n))
            insample_mask = np.zeros((self.batch_size, self.input_size,self.n))
            outsample = np.zeros((self.batch_size, self.output_size+self.label_size,self.n))
            outsample_mask = np.zeros((self.batch_size, self.output_size+self.label_size,self.n))
            sampled_ts_indices = np.random.randint(self.train_length-self.output_size-self.input_size, 
                                                   size=self.batch_size) # 抽样得到序列的索引
            sampled_ts_indices = np.sort(sampled_ts_indices)
            for i, sampled_index in enumerate(sampled_ts_indices):
                out_index = (sampled_index+self.input_size-self.label_size,sampled_index+self.input_size+self.output_size)
                outsample[i,:,:] = self.timeseries[:,out_index[0]:out_index[1]].T
                outsample_mask[i, :] = 1.0
                in_index = (sampled_index, sampled_index+self.input_size)
                sample = self.timeseries[:,in_index[0]:in_index[1]].T
                insample[i,-sample.shape[0]:,:] = sample
                insample_mask[i,-sample.shape[0]:,:] = 1.0  
                # Mask的作用：当序列长度不足输出长度时，使得不足的区域值均为0，数据从后向前填入
            yield insample, insample_mask, outsample, outsample_mask
    def last_sampling(self):
        # print(self.timeseries.shape,self.train_length)
        outsample = np.zeros((5,self.output_size+self.label_size,self.n))
        insample = np.zeros((5,self.input_size,self.n))
        for i, idx in enumerate(range(self.train_length,self.timeseries.shape[1]-self.output_size,self.output_size)):
            outsample[i,:,:] = self.timeseries[:,idx-self.label_size:idx+self.output_size].T
            insample[i,:,:] = self.timeseries[:,idx-self.input_size:idx].T # 抽样得到的单序列
        insample_mask = np.ones_like(insample)
        outsample_mask = np.ones_like(outsample)
        return insample, insample_mask, outsample, outsample_mask

