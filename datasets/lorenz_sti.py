# 生成洛伦兹动力系统的仿真数据

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy import special
import gin
np.random.seed(20230823)


def lorenz(vars, t, pars=[10., 28., 8/3]):
    '''
    洛伦兹动力系统：
    dx/dt = p(y-x)
    dy/dt = x(q-z)-y
    dz/dt = xy-rz

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
    return np.array([dx_dt, dy_dt, dz_dt])

# 定义96维耦合洛伦兹系统的演化函数


@gin.configurable
class lorenz_coupled():
    def __init__(self,
                 x: np.array,
                 t: list,
                 start: int,
                 stop: int,
                 target: int = 12,
                 input_size: int = 27,
                 output_size: int = 12,
                 noise:int=0.5,
                 sigma: float = 10.0,
                 beta: float = 8.0/3.0,
                 rho: float = 28.0,
                 gamma: float = 0.1,
                 delta: float = 0.1):
        '''
        x: initial value for generate curve
        start: start point of integration range 
        stop: end point of integration range
        t: sample points
        target: index of target value;
        output_size: prediction steps;
        noise: the variance of noise;
        sigam, beta, rho ,gamma and delta: lorenz system parameters;
        '''
        self.n = x.shape[0]
        self.target = int(target)
        self.input_size = input_size
        self.output_size = output_size
        self.noise=noise
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.gamma = gamma
        self.delta = delta
        self.timeseries = self._generate_data(
            start=start, stop=stop, t=t, x=x).y
        self.test_length = 5*output_size
        self.train_length = self.timeseries.shape[1]-self.test_length
        self.sample_points = pd.read_csv(r'D:\ML\Time_series\mymodel\datasets\lorenz\sample_points.csv',
                                         index_col=None,header=None).values.squeeze()
        self.index = 0
        
    def _lorenz_equation(self, t, y):
        '''
        coupled lorenz equation
        '''
        dxdt = np.zeros_like(y)
        end = int(self.n/3)
        for i in range(0, end, 3):
            dxdt[i] = self.sigma*(y[i+1] - y[i])
            dxdt[i+1] = self.rho*y[i] - y[i+1] - y[i]*y[i+2]
            dxdt[i+2] = y[i]*y[i+1] - self.beta*y[i+2] + self.gamma*(y[(i+3) % self.n]-y[(i-3) % self.n]) + \
                self.delta*(y[(i+6) % self.n]-2*y[i]+y[(i-6) % self.n])
        return dxdt

    def _generate_data(self, start, stop, t, x):
        return solve_ivp(self._lorenz_equation, t_span=[start, stop], y0=x, t_eval=t)

    def __iter__(self):
        '''
        return: 
         X: "spatial dimension, input size"
         y: "input size+output size"
        '''
        self.index=0
        while True:
            sampled_point = self.sample_points[self.index]
            self.index += 1
            X = self.timeseries[:, -self.input_size+sampled_point:sampled_point]
            if self.noise>0:
                X = X+self.noise*np.random.randn(self.n,self.input_size)
            y = np.concatenate((X[self.target,:self.input_size],self.timeseries[self.target,sampled_point:sampled_point+self.output_size]))

            sampled_point += self.output_size
            yield X, y
            
    def reset(self,):
        '''
        Reinitialize the sampling points index 
        '''
        self.index = 0
        