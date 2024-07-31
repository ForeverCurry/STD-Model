"""
Timeseries sampler
"""
import numpy as np

class Sampler():
    def __init__(self,
                 timeseries: np.ndarray,
                 input_size: int,
                 output_size: int,
                 target: int,
                 pretrain:bool=True):
        """
        Timeseries sampler.

        timeseries: Timeseries data to sample from. Shapes: (spatial dimension, temporal dimension);
        input_size: Input data size.
        output_size: Outsample window size.
        target: index of target variable.
        pretrain: Whether a pre-trained model is required.
        """
        self.timeseries = timeseries  # 高维序列
        self.pretrain = pretrain
        self.target=target
        self.d, self.length = self.timeseries.shape
        self.target_data = self.timeseries[target, :]
        self.input_size = input_size
        self.output_size = output_size
        self.test_length = 2*output_size
        self.train_length = self.timeseries.shape[1]-self.test_length
        self.start_index = self.input_size
        self.stop_index = self.train_length
    def __iter__(self):
        '''
        return: 
         X: "spatial dimension, input size"
         y: "input size+output size"
        '''
        # sequential sampling
        self.sample_index = self.start_index
        while True:
            if self.pretrain:
                X = self.timeseries[:, -self.input_size +self.sample_index:self.sample_index]
            else:
                X = self.timeseries[:, -self.input_size +self.sample_index:self.sample_index+self.output_size]
            y = self.target_data[self.sample_index-self.input_size:
                                    self.sample_index+self.output_size]

            if self.sample_index <= self.stop_index:
                self.sample_index += self.output_size
            else:
                self.sample_index = self.start_index
            yield X,  y
            
    def reset(self):
        '''
        Reinitialize the sampling points index 
        '''
        self.sample_index = self.start_index
