import torch
from common.losses import nrmse_loss
from common.ops import initialize_weights

class ARNN(torch.torch.nn.Module):
    '''
    ARNN algorithm for time series prediction.
    '''
    def __init__(self,
                 output_size: int,
                 x: torch.Tensor,
                 target: int,
                 iter: int=50,
                 out_feature: int=128,
                 random_transform: bool=True):
        super().__init__()
        self.max_iter = iter
        self.output_size = output_size  # L-1
        if not torch.is_tensor(x):
            self.noisy_x = torch.from_numpy(x).type(torch.float32)
        else:
            self.noisy_x = x
        self.d, self.input_size = x.shape
        self.train_y = self.noisy_x[target, :]
        self.hankel_y, self.y_mask = self.generate_hankel(self.train_y)  # shape:L*m
        if random_transform:   
            model = basisnet(self.d, self.input_size, out_feature).float()
            initialize_weights(model)           
            self.model = model
            self.fx = model(self.noisy_x)
        else:
            print("------------Linear-------------")
            self.fx = self.noisy_x
        self.d,self.m = self.fx.shape
        self.A = torch.randn(self.output_size+1, self.d, requires_grad=False)
        self.B = torch.randn(self.d, self.output_size+1, requires_grad=False)
        self.index_train = torch.zeros(self.d)

    def train(self, k=0.6, sigma=1e-5):
        """     
        ARNN algorithm.       
        
        :param k: Percentage of indices to be selected during training.
        :param sigma: Threshold for stopping the training.
        :returns: y_pre_mean: Predicted output sequence.
        """
        pre_init = torch.zeros(self.output_size)
        # hankel_y, _ = self.generate_hankel(y)
        for i in range(self.max_iter):

            index = torch.randint(0, self.d, (int(k*self.d),))
            self.update_B(index)
            y_temp = self.predict_b()
            y = torch.cat((self.train_y, y_temp), dim=0)
            hankel_y, _ = self.generate_hankel(y)
            self.update_A(hankel_y)

            y_pre_mean = self.predict_a()

            consistence = nrmse_loss(pre_init, y_pre_mean)
            if consistence <= sigma:
                return y_pre_mean, self.A
            else:
                pre_init = y_temp 
        return y_pre_mean, self.A

    def update_B(self, index):
        """
        Update matrix B based on input and target variables.

        :param index: Indices for updating matrix B.
        """
        y = self.hankel_y[:, :self.input_size -
                          self.output_size]  # shape: L*(m-L+1)
        y = torch.transpose(y, 1, 0)
        x = self.fx[index, :self.input_size -
                    self.output_size]  # shape:k*(m-L+1)
        x = torch.transpose(x, 1, 0)
        x = torch.where(torch.isnan(x),torch.nanmedian(x),x)
        new_B = torch.linalg.lstsq(y, x).solution  # shape:L*k
        new_B = new_B.T
        self.B[index, :] = (self.B[index,:] + new_B+ new_B*(1-self.index_train[index]).reshape(-1,1))/2
        self.index_train[index] = 1

    def update_A(self, y):
        """
        Update matrix A based on input and target variables.

        param y: Input tensor for updating matrix A.
        """
        x = torch.cat([self.B, self.fx], dim=1)
        x = torch.where(torch.isnan(x),torch.nanmedian(x),x)
        y = torch.cat([torch.eye(self.output_size+1), y], dim=1)
        A = torch.linalg.lstsq(x.T, y.T).solution
        self.A = A.T

    def predict_b(self):
        """
        Predict the output sequence based on current matrix B.
        
        :returns: Predicted output sequence.
        """
        x = self.fx[:, -self.output_size:] - \
            torch.matmul(self.B, self.hankel_y[:, -self.output_size:] *
                     self.y_mask[:, -self.output_size:])
        b = torch.zeros((self.output_size*self.d, self.output_size))
        for i, v in enumerate(self.B):  # 取每行
            for j in range(self.output_size):
                b[i*self.output_size+j:(i+1)*self.output_size,
                  j] = torch.flip(v[1+j:], dims=(0,))  # 分块下三角矩阵
        x = torch.flatten(x, 0, -1)  # 按行展开
        x = torch.where(torch.isnan(x),torch.nanmedian(x),x)
        return torch.linalg.lstsq(b, x).solution  # 给出预测

    def predict_a(self):
        """
        Predict the output sequence based on current matrix A.
        
        :returns: Predicted output sequence.
        """
        pre_hankel_y = torch.matmul(self.A, self.fx[:, -self.output_size:])
        pre_hankel_y = torch.flip(pre_hankel_y, dims=(0,))
        y_mean = torch.zeros(self.output_size)
        for i in range(self.output_size):
            y_i = torch.diagonal(pre_hankel_y, offset=i)
            y_mean[i] = torch.mean(y_i)
        return y_mean

    def generate_hankel(self, y):
        if len(y) != self.output_size+self.input_size:
            y_all = torch.zeros(self.output_size+self.input_size)
            y_all[:len(y)] = y
        else:
            y_all = y
        outsample = torch.zeros((self.output_size+1, self.input_size))
        outsample_mask = torch.zeros((self.output_size+1, self.input_size))
        outsample[0, :] = y_all[:self.input_size]
        outsample_mask[0, :] = 1.0
        for i in range(self.output_size):
            outsample[i+1, :] = y_all[i+1:self.input_size+i+1]
            outsample_mask[i+1, :-i-1] = 1.0
        return outsample, outsample_mask

class basisnet(torch.nn.Module):
    def __init__(self,
                in_features: int,
                input_size: int,
                output_size: int,
                hidden_features: int=512,
                layers: int=3):
        super().__init__()
        self.elu = torch.nn.ELU(inplace=False)
        self.temporal_layer = torch.nn.Linear(
            in_features=input_size, out_features=input_size)
        self.inlayer = torch.nn.Linear(
            in_features=in_features, out_features=hidden_features, bias=False)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(in_features=hidden_features, out_features=hidden_features, bias=False)
             for _ in range(layers - 1)])
        self.outlayer = torch.nn.Linear(
            in_features=hidden_features, out_features=output_size, bias=False)

    def forward(self, x):
        block_input = torch.tanh(self.inlayer(x.T))
        for layer in self.layers:
            block_input = torch.tanh(layer(block_input))
        output = self.outlayer(block_input)
        return  output.T