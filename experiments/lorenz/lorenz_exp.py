import numpy as np
np.random.seed(230823)
from models.STD import pre_exp,refine_exp
from itertools import product
from datasets.lorenz_sti import lorenz_coupled
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

parser = argparse.ArgumentParser('Lorenz experiment')
parser.add_argument('--lambda_1', type=list, default=[0.1,1,10],
                    help='Cross-validation parameter sets for lambda_1',)
parser.add_argument('--lambda_2', type=list, default=[0.1,0.5,1], 
                    help='Cross-validation parameter sets for lambda_1')
parser.add_argument('--noisy', type=float, default=0.5,
                    help='The variance of the noise')
parser.add_argument('--in_feature', type=int, default=90,
                    help='Number of spatial dimension of data')
parser.add_argument('--input_size', type=int, default=27,
                    help='Number of training lengh')
parser.add_argument('--output_size', type=int, default=12,
                    help='Number of prediction steps')
parser.add_argument('--val_size', type=int, default=20,
                    help='Size of validation set')
parser.add_argument('--test_size', type=int, default=30,
                    help='Size of test set')
parser.add_argument('--target', type=int, default=20,
                    help='Index of target')
parser.add_argument('--niters', type=int, default=100000,
                    help='Maximum number of iterations')
parser.add_argument('--epsilon', type=float, default=1e-5,
                    help='Value of difference when training will be stopped')
# parser.add_argument('--warm', action='store_True', default=False)
parser.add_argument('--refine',action='store_true', default=False,
                    help='If true')
parser.add_argument('--refine_model', type=str, default=None,
                    help="if refine=True, the refined model is in ['ETS','Theta', 'Arima', 'RDE', 'ARNN']")
args = parser.parse_args()

if __name__ == '__main__':
    ### Training Set
    start = float(-5.)
    stop = float(5.)
    step = float(0.02)
    init = np.random.uniform(-5, 5, args.in_feature)
    t_eval = np.arange(start, stop, step)
    training_set = lorenz_coupled(x=init, t=t_eval, start=start, stop=stop, input_size=args.input_size, 
                                  target=args.target, output_size=args.output_size,noise=args.noisy)

    if not args.refine:
        exp = pre_exp(target=args.target,in_feature=args.in_feature, input_size=args.input_size,
                    output_size = args.output_size, dataset=training_set, warm =True)
        ### Validation
        hyper = product(args.lambda_1,args.lambda_2)
        best_par = exp.val(hyper,size=args.val_size)
        
        ### Test
        nrmse, pcc = exp.test(best_par, size=args.test_size+args.val_size, save=False)
        ave_loss = sum(nrmse[-args.test_size:])/args.test_size
        ave_pcc = sum(pcc[-args.test_size:])/args.test_size
        print(f'Target {args.target} of Lorenz:\nModel loss: {ave_loss}    |      Pearson Correlation Coefficient: {ave_pcc}')

    else:
        assert args.refine_model in ['ETS', 'Theta', 'Arima', 'RDE', 'ARNN']
        exp = refine_exp(target=args.target,in_feature=args.in_feature, input_size=args.input_size,
                    output_size = args.output_size, dataset=training_set, base_model=args.refine_model)
        
        ### refine
        nrmse, temp_nrmse, pcc, temp_pcc = exp.test(size=args.test_size+args.val_size, save=False)
        ref_loss = sum(nrmse[-args.test_size:])/args.test_size
        temp_loss = sum(temp_nrmse[-args.test_size:])/args.test_size
        print(f'Target {args.target} of Lorenz: Model {args.refine_model}\nOriginal loss: {temp_loss:.4f}    |      refined loss: {ref_loss:.4f}\n Original PCC: {temp_pcc:.4f}    |      refined PCC: {pcc:.4f}')
        
        
    
    
