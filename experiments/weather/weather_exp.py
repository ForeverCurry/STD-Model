import os
import argparse
from models.STD import pre_exp,refine_exp
from dataset.weather import weatherDataset
from itertools import product
from common.sampler import Sampler
from common.plot import plot_result
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

parser = argparse.ArgumentParser('weather experiment')
parser.add_argument('--lambda_1', type=list, default=[0.01,0.1,1],
                    help='Cross-validation parameter sets for lambda_1',)
parser.add_argument('--lambda_2', type=list, default=[0.01,0.1,1], 
                    help='Cross-validation parameter sets for lambda_2')
parser.add_argument('--in_feature', type=int, default=16,
                    help='Number of spatial dimension of data')
parser.add_argument('--input_size', type=float, default=14,
                    help='Number of training length')
parser.add_argument('--output_size', type=int, default=6,
                    help='Number of prediction steps')
parser.add_argument('--val_size', type=int, default=20,
                    help='Size of validation set')
parser.add_argument('--test_size', type=int, default=30,
                    help='Size of test set')
parser.add_argument('--target', type=int, default=15,
                    help='Index of target')
parser.add_argument('--niters', type=int, default=100000,
                    help='Maximum number of iterations')
parser.add_argument('--epsilon', type=float, default=1e-5,
                    help='Value of difference when training will be stopped')
parser.add_argument('--warm',action='store_true', default=False,
                    help='If true')
parser.add_argument('--refine',action='store_true', default=False,
                    help='If true')
parser.add_argument('--refine_model', type=str, default=None,
                    help="if refine=True, the refined model is in ['ETS','Theta', 'Arima', 'MVE','ARNN','RDE']")
args = parser.parse_args()


if __name__ == '__main__':

    # Training Set
    dataset = weatherDataset.load()
    training_set = dataset.to_numpy().T

    if not args.refine:
        #### load data
        training_set = Sampler(training_set, args.input_size, args.output_size, target=args.target)
        #### Training model
        exp = pre_exp(target=args.target, in_feature=args.in_feature, input_size=args.input_size,
                    output_size = args.output_size, dataset=training_set, warm=args.warm)
        
        ### Cross validation
        # hyper = product(args.lambda_1,args.lambda_2)
        # best_par = exp.val(hyper,size=args.val_size)
        best_par = [1,0.1]
        ### Test
        nrmse, pccs = exp.test(best_par, size=args.test_size+args.val_size, save='weather')
        ave_loss = sum(nrmse[-args.test_size:])/args.test_size
        ave_pcc = sum(pccs[-args.test_size:])/args.test_size
        print(f'Average loss of operative temperature is {ave_loss:.4f} and pcc is {ave_pcc:.4f}')
        titles = [f'Operative temperature ']
        plot_result('./weather/STD_weather.csv', args.input_size, args.output_size, args.test_size, titles, './png/plankton.pdf')
    else:
        assert args.refine_model in ['ETS', 'Theta', 'Arima', 'MVE', 'RDE', 'ARNN']
        if args.refine_model == 'MVE':
            training_set = Sampler(training_set, args.input_size, args.output_size, target=args.target,
                                  train=False)
        else:
            training_set = Sampler(training_set, args.input_size, args.output_size, target=args.target)
        exp = refine_exp(target=args.target,in_feature=args.in_feature, input_size=args.input_size,
                    output_size = args.output_size, dataset=training_set, base_model=args.refine_model)
        
        ### refine
        nrmse, temp_nrmse = exp.test(size=args.test_size+args.val_size, save='weather')
        ref_loss = sum(nrmse[-args.test_size:])/args.test_size
        temp_loss = sum(temp_nrmse[-args.test_size:])/args.test_size
        print(f'Operative temperature: Model {args.refine_model}\nOriginal loss: {temp_loss:.4f}    |      refined loss: {ref_loss:.4f}')
    
