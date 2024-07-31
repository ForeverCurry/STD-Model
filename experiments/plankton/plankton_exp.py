import os
import argparse
from experiments.experiment import pre_exp,refine_exp
from itertools import product
from datasets.plankton import planktonDataset
from common.sampler import Sampler
from common.Plot.plot import plot_result
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

parser = argparse.ArgumentParser('plankton experiments')
parser.add_argument('--lambda_1', type=list, default=[0.01,0.1,1],
                    help='Cross-validation parameter sets for lambda_1',)
parser.add_argument('--lambda_2', type=list, default=[0.01,0.1], 
                    help='Cross-validation parameter sets for lambda_1')
parser.add_argument('--in_feature', type=int, default=12,
                    help='Number of spatial dimension of data')
parser.add_argument('--input_size', type=float, default=10,
                    help='Number of training length')
parser.add_argument('--output_size', type=int, default=4,
                    help='Number of prediction steps')
parser.add_argument('--val_size', type=int, default=20,
                    help='Size of validation set')
parser.add_argument('--test_size', type=int, default=30,
                    help='Size of test set')
parser.add_argument('--target', nargs='+',type=int, default=[3,11],
                    help='Index of target')
parser.add_argument('--niters', type=int, default=100000,
                    help='Maximum number of iterations')
parser.add_argument('--epsilon', type=float, default=1e-5,
                    help='Value of difference when training will be stopped')
parser.add_argument('--refine',action='store_true', default=False,
                    help='If true, perform refinement experiment')
parser.add_argument('--refine_model', type=str, default=None,
                    help="if refine=True, the refined model is in ['ETS','Theta', 'Arima', 'ARNN','RDE']")
args = parser.parse_args()

if __name__ == '__main__':
    ### Training Set
    dataset = planktonDataset.load()
    training_set = dataset
    columns = dataset.columns.values
    training_values = training_set.to_numpy().T

    if not args.refine:
        #### Training model
        for target in args.target:
            training_set = Sampler(training_values, args.input_size, args.output_size, target)
            exp = pre_exp(target=target, in_feature=args.in_feature, input_size=args.input_size,
                        output_size = args.output_size, dataset=training_set, warm=True)
        
            # Cross validation
            hyper = product(args.lambda_1,args.lambda_2)
            best_par = exp.val(hyper,size=args.val_size)

            # Test
            nrmse, pccs = exp.test(best_par, size=args.test_size+args.val_size, save=f'N{target+1}')
            ave_loss = sum(nrmse[-args.test_size:])/args.test_size
            pcc = sum(pccs[-args.test_size:])/args.test_size
            print(f'Average loss of target {target} of plankton is {ave_loss} and pcc is {pcc}')

    else:
        assert args.refine_model in ['ETS', 'Theta', 'Arima', 'MVE', 'ARNN','RDE']
        for target in args.target:
            if args.refine_model == 'MVE':
                training_set = Sampler(training_values, args.input_size, args.output_size, target, train=False)
            else:
                training_set = Sampler(training_values, args.input_size, args.output_size, target)
            exp = refine_exp(target=target,in_feature=args.in_feature, input_size=args.input_size,
                        output_size = args.output_size, dataset=training_set, base_model=args.refine_model)
            
            ### refine
            nrmse, temp_nrmse = exp.test(size=args.test_size+args.val_size, save=f'N{target+1}')
            ref_loss = sum(nrmse[-args.test_size:])/args.test_size
            temp_loss = sum(temp_nrmse[-args.test_size:])/args.test_size
            print(f'target {target} of plankton: Model {args.refine_model}\nOriginal loss: {temp_loss:.4f}     |      refined loss: {ref_loss:.4f}')

    
