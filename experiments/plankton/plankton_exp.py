import os
import argparse
from models.STD import STD, pre_exp, refine_exp
from itertools import product
from datasets.plankton import planktonDataset
from common.sampler import Sampler

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

parser = argparse.ArgumentParser('plankton experiments')
parser.add_argument('--lambda_1', type=list, default=[0.01,0.1,1],
                    help='Cross-validation parameter sets for lambda_1',)
parser.add_argument('--lambda_2', type=list, default=[0.01,0.05,0.1], 
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
parser.add_argument('--target', type=int, default=11,
                    help='Index of target')
parser.add_argument('--niters', type=int, default=100000,
                    help='Maximum number of iterations')
parser.add_argument('--epsilon', type=float, default=1e-5,
                    help='Value of difference when training will be stopped')
parser.add_argument('--refine',action='store_true', default=False,
                    help='If true')
parser.add_argument('--refine_model', type=str, default=None,
                    help="if refine=True, the refined model is in ['ETS','Theta', 'Arima', 'ARNN','RDE']")
args = parser.parse_args()

if __name__ == '__main__':
    ### Training Set
    dataset = planktonDataset.load()
    training_set = dataset
    columns = dataset.columns.values
    training_values = training_set.to_numpy().T
    training_set = Sampler(training_values, args.input_size, args.output_size, args.target,
                        pretrain=False)
    if not args.refine:
        #### Training model
        exp = pre_exp(model=STD,target=args.target,in_feature=args.in_feature, input_size=args.input_size,
                        output_size = args.output_size, dataset=training_set, warm=True)
        
        # Cross validation
        hyper = product(args.lambda_1,args.lambda_2)
        best_par = exp.val(hyper,size=args.val_size)
        # best_par=[0.01,0.1]
        # Test
        nrmse, pcc = exp.test(best_par, size=args.test_size+args.val_size, save=False)
        ave_loss = sum(nrmse[-args.test_size:])/args.test_size
        print(f'Average loss of target {args.target} of plankton is {ave_loss}')
    else:
        assert args.refine_model in ['ETS', 'Theta', 'Arima', 'ARNN','RDE']
        exp = refine_exp(target=args.target,in_feature=args.in_feature, input_size=args.input_size,
                    output_size = args.output_size, dataset=training_set, base_model=args.refine_model)
        
        ### refine
        nrmse, temp_nrmse, ref_pcc, temp_pcc = exp.test(size=args.test_size+args.val_size, save=False)
        ref_loss = sum(nrmse[-args.test_size:])/args.test_size
        temp_loss = sum(temp_nrmse[-args.test_size:])/args.test_size
        print(f'target {args.target} of plankton: Model {args.refine_model}\nOriginal loss: {temp_loss:.4f}     |      refined loss: {ref_loss:.4f}\n Original PCC: {temp_pcc:.4f}     |      refined PCC: {ref_pcc:.4f}')

    