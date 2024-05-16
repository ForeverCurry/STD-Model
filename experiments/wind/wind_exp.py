import os
import argparse
from models.STD import pre_exp,refine_exp
from datasets.wind import WindDataset
from itertools import product
from common.sampler import Sampler
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

parser = argparse.ArgumentParser('weather experiment')
parser.add_argument('--lambda_1', type=list, default=[0.1,0.01,1],
                    help='Cross-validation parameter sets for lambda_1',)
parser.add_argument('--lambda_2', type=list, default=[0.05,0.5,1,10], 
                    help='Cross-validation parameter sets for lambda_1')
parser.add_argument('--in_feature', type=int, default=154,
                    help='Number of spatial dimension of data')
parser.add_argument('--input_size', type=float, default=48,
                    help='Number of training length')
parser.add_argument('--output_size', type=int, default=12,
                    help='Number of prediction steps')
parser.add_argument('--val_size', type=int, default=50,
                    help='Size of validation set')
parser.add_argument('--test_size', type=int, default=50,
                    help='Size of test set')
parser.add_argument('--target', type=int, default=24,
                    help='Index of target')
parser.add_argument('--niters', type=int, default=100000,
                    help='Maximum number of iterations')
parser.add_argument('--epsilon', type=float, default=1e-5,
                    help='Value of difference when training will be stopped')
parser.add_argument('--refine',action='store_true', default=False,
                    help='If true')
parser.add_argument('--refine_model', type=str, default=None,
                    help="if refine=True, the refined model is in ['ETS','Theta', 'Arima', 'RDE', 'ARNN']")
args = parser.parse_args()


if __name__ == '__main__':
    # Training Set
    dataset = WindDataset.load()
    training_set = dataset.to_numpy().T

    dict = {105:'Osaka', 24:'Fukushima'}
    if not args.refine:
        path = []
        for target in args.target:
            #### Load data
            training_set = Sampler(training_set, args.input_size, args.output_size, target=target)
            #### Training model
            exp = pre_exp(target=args.target,in_feature=args.in_feature, input_size=args.input_size,
                            output_size = args.output_size,dataset=training_set, warm=True)
            
            ### Cross validation
            hyper = product(args.lambda_1,args.lambda_2)
            best_par = exp.val(hyper,size=args.val_size)

            ### Test
            nrmse, pcc = exp.test(best_par, size=args.test_size+args.val_size, save=dict[args.target])
            ave_loss = sum(nrmse[-args.test_size:])/args.test_size
            print(f'Average loss of {dict[args.target]} is {ave_loss}')
            path.append(f'.\{dict[target]}\STD_{dict[target]}.csv')
        ### Plot results
        titles = ['Osaka wind speed','Fukushima wind speed']
        plot_result(path, args.input_size, args.output_size, args.test_size, [[5,25,35],[5,20,45]],
            titles,save_path='png\wind_result.pdf')
    else:
        assert args.refine_model in ['ETS', 'Theta','Arima', 'MVE','RDE', 'ARNN']\
        #### load data
        for target in args.target:
            if args.refine_model == 'MVE':
                training_set = Sampler(train_set, args.input_size, args.output_size, target=target, train=False)
            else:
                training_set = Sampler(train_set, args.input_size, args.output_size, target=target)
        exp = refine_exp(target=args.target,in_feature=args.in_feature, input_size=args.input_size,
                    output_size = args.output_size, dataset=training_set, base_model=args.refine_model)
        
        ### refine
        nrmse, temp_nrmse, ref_pcc, temp_pcc = exp.test(size=args.test_size+args.val_size, save=False)
        ref_loss = sum(nrmse[-args.test_size:])/args.test_size
        temp_loss = sum(temp_nrmse[-args.test_size:])/args.test_size
        print(f'{dict[args.target]}: Model {args.refine_model}\n Original loss: {temp_loss:.4f}    |      refined loss: {ref_loss:.4f}\n Original PCC: {temp_pcc:.4f}    |      refined PCC: {ref_pcc:.4f}')
    

