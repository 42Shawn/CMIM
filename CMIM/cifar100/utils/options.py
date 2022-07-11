import argparse
import os
"""
args
"""

parser = argparse.ArgumentParser(description='CMIM')

# Logging
parser.add_argument(
    '--results_dir',
    metavar='RESULTS_DIR',
    default='./results',
    help='results dir')

parser.add_argument(
    '--save',
    metavar='SAVE',
    default='20220711v0',
    help='saved folder (named by datetime)')

parser.add_argument(
    '--resume',
    dest='resume',
    action='store_true',
    help='resume to latest checkpoint')

parser.add_argument(
    '-e',
    '--evaluate',
    type=str,
    metavar='FILE',
    help='evaluate model FILE on validation set')

parser.add_argument(
    '--seed', 
    default= None,
    type=int, 
    help='random seed')

# Model
parser.add_argument(
    '--model',
    '-a',
    metavar='MODEL',
    default='resnet18_1w1a',
    help='model architecture ')

parser.add_argument(
    '--model_fp',
    default='resnet18_fp')

parser.add_argument(
    '--model_fp_dir',
    default='../cifar100/models_bnn/model_fp.pth.tar',
    help='dir of fp net. Download the checkpoint at https://drive.google.com/file/d/1y6Z11jGdO9SNHM8Wh6YAcdjGkngyJs_9/view?usp=sharing and put the downloaded file in the default dir ')

parser.add_argument(
    '--dataset',
    default='cifar100',
    type=str,
    help='dataset, default:cifar100')

parser.add_argument(
    '--data_path',
    type=str,
    default='/home/shangyuzhang/data',
    help='The dictionary where the dataset is stored.')

parser.add_argument(
    '--type',
    default='torch.cuda.FloatTensor',
    help='type of tensor - e.g torch.cuda.FloatTensor')

# Training
parser.add_argument(
    '--gpus',
    default='2',
    help='gpus used for training - e.g 0,1,3')

parser.add_argument(
    '--alpha',
    default=3.2,
    type=float,
    help='alpha in loss function')

parser.add_argument(
    '--beta',
    default=0.1,
    type=float,
    help='beta in loss function')

parser.add_argument(
    '--lr', 
    default=0.1, 
    type=float, 
    help='learning rate')

parser.add_argument(
    '--weight_decay',
    type=float,
    default=1e-4,
    help='Weight decay of loss. default:1e-4')

parser.add_argument(
    '--momentum',
    default=0.9, 
    type=float, 
    metavar='M',
    help='momentum')

parser.add_argument(
    '--workers',
    default=4,
    type=int,
    metavar='N',
    help='number of data loading workers (default: 8)')

parser.add_argument(
    '--epochs',
    default=400,
    type=int,
    metavar='N',
    help='number of total epochs to run')

parser.add_argument(
    '--start_epoch',
    default=-1,
    type=int,
    metavar='N',
    help='manual epoch number (useful on restarts)')

parser.add_argument(
    '-b',
    '--batch_size',
    default=100,
    type=int,
    metavar='N',
    help='mini-batch size for training (default: 256)')

parser.add_argument(
    '-bt',
    '--batch_size_test',
    default=256,
    type=int,
    help='mini-batch size for testing (default: 128)')

parser.add_argument(
    '--print_freq',
    '-p',
    default=100,
    type=int,
    metavar='N',
    help='print frequency (default: 100)')

parser.add_argument(
    '--time_estimate',
    default=1,
    type=int,
    metavar='N',
    help='print estimating finish time,set to 0 to disable')

parser.add_argument(
    '--rotation_update',
    default=1,
    type=int,
    metavar='N',
    help='interval of updating rotation matrix (default:1)')

parser.add_argument(
    '--Tmin',
    default=1e-2, 
    type=float, 
    metavar='M',
    help='minimum of T (default:1e-2)')

parser.add_argument(
    '--Tmax',
    default=1e1, 
    type=float, 
    metavar='M',
    help='maximum of T (default:1e1)')

parser.add_argument(
    '--lr_type',
    type=str,
    default='cos',
    help='choose lr_scheduler,(default:cos)')

parser.add_argument(
    '--lr_decay_step',
    nargs='+',
    type=int,
    help='lr decay step for MultiStepLR')

parser.add_argument(
    '--a32',
    dest='a32',
    action='store_true',
    help='w1a32')

parser.add_argument(
    '--warm_up',
    dest='warm_up',
    action='store_true',
    help='use warm up or not')

parser.add_argument(
    '--feat_dim',
    type=int,
    default=128,
    help='dimension of the projection space')

parser.add_argument(
    '--nce_n',
    type=int,
    default=4096*1,
    help='number of negatives paired with each positive')

parser.add_argument(
    '--nce_t',
    type=float,
    default=0.1,
    help='temperature parameter')

parser.add_argument(
    '--nce_mom',
    type=float,
    default=0.5,
    help='momentum for non-parametric updates')

args = parser.parse_args()
