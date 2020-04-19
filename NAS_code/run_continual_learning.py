import os
import shutil
import scipy.io
from args import *
args = parser.parse_args()

i = 0
# for task_division in ['90,10', '80,10,10', '70,10,10,10', '60,10,10,10,10', '50,10,10,10,10,10', '10,10,10,10,10,10,10,10,10,10']:
# for task_division in ['1,1,1,1,1,1,1,1,1,1','5,1,1,1,1,1', '6,1,1,1,1', '7,1,1,1', '8,1,1', '9,1']:
# for task_division in ['1,1', '2,1', '3,1', '4,1', '5,1', '6,1', '7,1', '8,1', '9,1']:
# for task_division in ['90,10', '80,10', '70,10', '60,10', '50,10', '40,10', '30,10', '20,10', '10,10']:

for task_division in ['10,10,10,10,10,10,10,10,10,10']:
    i = 0
    dataset = 'cifar100'

    epoch_cloud =  max(40, int(int(task_division.split(',')[0])* 10))
    epoch_edge = max(50, int(int(task_division.split(',')[0])+int(task_division.split(',')[1])* 1.0))


    command_tmp = 'python continual_learning.py --num_epoch ' + str(epoch_cloud) +' --epoch_edge '+str(epoch_edge)+\
                  ' --dataset '+ dataset  +' --task_division ' + task_division
    print('command:\n', command_tmp)
    os.system(command_tmp)


    i = int(task_division.split(',')[0]) if dataset == 'cifar10' else int(int(task_division.split(',')[0])/10)

    scipy.io.savemat('../../results/tuning_{}.mat'.format(i), {'i':i,  'epoch':epoch_cloud, 'task':args.task_division})


