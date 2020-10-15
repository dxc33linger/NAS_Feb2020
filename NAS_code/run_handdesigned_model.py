import os
import shutil
import scipy.io
from args import *

args = parser.parse_args()

if os.path.exists('../../mask_library'):
    shutil.rmtree('../../mask_library')
os.mkdir('../../mask_library')

i = 0
# for task_division in ['90,10', '80,10,10', '70,10,10,10', '60,10,10,10,10', '50,10,10,10,10,10', '10,10,10,10,10,10,10,10,10,10']:
# for task_division in ['1,1,1,1,1,1,1,1,1,1','5,1,1,1,1,1', '6,1,1,1,1', '7,1,1,1', '8,1,1', '9,1']:
# for task_division in ['1,1', '2,1', '3,1', '4,1', '5,1', '6,1', '7,1', '8,1', '9,1']:
# for task_division in ['90,10', '80,10', '70,10', '60,10', '50,10', '40,10', '30,10', '20,10', '10,10']:
for default_model in ['vgg16']:  # , 'vgg16', 'resnet20', 'resnet20_noshort', 'resnet56' ,'resnet56_noshort'
    for task_division in ['9,1','5,1,1,1,1,1']:
        lr = 0.1
        for times in [2.0]:
            dataset = 'cifar10'
            epoch_cloud = 90
            epoch_edge =  60

            command_tmp = 'python continual_learning.py --gpu 0 --seed 333 --score grad_w --num_epoch ' + str(
                epoch_cloud) + ' --epoch_edge ' + str(epoch_edge) + \
                          ' --dataset ' + dataset + ' --task_division ' + task_division + ' --lr ' + str(
                lr) + ' --times ' + str(times) \
                          + ' --default_model ' + default_model
            print('command:\n', command_tmp)
            os.system(command_tmp)

            # i = int(task_division.split(',')[0]) if dataset == 'cifar10' else int(int(task_division.split(',')[0])/10)
            i += 1
            scipy.io.savemat('../../results/tuning_{}.mat'.format(i),
                             {'i': i, 'epoch': epoch_cloud, 'task': args.task_division})
