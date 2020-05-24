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

for lr in [0.05 ]:
	for times in [0.1]:
		dataset = 'cifar10'
		task_division = '9,1'
		epoch_cloud = 90 #max(50, int(int(task_division.split(',')[0])* 7))
		epoch_edge = 50

		command_tmp = 'python continual_learning.py --gpu 0 --seed 333 --score abs_w --num_epoch ' + str(epoch_cloud) +' --epoch_edge '+str(epoch_edge)+\
					  ' --dataset '+ dataset  +' --task_division ' + task_division + ' --lr ' + str(lr) + ' --times ' +str(times)
		print('command:\n', command_tmp)
		os.system(command_tmp)


		# i = int(task_division.split(',')[0]) if dataset == 'cifar10' else int(int(task_division.split(',')[0])/10)
		i += 1
		scipy.io.savemat('../../results/tuning_{}.mat'.format(i), {'i':i,  'epoch':epoch_cloud, 'task':args.task_division})


