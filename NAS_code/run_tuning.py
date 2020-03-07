import os
import shutil
import scipy.io
import random
import numpy as np
import scipy.io as scio


if os.path.exists('../../results'):
    shutil.rmtree('../../results')
os.mkdir('../../results')

if os.path.exists('../../log_files'):
    shutil.rmtree('../../log_files')
os.mkdir('../../log_files')

i=0


for num_module in [3]:
	for max_block in [1, 3, 5]:
						
		command_tmp = 'python main.py --dataset cifar10 --num_epoch 90  --num_module ' + str(num_module) + ' --max_block '+str(max_block)
		
		print('command:\n', command_tmp)

		os.system(command_tmp)
		i = i+1
		scio.savemat('../../results/tuning_'+str(i)+'_finished.mat', {'num_module': num_module, 'max_block': max_block})



# --prune_thres 0.85