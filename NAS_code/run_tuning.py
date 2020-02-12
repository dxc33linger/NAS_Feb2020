import os
import shutil
import scipy.io
import random
import numpy as np
import scipy.io as scio


if os.path.exists('../results'):
    shutil.rmtree('../results')
os.mkdir('../results')

if os.path.exists('../log_files'):
    shutil.rmtree('../log_files')
os.mkdir('../log_files')

i=0


for seed in [3, 33, 333]:   
	for max_block in [1, 3, 5, 7, 9]:
						
		command_tmp = 'python main.py --dataset cifar10 --num_epoch 150  --seed ' + str(seed) + ' --max_block '+str(max_block)
		
		print('command:\n', command_tmp)

		os.system(command_tmp)
		i=i+1
		scio.savemat('../results/tuning_'+str(i)+'_finished.mat', {'seed': seed})



# --prune_thres 0.85