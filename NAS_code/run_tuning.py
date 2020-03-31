import os
import shutil
import scipy.io
import random
import numpy as np
import scipy.io as scio


if os.path.exists('../../results'):
    shutil.rmtree('../../results')
os.mkdir('../../results')


i=0

for num_module in [3]:
	for cfg in [1,2,3]:
		for alpha in [0.01, 0.1, 1]:

			# if not os.path.exists('../../results/alpha{}_maxBlock{}'.format(alpha, cfg)):
			# 	os.mkdir('../../results/alpha{}_maxBlock{}'.format(alpha, cfg ))

			command_tmp = 'python main.py --mode continual --dataset cifar10 --num_epoch 80  --num_module ' + str(num_module) + ' --cfg '+str(cfg) + ' --alpha '+str(alpha)

			print('command:\n', command_tmp)

			os.system(command_tmp)
			i = i+1
			scio.savemat('../../results/tuning_{}_finished.mat'.format(i), {'num_module': num_module, 'cfg': cfg})



# --prune_thres 0.85