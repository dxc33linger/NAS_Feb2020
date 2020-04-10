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

for cfg in [44]:
	for alpha in [0.2]:
		for mode in ['continual', 'regular']:
			for experiment in [0, 1, 2, 3, 4]:
				if not os.path.exists('../../results/mode_{}'.format(mode)):
					os.mkdir('../../results/mode_{}'.format(mode))

				command_tmp = 'python train.py  --dataset cifar10 --num_epoch 2 --mode ' + str(mode) + ' --cfg '+str(cfg) + ' --alpha '+str(alpha)

				print('command:\n', command_tmp)

				os.system(command_tmp)
				i = i+1
				scio.savemat('../../results/mode_{}/tuning_{}_finished.mat'.format(mode, i), {'alpha': alpha, 'cfg': cfg})



# --prune_thres 0.85