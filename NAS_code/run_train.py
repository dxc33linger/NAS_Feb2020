import os
import shutil
import scipy.io
import random
import numpy as np
import scipy.io as scio


if os.path.exists('../../results'):
    shutil.rmtree('../../results')
os.mkdir('../../results')

if os.path.exists('../../loss-landscape/cifar10/trained_nets'):
	shutil.rmtree('.. /.. / loss - landscape / cifar10 / trained_nets')
os.mkdir('.. /.. / loss - landscape / cifar10 / trained_nets')
i=0

for cfg in [1]:
	for alpha in [0.1, 0.25, 0.5, 1]:
		for mode in [ 'regular']:
			if not os.path.exists('../../results/mode_{}'.format(mode)):
				os.mkdir('../../results/mode_{}'.format(mode))

			cfg = 66
			command_tmp = 'python train.py  --dataset cifar10 --num_epoch 10 --mode ' + str(mode) + ' --cfg '+str(cfg) + ' --alpha '+str(alpha)

			print('command:\n', command_tmp)

			os.system(command_tmp)
			i = i+1
			scio.savemat('../../results/mode_{}/tuning_{}_finished.mat'.format(mode, i), {'alpha': alpha, 'cfg': cfg})



