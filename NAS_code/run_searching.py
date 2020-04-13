import os
import shutil
import scipy.io
import random
import numpy as np
import scipy.io as scio

if os.path.exists('../../results'):
    shutil.rmtree('../../results')
os.mkdir('../../results')

# if os.path.exists('../../loss-landscape/cifar10/trained_nets'):
#     shutil.rmtree('../../loss-landscape/cifar10/trained_nets')
# os.mkdir('../../loss-landscape/cifar10/trained_nets')
i=0

for POP_SIZE in [50]:
    for alpha in [0.2]:
        for mode in ['continual']:
            for DNA_SIZE in [10]:
                for N_GENERATIONS in [4]:

                    if not os.path.exists('../../results/mode_{}'.format(mode)):
                        os.mkdir('../../results/mode_{}'.format(mode))

                    command_tmp = 'python search_EA.py  --dataset cifar100 --num_epoch 5 --mode ' + str(mode) + \
                                  ' --DNA_SIZE '+str(DNA_SIZE) + ' --alpha '+str(alpha) + ' --POP_SIZE ' +str(POP_SIZE) +' --N_GENERATIONS ' +str(N_GENERATIONS)
                    print('command:\n', command_tmp)

                    os.system(command_tmp)
                    i = i+1
                    scio.savemat('../../results/mode_{}/tuning_{}_finished.mat'.format(mode, i), {'alpha': alpha, 'DNA_SIZE': DNA_SIZE, 'POP_SIZE':POP_SIZE})


