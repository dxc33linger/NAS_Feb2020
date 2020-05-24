import os
import shutil

import scipy.io as scio

if os.path.exists('../../results'):
    shutil.rmtree('../../results')
os.mkdir('../../results')


# if os.path.exists('../../loss-landscape/cifar10/trained_nets'):
#     shutil.rmtree('../../loss-landscape/cifar10/trained_nets')
# os.mkdir('../../loss-landscape/cifar10/trained_nets')
i=0

for POP_SIZE in [40]:
    for alpha in [0.5]:
        for DNA_SIZE in [3]: # 5 7 9
                for N_GENERATIONS in [5]:
                    for CROSS_RATE in [0.6]:

                        mode = 'continual'
                        if not os.path.exists('../../results/mode_{}'.format(mode)):
                            os.mkdir('../../results/mode_{}'.format(mode))

                        command_tmp = 'python search_EA.py --gpu 0 --seed 333 --dataset cifar10 --num_epoch 20 --task_division 9,1 --mode ' + str(mode) + \
                                      ' --DNA_SIZE '+str(DNA_SIZE) + ' --alpha '+str(alpha) + ' --POP_SIZE ' +str(POP_SIZE) +' --N_GENERATIONS ' +str(N_GENERATIONS) + ' --CROSS_RATE ' + str(CROSS_RATE)\


                        print('command:\n', command_tmp)

                        os.system(command_tmp)
                        i = i+1
                        scio.savemat('../../results/mode_{}/tuning_{}_finished.mat'.format(mode, i), {'alpha': alpha, 'DNA_SIZE': DNA_SIZE, 'POP_SIZE':POP_SIZE})


