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

for POP_SIZE in [60]:
    for alpha in [0.8]:
        for DNA_SIZE in [5]: # 5 7 9
                for N_GENERATIONS in [5]:
                    for CROSS_RATE in [0.8]:

                        mode = 'continual'
                        if not os.path.exists('../../results/mode_{}'.format(mode)):
                            os.mkdir('../../results/mode_{}'.format(mode))

                        command_tmp = 'python search_EA.py --gpu 1 --seed 333 --dataset cifar100 --num_epoch 20 --task_division 80,10,10 --mode ' + str(mode) + \
                                      ' --DNA_SIZE '+str(DNA_SIZE) + ' --alpha '+str(alpha) + ' --POP_SIZE ' +str(POP_SIZE) +' --N_GENERATIONS ' +str(N_GENERATIONS) + ' --CROSS_RATE ' + str(CROSS_RATE)\


                        print('command:\n', command_tmp)

                        os.system(command_tmp)
                        i = i+1
                        scio.savemat('../../results/mode_{}/tuning_{}_finished.mat'.format(mode, i), {'alpha': alpha, 'DNA_SIZE': DNA_SIZE, 'POP_SIZE':POP_SIZE})


