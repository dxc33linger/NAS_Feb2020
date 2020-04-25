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

for POP_SIZE in [4]:
    for alpha in [0.2]:
        for DNA_SIZE in [7]:
                for N_GENERATIONS in [2]:
                    for CROSS_RATE in [0.6]:


                        evaluation = 'accuracy'  # robustness
                        mode = 'regular'
                        if not os.path.exists('../../results/mode_{}'.format(mode)):
                            os.mkdir('../../results/mode_{}'.format(mode))

                        command_tmp = 'python search_EA.py  --dataset cifar10 --num_epoch 1 --task_division 10,10 --mode ' + str(mode) + \
                                      ' --DNA_SIZE '+str(DNA_SIZE) + ' --alpha '+str(alpha) + ' --POP_SIZE ' +str(POP_SIZE) +' --N_GENERATIONS ' +str(N_GENERATIONS) + ' --CROSS_RATE ' + str(CROSS_RATE)\
                                      + ' --evaluation ' + evaluation

                        print('command:\n', command_tmp)

                        os.system(command_tmp)
                        i = i+1
                        scio.savemat('../../results/mode_{}/tuning_{}_finished.mat'.format(mode, i), {'alpha': alpha, 'DNA_SIZE': DNA_SIZE, 'POP_SIZE':POP_SIZE})


