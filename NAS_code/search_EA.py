
import os
import sys
import utils
import logging
import functions
import numpy as np
from block_library import *
from dataload_regular import *
from dataload_continual import *
from make_architecture import *
import scipy.io as scio
import torch
import matplotlib.pyplot as plt

from args import parser
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
	format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('../../results', 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("\n\n\n - - - - - NAS search_EA.py - - - - - - - ")
logging.info("args = %s", args)


nas = functions.NAS()

logging.info('Current dataset mode: %s', args.mode)
if args.mode == 'regular': # full dataset
	trainloader, testloader = dataload()

elif args.mode == 'continual':  ## continual learning
	task_list, total_num_task = nas.create_task()
	logging.info('Task list %s: ', task_list)
	task_division = list(map(int, args.task_division.split(",")))
	cloud_list = task_list[0: task_division[0]]
	trainloader, testloader = dataload_partial(cloud_list, 0)
	for batch_idx, (data, target) in enumerate(trainloader):
		logging.info('CLOUD re-assigned label: %s\n', np.unique(target))
		break

DNA_SIZE = 10            # DNA length
POP_SIZE = 50           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 200
X_BOUND = [0, 5]         # x upper and lower bounds

def generate_pop(DNA_SIZE, POP_SIZE):
	block_pop = np.zeros((POP_SIZE, DNA_SIZE*3+2))
	size_pop = np.zeros((POP_SIZE, DNA_SIZE*3+3))
	ds_pop = np.zeros((POP_SIZE, DNA_SIZE*3+2))

	for idx in range(POP_SIZE):
		param = 5.1
		while param >= 5.0:  # disgard network that larger than 5M
			block_cfg, size_cfg, ds_cfg = make_cfg(DNA_SIZE=10)
			model = nas.initial_network(block_cfg, size_cfg, ds_cfg)
			param = utils.count_parameters_in_MB(nas.net)
			logging.info('param size = {0:.2f}MB'.format(param))

		block_pop[idx] = block_cfg
		size_pop[idx] = size_cfg
		ds_pop[idx] = ds_cfg
		logging.info('block_pop = {} shape: {}'.format(block_pop, block_pop.shape))
		logging.info(' size_pop = {} shape: {}'.format(size_pop, size_pop.shape))
		logging.info('   ds_pop = {} shape: {}\n'.format(ds_pop, ds_pop.shape))


def select(pop, fitness):    # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness/fitness.sum())
    return pop[idx]

# def get_fitness(pred):
generate_pop(DNA_SIZE, POP_SIZE)


model = nas.initial_network(block_cfg, size_cfg, ds_cfg)
nas.initialization()
param = utils.count_parameters_in_MB(nas.net)
logging.info('block_cfg = {} length: {}'.format(block_cfg, len(block_cfg)))
logging.info(' size_cfg = {} length: {}'.format(size_cfg, len(size_cfg)))
logging.info('   ds_cfg = {} length: {}\n'.format(ds_cfg, len(ds_cfg)))
logging.info('param size = {0:.2f}MB'.format(param))


train_acc = np.zeros([1, args.num_epoch])
test_acc = np.zeros([1, args.num_epoch])
accu_wNoise = np.zeros([1, args.num_epoch])

for epoch in range(args.num_epoch):
	train_acc[0, epoch] = nas.train(epoch, trainloader)
	test_acc[0, epoch] = nas.test(testloader)
	logging.info('epoch {0} lr {1} ========== train_acc {2:.4f} test_acc {3:.4f}'.format(
		epoch, nas.current_lr, train_acc[0, epoch], test_acc[0, epoch]))
	nas.check_weight()
	if epoch % 5 == 0 or epoch % 3 == 0 or epoch == args.num_epoch - 1:
		# add noise and test
		logging.info('Add Noise within range -{:.5f}, {:.5f}'.format(args.alpha, args.alpha))
		model_wNoise = nas.add_noise(args.alpha)
		accuracy_wNoise = nas.test(testloader)
		logging.info('Accuracy changes {:.4f} -> {:.4f} after adding Gaussian noise with alpha={}'.format(test_acc[0, epoch], accuracy_wNoise, args.alpha))
		accu_wNoise[0, epoch] = accuracy_wNoise
		scio.savemat('../../results/mode_{}/epoch{}_alpha{}_cfg{}_Acc{:.4f}to{:.4f}.mat'.format(args.mode, epoch, args.alpha, block_cfg, test_acc[0, epoch], accuracy_wNoise),
					 {'alpha': args.alpha, 'test_acc':test_acc[0, epoch], 'accuracy_wNoise': accuracy_wNoise})
		# load back previous weight without noise
		logging.info('Loading back weights without noise.....')
		nas.load_weight_back()
		print('Is accuracy the same?', nas.test(testloader))
