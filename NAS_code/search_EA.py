import os
import sys
import time

import scipy.io as scio
import torch

import functions
import utils
from args import parser
import make_architecture
from block_library import *
from dataload_regular import *
from dataload_continual import *
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
log_format = '%(asctime)s   %(message)s'
make_architecture.logging.basicConfig(stream=sys.stdout, level=make_architecture.logging.INFO,
									  format=log_format, datefmt='%m/%d %I:%M%p')
fh = make_architecture.logging.FileHandler(os.path.join('../../results', 'log.txt'))
fh.setFormatter(make_architecture.logging.Formatter(log_format))
make_architecture.logging.getLogger().addHandler(fh)
make_architecture.logging.info("******************************************************")
make_architecture.logging.info("                   NAS search_EA.py")
make_architecture.logging.info("args = %s", args)



def generate_pop(DNA_SIZE = args.DNA_SIZE, POP_SIZE = args.POP_SIZE):
	make_architecture.logging.info('\n-----generating population------')
	block_pop = []
	size_pop = []

	for idx in range(POP_SIZE):
		param = 5.1
		while param >= 4.0:  # disgard network that larger than 5M
			block_cfg, size_cfg, ds_cfg = make_architecture.make_cfg(DNA_SIZE)
			model = nas.initial_network(block_cfg, size_cfg, ds_cfg)
			param = utils.count_parameters_in_MB(nas.net)
			make_architecture.logging.info('param size = {0:.2f}MB'.format(param))
		block_pop.append(block_cfg)
		size_pop.append(size_cfg)
		make_architecture.logging.info('pop idx {} generated\n'.format(idx))

	make_architecture.logging.info('block_pop shape: POP {}, DNA {}'.format(len(block_pop), len(block_pop[0])))
	make_architecture.logging.info(' size_pop shape: POP {}, DNA {}'.format(len(size_pop), len(size_pop[0])))
	make_architecture.logging.info('ds_cfg {}'.format(ds_cfg))
	make_architecture.logging.info('Population block_pop {}'.format(block_pop))
	make_architecture.logging.info('Population size_pop {}\n'.format(size_pop))

	pop = {'block_pop': block_pop,
		   'size_pop': size_pop,
		   'ds_cfg': ds_cfg}
	return pop


def get_fitness(pop):
	make_architecture.logging.info('\n-----calculating fitness------')
	block_pop = list(pop['block_pop'])
	size_pop = list(pop['size_pop'])
	ds_cfg = list(pop['ds_cfg'])
	max_fitness = 0
	fitness = np.zeros((1, make_architecture.args.POP_SIZE))
	pop_accu = []
	pop_delta_accu = []
	for pop_id, block_cfg in enumerate(block_pop):
		size_cfg = size_pop[pop_id]
		model = nas.initial_network(block_cfg, size_cfg, ds_cfg)
		nas.initialization()
		param = utils.count_parameters_in_MB(nas.net)
		make_architecture.logging.info('\npop_id {} block_cfg {} size_cfg {}, param size = {:.2f}MB'.format(pop_id, block_cfg, size_cfg, param))

		train_acc = np.zeros([1, make_architecture.args.num_epoch])
		test_acc = np.zeros([1, make_architecture.args.num_epoch])
		accu_wNoise = np.zeros([1, make_architecture.args.num_epoch])

		for epoch in range(make_architecture.args.num_epoch):
			train_acc[0, epoch] = nas.train(epoch, trainloader)
			test_acc[0, epoch] = nas.test(testloader)
			delta_accu = 1.0
			make_architecture.logging.info('pop_id {} epoch {} lr {} ========== train_acc {:.3f} test_acc {:.3f}'.format(pop_id, epoch, nas.current_lr, train_acc[0, epoch], test_acc[0, epoch]))
			if train_acc[0, epoch] <= 0.2:
				fitness[0, pop_id] = 0.001
				make_architecture.logging.info('Disgard this pop since the accuracy is too low')
				break
			if epoch == 4:	# add noise and test
				make_architecture.logging.info('Add Noise within range -{:.3f}, {:.3f}'.format(make_architecture.args.alpha, make_architecture.args.alpha))
				nas.add_noise(make_architecture.args.alpha)
				accuracy_wNoise = nas.test(testloader)
				accu_wNoise[0, epoch] = accuracy_wNoise
				delta_accu = test_acc[0, epoch] - accuracy_wNoise
				scio.savemat('../../results/mode_{}/pop{}_epoch{}_alpha{}_cfg{}_Acc{:.3f}to{:.3f}_param{:.2f}MB.mat'.format(make_architecture.args.mode, pop_id, epoch, make_architecture.args.alpha, block_cfg, test_acc[0, epoch], accuracy_wNoise, param),
							 {'alpha': make_architecture.args.alpha, 'test_acc': test_acc[0, epoch], 'accu_wNoise': accu_wNoise, 'train_acc':train_acc, 'test_acc':test_acc})
				make_architecture.logging.info('Accuracy changes {:.3f} -> {:.3f} after adding Gaussian noise with alpha={}'.format(test_acc[0, epoch], accuracy_wNoise, make_architecture.args.alpha))
				# load back previous weight without noise
				make_architecture.logging.info('Loading back weights without noise.....')
				nas.load_weight_back()
				if train_acc[0, epoch] <= 0.3:
					fitness[0, pop_id] = 0.001
				else:
					if make_architecture.args.evaluation == 'robustness':
						fitness[0, pop_id] = abs(1./ (delta_accu+0.001)) #* pow(test_acc[0, epoch], 2)
					elif make_architecture.args.evaluation == 'accuracy':
						fitness[0, pop_id] = 1./(1. - test_acc[0, epoch])

				if fitness[0, pop_id] > max_fitness:
					max_fitness = fitness[0, pop_id]
					torch.save(nas.net, '../../results/model_with_best_fitness')
					scio.savemat('../../results/best_model_cfg.mat', {'block_cfg': block_cfg, 'size_cfg': size_cfg, 'ds_cfg': ds_cfg})
		make_architecture.logging.info('Pop idx {} delta_accu {:.3f} fitness {:.3f} Param {:.2f}MB'.format(pop_id, delta_accu, fitness[0, pop_id], param))
		torch.cuda.empty_cache()
		pop_accu.append(test_acc[0, epoch])
		pop_delta_accu.append(delta_accu)

	make_architecture.logging.info('fitness {} '.format(fitness))
	make_architecture.logging.info('pop_accu {} '.format(pop_accu))
	make_architecture.logging.info('pop_delta_accu {} '.format(pop_delta_accu))
	return fitness, pop_accu, pop_delta_accu


def select(pop, fitness):    # nature selection wrt pop's fitness
	make_architecture.logging.info('\n-----selecting wrt pops fitness------')
	sum_fit = np.sum(fitness[0, :])
	idx = np.random.choice(np.arange(make_architecture.args.POP_SIZE), size=make_architecture.args.POP_SIZE, replace=True,
						   p = fitness[0, :]/sum_fit)
	make_architecture.logging.info('selected idx %s', idx)
	block_pop = np.array(pop['block_pop'])[idx, :]
	size_pop = np.array(pop['size_pop'])[idx, :]
	new_pop = {'block_pop': block_pop,
			   'size_pop': size_pop,
			   'ds_cfg': np.array(pop['ds_cfg'])}
	make_architecture.logging.info('Population block_pop {}'.format(block_pop))
	make_architecture.logging.info('Population size_pop {}\n'.format(size_pop))
	return new_pop


def crossover(CROSS_RATE, block_parent, size_parent, pop_copy):     # mating process (genes crossover)
	new_block_pop = pop_copy['block_pop']
	new_size_pop = pop_copy['size_pop']
	if np.random.rand() < CROSS_RATE:
		make_architecture.logging.info('\n-----mating/crossover------')
		i_ = np.random.randint(0, make_architecture.args.POP_SIZE, size=1)     # select another individual from pop
		cross_points = np.random.randint(0, 2, size=new_block_pop.shape[1]).astype(np.bool)   # choose crossover points
		block_parent[cross_points] = new_block_pop[i_, cross_points]         # mating and produce one child
		cross_points = np.random.randint(0, 2, size=new_size_pop.shape[1]).astype(np.bool)   # choose crossover points
		size_parent[cross_points] = new_size_pop[i_, cross_points]         # mating and produce one child

	return block_parent, size_parent


def mutate(MUTATION_RATE, block_child, size_child):
	for point in range(len(block_child)):
		if np.random.rand() < MUTATION_RATE:
			make_architecture.logging.info('\n-----mutation------')
			block_child[point] = np.random.randint(0, num_block['plain'], size=1)
			size_child[point] = (2 ** np.random.randint(4, 10, size=1))
	return block_child, size_child




nas = functions.NAS()
num_block, pool_blocks, densenet_num = number_of_blocks()

make_architecture.logging.info('Current dataset mode: %s', args.mode)
if args.mode == 'regular': # full dataset
	trainloader, testloader = dataload()

elif args.mode == 'continual':  ## continual learning
	task_list, total_num_task = nas.create_task()
	make_architecture.logging.info('Task list %s: ', task_list)
	task_division = list(map(int, args.task_division.split(",")))
	cloud_list = task_list[0: task_division[0]]
	trainloader, testloader = dataload_partial(cloud_list, 0)
	for batch_idx, (data, target) in enumerate(trainloader):
		make_architecture.logging.info('CLOUD re-assigned label: %s\n', np.unique(target))
		break

time_start=time.time()
record_generation = np.zeros((args.N_GENERATIONS, 3, args.POP_SIZE))

pop = generate_pop()
for gen in range(args.N_GENERATIONS):
	make_architecture.logging.info('\n\nGeneration {}'. format(gen))
	pop_fitness, pop_accuracy, pop_delta_accu =  get_fitness(pop)
	record_generation[gen, 0, :] = pop_fitness
	record_generation[gen, 1, :] = pop_accuracy
	record_generation[gen, 2, :] = pop_delta_accu

	pop  = select(pop, pop_fitness)
	pop_copy = pop.copy()
	size_pop= pop['size_pop']

	for i, block_parent in enumerate(pop['block_pop']):
		block_child, size_child = crossover(args.CROSS_RATE, block_parent, size_pop[i], pop_copy)
		block_child, size_child = mutate(args.MUTATION_RATE, block_child, size_child)
		block_parent, size_pop[i] = block_child, size_child
		size_pop[i][0] = 16
		for idx, block in enumerate(block_parent):
			if block in pool_blocks:  ## if block is pooling layer, size is same with last layer
				size_pop[i][idx + 1] = size_pop[i][idx]
			elif block in densenet_num:  # non-DB - DB
				size_pop[i][idx + 1] = 24
		size_pop[i][0] = 16

	for key, val in pop.items():
		pop[key] = val.tolist()
	make_architecture.logging.info('block_pop {}'.format(pop['block_pop']))
	make_architecture.logging.info(' size_pop {}'.format(pop['size_pop']))
	make_architecture.logging.info('   ds_cfg {}'.format(pop['ds_cfg']))
	torch.cuda.empty_cache()
	scio.savemat('../../results/mode_{}/generation{}.mat'.format(args.mode, gen ), {'pop': pop})


# Document
time_end=time.time()
make_architecture.logging.info('time cost {}'.format(time_end - time_start))
scio.savemat('../../results/mode_{}/POPsize{}_DNA{}_maxAccu{:.3f}_minDeltaAccu{:.3f}.mat'.format(args.mode, args.POP_SIZE, args.DNA_SIZE, max(pop_accuracy), min(pop_delta_accu)),
	{'pop': pop, 'block_pop':pop['block_pop'],  'size_pop': pop['size_pop'], 'ds_cfg': pop['ds_cfg'],
	 'pop_fitness':pop_fitness, 'pop_accuracy': pop_accuracy, 'record_generation': record_generation,
	 'time_start':time_start, 'time_end':time_end, 'alpha':args.alpha,
	 'dataset':args.dataset, 'task_division': args.task_division, 'mode': args.mode})

