import os
import sys
import time
import scipy.io as scio
import torch
import math
import functions
import utils
from args import parser
from make_architecture import *
from block_library_v1 import *
from dataload_regular import *
from dataload_continual import *
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
									  format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('../../results', 'log_search.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("******************************************************")
logging.info("                   NAS search_EA.py")
logging.info("args = %s", args)



def generate_pop(DNA_SIZE = args.DNA_SIZE, POP_SIZE = args.POP_SIZE):
	logging.info('\n-----generating population------')
	block_pop = []
	size_pop = []

	for idx in range(POP_SIZE):
		param = 5.1
		while param >= 4.0:  # disgard network that larger than 3M
			block_cfg, size_cfg, ds_cfg = make_cfg(DNA_SIZE)
			model = nas.initial_network(block_cfg, size_cfg, ds_cfg)
			param = utils.count_parameters_in_MB(nas.net)
			logging.info('param size = {0:.2f}MB'.format(param))
		block_pop.append(block_cfg)
		size_pop.append(size_cfg)
		logging.info('pop idx {} generated\n'.format(idx))

	logging.info('block_pop shape: POP {}, DNA {}'.format(len(block_pop), len(block_pop[0])))
	logging.info(' size_pop shape: POP {}, DNA {}'.format(len(size_pop), len(size_pop[0])))
	logging.info('ds_cfg {}'.format(ds_cfg))
	logging.info('Population block_pop {}'.format(block_pop))
	logging.info('Population size_pop {}\n'.format(size_pop))

	pop = {'block_pop': block_pop,
		   'size_pop': size_pop,
		   'ds_cfg': ds_cfg}
	return pop


def get_fitness(pop):
	logging.info('\n-----calculating fitness------')
	block_pop = list(pop['block_pop'])
	size_pop = list(pop['size_pop'])
	ds_cfg = list(pop['ds_cfg'])
	max_fitness = 0
	fitness = np.zeros((1, args.POP_SIZE))
	pop_accu = []
	pop_delta_accu = []
	for pop_id, block_cfg in enumerate(block_pop):
		size_cfg = size_pop[pop_id]
		model = nas.initial_network(block_cfg, size_cfg, ds_cfg)
		nas.initialization()
		param = utils.count_parameters_in_MB(nas.net)
		logging.info('\npop_id {} block_cfg {} size_cfg {}, param size = {:.2f}MB'.format(pop_id, block_cfg, size_cfg, param))

		train_acc = np.zeros([1, args.num_epoch])
		test_acc = np.zeros([1, args.num_epoch])
		accu_wNoise = np.zeros([1, args.num_epoch])

		for epoch in range(args.num_epoch):
			train_acc[0, epoch] = nas.train(epoch, trainloader)
			test_acc[0, epoch] = nas.test(testloader)
			delta_accu = 1.0
			logging.info('pop_id {} epoch {} lr {} ========== train_acc {:.3f} test_acc {:.3f}'.format(pop_id, epoch, nas.current_lr, train_acc[0, epoch], test_acc[0, epoch]))
			if train_acc[0, epoch] <= 0.12:
				fitness[0, pop_id] = 0.0001
				logging.info('Discard this pop since the accuracy is too low')
				break
			if epoch == args.num_epoch - 1:	# add noise and test
				logging.info('Add Noise within range -{:.3f}, {:.3f}'.format(args.alpha, args.alpha))
				nas.add_noise(args.alpha)
				accuracy_wNoise = nas.test(testloader)
				accu_wNoise[0, epoch] = accuracy_wNoise
				delta_accu = test_acc[0, epoch] - accuracy_wNoise
				scio.savemat('../../results/mode_{}/pop{}_epoch{}_alpha{}_Acc{:.3f}to{:.3f}_param{:.2f}MB.mat'.
							 format(args.mode, pop_id, epoch, args.alpha, test_acc[0, epoch], accuracy_wNoise, param),
							 {'alpha': args.alpha, 'test_acc': test_acc[0, epoch],
							  'accu_wNoise': accu_wNoise, 'train_acc':train_acc, 'test_acc':test_acc,
							  'block_cfg': block_cfg, 'size_cfg': size_cfg, 'ds_cfg':ds_cfg})
				if test_acc[0, epoch] > 0.85:
					torch.save(nas.net, '../../results/mode_{}/pop{}_epoch{}_alpha{}_Acc{:.3f}to{:.3f}_param{:.2f}MB'.
							 format(args.mode, pop_id, epoch, args.alpha, test_acc[0, epoch], accuracy_wNoise, param))
				logging.info('Accuracy changes {:.3f} -> {:.3f} after adding Gaussian noise with alpha={}'.format(test_acc[0, epoch], accuracy_wNoise, args.alpha))
				# load back previous weight without noise
				logging.info('Loading back weights without noise.....')
				nas.load_weight_back()
				if train_acc[0, epoch] <= 0.5 or param > 4.0:
					fitness[0, pop_id] = 0.0001
				else:
					if args.mode == 'continual':
						# fitness[0, pop_id] = abs(math.log(abs(delta_accu + 0.00001)))* pow(test_acc[0, epoch], 1)
						fitness[0, pop_id] = abs(1.0/(abs(delta_accu + 0.00001)))* pow(test_acc[0, epoch], 1)
					#
					elif args.mode == 'regular':
						fitness[0, pop_id] = test_acc[0, epoch]
						# fitness[0, pop_id] = abs(train_acc[0, epoch])# - test_acc[0, epoch])
						# fitness[0, pop_id] = abs(math.log(abs(delta_accu + 0.00001)))
				if fitness[0, pop_id] > max_fitness:
					max_fitness = fitness[0, pop_id]
					maxfit_accu = test_acc[0, epoch]
					maxfit_deltaAcc = delta_accu
					torch.save(nas.net, '../../results/model_with_best_fitness')
					scio.savemat('../../results/best_model_cfg.mat', {'block_cfg': block_cfg, 'size_cfg': size_cfg, 'ds_cfg': ds_cfg,
																	  'test_accu': test_acc[0, epoch], 'train_acc': train_acc[0, epoch],
																	  'delta_acc':delta_accu, 'accuracy_wNoise':accuracy_wNoise, 'param': param})


		logging.info('Pop idx {} delta_accu {:.3f} fitness {:.3f} Param {:.2f}MB'.format(pop_id, delta_accu, fitness[0, pop_id], param))
		torch.cuda.empty_cache()
		pop_accu.append(test_acc[0, epoch])
		pop_delta_accu.append(delta_accu)

	logging.info('fitness {} '.format(fitness))
	logging.info('pop_accu {} '.format(pop_accu))
	logging.info('pop_delta_accu {} '.format(pop_delta_accu))
	return fitness, pop_accu, pop_delta_accu, maxfit_accu, maxfit_deltaAcc


def select(pop, fitness):    # nature selection wrt pop's fitness
	logging.info('\n-----selecting wrt pops fitness------')
	sum_fit = np.sum(fitness[0, :])
	idx = np.random.choice(np.arange(args.POP_SIZE), size=args.POP_SIZE, replace=True,
						   p = fitness[0, :]/sum_fit)
	logging.info('selected idx %s', idx)
	block_pop = np.array(pop['block_pop'])[idx, :]
	size_pop = np.array(pop['size_pop'])[idx, :]
	new_pop = {'block_pop': block_pop,
			   'size_pop': size_pop,
			   'ds_cfg': np.array(pop['ds_cfg'])}
	logging.info('Population block_pop {}'.format(block_pop))
	logging.info('Population size_pop {}\n'.format(size_pop))
	return new_pop


def crossover(CROSS_RATE, block_parent, size_parent, pop_copy):     # mating process (genes crossover)
	new_block_pop = pop_copy['block_pop']
	new_size_pop = pop_copy['size_pop']
	if np.random.rand() < CROSS_RATE:
		logging.info('\n-----mating/crossover------')
		i_ = np.random.randint(0, args.POP_SIZE, size=1)     # select another individual from pop
		cross_points = np.random.randint(0, 2, size=new_block_pop.shape[1]).astype(np.bool)   # choose crossover points
		block_parent[cross_points] = new_block_pop[i_, cross_points]         # mating and produce one child
		cross_points = np.random.randint(0, 2, size=new_size_pop.shape[1]).astype(np.bool)   # choose crossover points
		size_parent[cross_points] = new_size_pop[i_, cross_points]         # mating and produce one child

	return block_parent, size_parent


def mutate(MUTATION_RATE, block_child, size_child):
	for point in range(len(block_child)):
		if np.random.rand() < MUTATION_RATE:
			logging.info('\n-----mutation------')
			block_child[point] = np.random.randint(0, num_block['plain'], size=1)
			size_child[point] = (2 ** np.random.randint(4, 10, size=1))
	return block_child, size_child




nas = functions.NAS()
num_block, pool_blocks, densenet_num = number_of_blocks()

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

time_start=time.time()
record_generation = np.zeros((args.N_GENERATIONS, 3, args.POP_SIZE))

pop = generate_pop()
for gen in range(args.N_GENERATIONS):
	logging.info('\n\nGeneration {}'. format(gen))
	pop_fitness, pop_accuracy, pop_delta_accu, maxfit_accu, maxfit_deltaAcc =  get_fitness(pop)
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
	logging.info('block_pop {}'.format(pop['block_pop']))
	logging.info(' size_pop {}'.format(pop['size_pop']))
	logging.info('   ds_cfg {}'.format(pop['ds_cfg']))
	torch.cuda.empty_cache()
	scio.savemat('../../results/mode_{}/generation{}.mat'.format(args.mode, gen ), {'pop': pop})


# Document
time_end=time.time()
hour = int(time_end - time_start) / 3600
logging.info('time cost {:.2f} second'.format(time_end - time_start))
logging.info('time cost {:.2f} hour'.format(hour))

scio.savemat('../../results/mode_{}/POPsize{}_DNA{}_maxfit_Accu{:.3f}_DeltaAccu{:.3f}.mat'.format(args.mode, args.POP_SIZE, args.DNA_SIZE, maxfit_accu, maxfit_deltaAcc),
	{'pop': pop, 'block_pop':pop['block_pop'],  'size_pop': pop['size_pop'], 'ds_cfg': pop['ds_cfg'],
	 'pop_fitness':pop_fitness, 'pop_accuracy': pop_accuracy, 'record_generation': record_generation,
	 'time_start':time_start, 'time_end':time_end, 'alpha':args.alpha,
	 'dataset':args.dataset, 'task_division': args.task_division, 'mode': args.mode})

