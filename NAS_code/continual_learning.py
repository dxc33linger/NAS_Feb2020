import logging
import os
import pickle
import sys
import scipy.io as scio
import functions_continual
import functions
from dataload_continual import *
from dataload_regular import *
from utils import *
import matplotlib.pyplot as plt

from args import parser
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu



log_path = 'log_main_edge.txt'.format()
log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
	format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('../../results',log_path))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("*************************************************************************************************")
logging.info("                                        continual_learning.py                                             ")
logging.info("args = %s", args)



random.seed(args.seed)
nas = functions.NAS()
logging.info('Current dataset mode: %s', args.mode)

num_classes, in_feature = num_classes_in_feature()

if args.mode == 'regular': # full dataset
	raise ValueError('This function is designed for continual learning')
elif args.mode == 'continual':  ## continual learning
	task_list, total_num_task = nas.create_task()
	logging.info('Task list %s: ', task_list)
	task_division = list(map(int, args.task_division.split(",")))
	cloud_list = task_list[0: task_division[0]]
	train_cloud, test_cloud = dataload_partial(cloud_list, 0)
	total_task = len(task_division)
	all_data_list = []
	all_data_list.append(cloud_list)

for batch_idx, (data, target) in enumerate(train_cloud):
	logging.info('CLOUD re-assigned label: %s\n', np.unique(target))
	break
for batch_idx, (data, target) in enumerate(test_cloud):
	logging.info('Batch: {} CLOUD re-assigned test label: {}\n'.format(batch_idx, np.unique(target)))
	break


file_name = '../../results/best_model_cfg.mat'
content = scio.loadmat(file_name)
block_cfg = content['block_cfg'][0].tolist()
size_cfg = content['size_cfg'][0].tolist()
ds_cfg = content['ds_cfg'][0].tolist()
nas.net = torch.load('../../results/model_with_best_fitness')
model = nas.net
logging.info('net', model)
param = count_parameters_in_MB(model)
logging.info('param size = {0:.2f}MB'.format(param))

method = functions_continual.ContinualNN(model)

logging.info("==================================  Train task 0 ==========================================")
"""Test data from the first task"""

train_acc = []
test_acc_0 = []
test_acc_current = []
test_acc_mix = []
test_task_accu = []  # At the end of each task, best overall test accuracy. Length = number of tasks
test_acc_0_end = []  # At the end of each task, the accuracy of task 0. Length = number of tasks

best_acc_0 = 0.0

nas.initialization(args.lr, args.num_epoch * 0.4)
for epoch in range(args.num_epoch):
	train_acc.append(nas.train(epoch, train_cloud))
	test_acc_0.append(nas.test(test_cloud))
	test_acc_current.append(np.zeros(1))
	test_acc_mix.append(np.zeros(1))

	if test_acc_0[-1] > best_acc_0:
		best_acc_0 = test_acc_0[-1]
	logging.info('>>>>>>>>>> Single-head This training on T0 testing accu is : {:.4f}'.format(test_acc_0[-1]))
	logging.info('>>>>>>>>>> epoch {} train_acc {:.4f}\n\n\n'.format(epoch, train_acc[-1]))
#
ratio = task_division[0] / sum(task_division)
print(task_division[0], '/', sum(task_division), '=')
logging.info('ratio: {}'.format(ratio))
current_mask_list, current_threshold_dict, mask_dict_pre, maskR_dict_pre, current_taylor_dict = method.sensitivity_rank_taylor_filter(ratio)
with open('../../mask_library/mask_task{}_threshold{}_acc{:.4f}.pickle'.format(0, ratio, best_acc_0), "wb") as f:
	pickle.dump((current_mask_list, current_threshold_dict, mask_dict_pre, maskR_dict_pre, current_taylor_dict), f)

test_task_accu.append(best_acc_0)
test_acc_0_end.append(best_acc_0)
torch.save(nas.net.state_dict(), '../../results/model_afterT{}_Accu{:.4f}_param{:.2f}M.pt'.format(0, best_acc_0, param))


for task_id in range(1, total_task):
	logging.info("================================== 1. Current Task is {} : Prepare dataset ==========================================".format(task_id))
	# -----------------------------------------
	# Prepare dataset
	# -----------------------------------------
	total = 0
	for i in range(task_id+1):
		total += task_division[i]
	current_edge_list = task_list[(total-task_division[task_id]) : total]
	all_list = task_list[0 : total]
	all_data_list.append(current_edge_list)
	alltask_memory = []
	for i in range(len(task_division)):
		alltask_memory.append(int(task_division[i] * args.total_memory_size / num_classes))

	logging.info('alltask_memory =  %s', alltask_memory)
	train_bm, _ = get_partial_dataset_cifar(0, all_data_list, num_images = alltask_memory)
	logging.info('all_data_list %s',all_data_list)

	for batch_idx, (data, target) in enumerate(train_bm):
		logging.info('batch {} train_bm (balanced memory) re-assigned training label: {}\n'.format(batch_idx, np.unique(target)))
		break


	# """Balance memory: same amounts of images from previous tasks and current task"""
	# memory_each_task = int(args.total_memory_size / task_id) # The previous tasks shares the memory
	# alltask_list = []
	# alltask_memory = []
	# alltask_single_list = []
	# for i in range(task_id+1):
	# 	alltask_list.append(task_list[i])
	# 	alltask_memory.append(memory_each_task)
	# 	alltask_single_list += task_list[i]
	# train_bm, _ = get_partial_dataset_cifar(0, alltask_list, num_images = alltask_memory)
	# logging.info('alltask_memory =  %s', alltask_memory)
	#
	# for batch_idx, (data, target) in enumerate(train_bm):
	# 	print('train_bm (balanced memory) re-assigned label: \n', np.unique(target))
	# 	break



	train_edge, test_edge = dataload_partial(current_edge_list, task_division[0]+ (task_id-1)*task_division[task_id] )
	for batch_idx, (data, target) in enumerate(train_edge):
		logging.info('Batch: {} EDGE re-assigned train label: {}\n'.format(batch_idx, np.unique(target)))
		break
	for batch_idx, (data, target) in enumerate(test_edge):
		logging.info('Batch: {} EDGE re-assigned test label: {}\n'.format(batch_idx, np.unique(target)))
		break


	_, test_mix_full = dataload_partial(all_list, 0)
	for batch_idx, (data, target) in enumerate(test_mix_full):
		logging.info('Batch: {} ALL re-assigned testing label: {}\n'.format(batch_idx, np.unique(target)))
		break

	logging.info("=============================== 2. Current Task is {} : Memory-assisted balancing ==================================".format(task_id))
	nas.initialization(args.lr*args.times, int(0.4*args.epoch_edge))
	best_acc_mix = 0.0
	for epoch in range(args.epoch_edge):
		train_acc.append(method.train_with_frozen_filter(epoch, train_bm, mask_dict_pre, maskR_dict_pre))

		# if epoch % 7 == 0:
		# 	train_acc[-1] = method.train_with_frozen_filter(epoch, train_edge, mask_dict_pre, maskR_dict_pre)

		test_acc_0.append(nas.test(test_cloud))
		test_acc_current.append(nas.test(test_edge))
		test_acc_mix.append(nas.test(test_mix_full))
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head T0 testing accu is : {:.4f}'.format( test_acc_0[-1]))
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head current testing accu is : {:.4f}'.format( test_acc_current[-1]))
		logging.info('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Single-head mixed all tasks testing accu is : {:.4f}'.format( test_acc_mix[-1]))
		logging.info('train_acc {0:.4f} \n\n\n'.format(train_acc[-1]))
		if epoch >= int(0.9*args.epoch_edge) and test_acc_mix[-1] > best_acc_mix:
			best_acc_mix = test_acc_mix[-1]


	test_task_accu.append(best_acc_mix)
	test_acc_0_end.append(test_acc_0[-1])

	logging.info('>>>>>>>>>> At the end of task {}, T0 accu is {:.4f}'.format(task_id, test_acc_0[-1]))
	torch.save(nas.net.state_dict(), '../../results/model_afterT{}_Accu{:.4f}_param{:.2f}M.pt'.format(task_id, best_acc_mix, param))
	#
	ratio = task_division[task_id] / sum(task_division)
	logging.info('ratio: {}'.format(ratio))

	if task_id != total_task - 1:
		logging.info("===================================== 3.  Current Task is {} : importance sampling ====================================".format(task_id))
		method.mask_frozen_weight(maskR_dict_pre)

		current_mask_list, current_threshold_dict, mask_dict_current, maskR_dict_current, current_taylor_dict = method.sensitivity_rank_taylor_filter(
			ratio)
		with open('../../mask_library/mask_task{}_threshold{}_acc{:.4f}.pickle'.format(task_id, ratio, best_acc_mix), "wb") as f:
			pickle.dump((current_mask_list, current_threshold_dict, mask_dict_current, maskR_dict_current, current_taylor_dict, mask_dict_pre, maskR_dict_pre), f)

		logging.info("Current Task is {} : Combine masks  ==========================================".format(task_id))
		mask_dict_pre, maskR_dict_pre = method.AND_twomasks(mask_dict_pre, mask_dict_current, maskR_dict_pre, maskR_dict_current)



## RESULTS DOCUMENTATION
logging.info("====================== Document results ======================")

title_font = { 'size':'8', 'color':'black', 'weight':'normal'} # Bottom vertical alignment for more space
axis_font = { 'size':'10'}
plt.figure()
x = np.linspace(task_division[0], num_classes, num = len(test_task_accu))
plt.xlim(0, num_classes)
plt.xlabel('Task ID')
plt.ylabel('Accuracy')
plt.plot(x, test_task_accu , 'g-o', alpha=1.0, label = 'our method')
plt.yticks(np.arange(0, 1.0, step=0.1))
plt.xticks(np.arange(0, num_classes+1, step= 10))
plt.legend(loc='best')
plt.savefig('../../results/cfg_{}_incremental_curve_T{}_{:.4f}.png'.format(block_cfg, task_id, best_acc_mix))
plt.title('Task: {} \n Memory: {}\n Epoch_edge: {} ModelSize: {}MB'.format(task_division, alltask_memory, args.epoch_edge, param), **title_font)

x = np.linspace(0, len(test_acc_mix), len(test_acc_mix))
plt.figure(figsize=(20,10))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(x, train_acc, 'k', alpha=0.5, label = 'Training accuracy')
plt.plot(x, test_acc_0, 'r',  alpha=0.5, label = 'Testing accuracy - cloud')
plt.plot(x, test_acc_current, 'g',  alpha=0.5, label = 'Testing accuracy - edge')
plt.plot(x, test_acc_mix, 'b',  alpha=0.5, label = 'Testing accuracy - mix')
plt.yticks(np.arange(0, 1.0, step=0.1))
plt.xticks(np.arange(0, len(test_acc_mix), step=10))
plt.grid(color='b', linestyle='-', linewidth=0.1)
plt.legend(loc='best')
plt.title('Learning curve')
plt.savefig('../../results/PSTmain_learning_curve_cfg_{}_acc{:.4f}.png'.format(block_cfg, best_acc_mix))

logging.info('param size = {0:.2f}MB'.format(param))
scio.savemat('../../results/PSTmain_cfg_{}_acc{:.4f}.mat'.format(block_cfg, best_acc_mix),
			 {'train_acc':train_acc, 'test_acc_0':test_acc_0,'test_acc_current':test_acc_current,
			  'test_acc_mix':test_acc_mix, 'best_acc_mix':best_acc_mix, 'best_acc_0': best_acc_0,
			  'block_cfg':block_cfg, 'size_cfg':size_cfg,'param':param,
			'epoch': args.num_epoch, 'epoch_edge': args.epoch_edge,
			'lr': args.lr, 'lr_step_size':args.lr_step_size,
			'test_acc_0_end':test_acc_0_end,
			'test_task_accu':test_task_accu, 'score': args.score,
			'dataset':args.dataset, 'task_list': task_list, 'seed':args.seed, 'shuffle':args.shuffle,
			'threshold_task_ratio':ratio})
# plt.show()

