'''
This is the codes for NAS project, starting from 03/2019
Author: XD

File description: main file
'''

import os
import sys
import utils
import logging
import train_search
import numpy as np
from dataload import *
import scipy.io as scio
import torch
from dataload_continual import *

from args import parser
args = parser.parse_args()
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
	format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('../../results', 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("\n\n\nargs = %s", args)
logging.info(" - - - - - NAS main.py - - - - - - - ")


num_block = {'plain': 6,
			'all': 8}

random.seed(args.seed)

# size_cfg = [3]
# block_cfg = []
# ds_cfg = []
#
# for i in range(args.num_module):
# 	blocks = np.random.randint(1, args.max_block+1) # number of blocks in a module
# 	block_cfg += np.random.randint(0, num_block['plain'], size = blocks).tolist()  # random index from block library
# 	ds_cfg += [False]*blocks  # False means
#
# 	block_cfg += [np.random.randint(0, num_block['all'])] # ds_cfg
# 	ds_cfg += [True]
# 	size_cfg += (2 ** np.random.randint(4, 10, size = blocks+1)).tolist()
#
# for j,x in enumerate(block_cfg):
# 	if x == 6 or x == 7: ## if block is pooling layer, size is same with last layer
# 		size_cfg[j+1] = size_cfg[j]

if args.cfg == 1:
	block_cfg = [1, 4, 0, 5, 0, 0, 1, 3]
	size_cfg = [3, 128, 128, 256, 256, 64, 512, 32, 64]
	ds_cfg = [False, True, False, False, True, False, False, True]
elif args.cfg == 2:
	block_cfg = [2, 5, 1, 6, 1, 1, 2, 4]
	size_cfg = [3, 128, 128, 256, 256, 64, 512, 32, 64]
	ds_cfg = [False, True, False, False, True, False, False, True]
elif args.cfg == 3:
	block_cfg = [4, 6, 1, 3, 2, 5, 1, 0]
	size_cfg = [3, 128, 128, 256, 256, 64, 512, 32, 64]
	ds_cfg = [False, True, False, False, True, False, False, True]


logging.info('block_cfg = {} length: {}'.format(block_cfg, len(block_cfg)))
logging.info(' size_cfg = {} length: {}'.format( size_cfg, len( size_cfg)))
logging.info('   ds_cfg = {} length: {}'.format(   ds_cfg, len(   ds_cfg)))


nas = train_search.NAS()

logging.info('Current dataset mode %s: ', args.mode)
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


model = nas.initial_network(block_cfg, size_cfg, ds_cfg)
nas.initialization()

train_acc = np.zeros([1, args.num_epoch])
test_acc = np.zeros([1, args.num_epoch])


for epoch in range(args.num_epoch):
	train_acc[0, epoch] = nas.train(epoch, trainloader)
	test_acc[0, epoch] = nas.test(testloader)
	logging.info('epoch {0} lr {1} train_acc {2:.4f} test_acc {3:.4f}'.format(
		epoch, nas.current_lr, train_acc[0, epoch], test_acc[0, epoch]))

	if epoch % 10 == 0:
		# add noise and test
		logging.info('\n\nAdd Noise within range -{:.5f}, {:.5f}'.format(args.alpha, args.alpha))
		model_wNoise = nas.add_noise(args.alpha)
		accuracy_wNoise = nas.test(testloader)
		logging.info('Accuracy changes {:.4f} -> {:.4f} after adding Gaussian noise with alpha={}'.format(test_acc[0, epoch], accuracy_wNoise, args.alpha))

		# torch.save(model_wNoise, '../../results/epoch{}_Noisealpha{}_cfg{}_diffAcc{:.4f}.mat'.format(args.alpha, args.max_block, epoch, args.alpha, block_cfg, test_acc[0, epoch]-accuracy_wNoise)
		scio.savemat('../../results/epoch{}_alpha{}_cfg{}_Acc{:.4f}to{:.4f}.mat'.format(epoch, args.alpha, block_cfg, test_acc[0, epoch],accuracy_wNoise),
					 {'alpha': args.alpha, 'test_acc':test_acc[0, epoch], 'accuracy_wNoise':accuracy_wNoise })
		# load back previous weight without noise
		logging.info('\n\nLoading back weights without noise.....')
		nas.load_weight_back()
		print('Is accuracy the same?', nas.test(testloader))



param = utils.count_parameters_in_MB(nas.net)

x = np.linspace(0, args.num_epoch, args.num_epoch)
plt.xlabel('Epoch')
plt.ylabel('Testing Accuracy')
plt.plot(x, train_acc[0, :] , 'g', alpha=0.5, label = 'Training accuracy')
plt.plot(x, test_acc[0, :],   'b', alpha=1.0, label = 'Testing accuracy')
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, args.num_epoch+1, step=10))
plt.grid(color='k', linestyle='-', linewidth=0.05)
plt.legend(loc='best')
plt.title('Learning curve\ncfg{}\nds_cfg{}'.format(block_cfg, ds_cfg))
plt.savefig('../../results/plot_LC_acc{:.3f}_param{:.2f}MB.png'.format(sum(test_acc[0,-5:])/5, param))
# plt.show()
torch.save(nas.net,'../../results/final_model_cfg{}_acc{:.3f}'.format(block_cfg, sum(test_acc[0,-5:])/5))

logging.info('param size = {0:.2f}MB'.format(param))
scio.savemat('../../results/test_acc{:.3f}_param{:.2f}MB.mat'.format(sum(test_acc[0,-5:])/5, param),
	{'block_cfg':block_cfg, 'ds_cfg':ds_cfg, 'size_cfg': size_cfg,
	 'lr_step_size': args.lr_step_size,  'lr_gamma': args.gamma, 'seed': args.seed,
	 'train_acc':train_acc, 'test_acc':test_acc, 'param':param, 'num_epoch': args.num_epoch,
	 'num_module': args.num_module, 'max_block': args.max_block})
logging.info('Results saved in ../../results/test_acc{:.3f}_param{:.1f}MB.mat'.format(sum(test_acc[0,-5:])/5, param))

