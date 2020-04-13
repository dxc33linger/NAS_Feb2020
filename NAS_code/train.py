'''
This is the codes for NAS project, starting from 03/2019
Author: XD

File description: main file
'''

import os
import sys
import utils
import logging
import functions
import numpy as np
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
logging.info("\n\n\n - - - - - NAS main.py - - - - - - - ")
logging.info("args = %s", args)

random.seed(args.seed)
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

#
# param = 5.1
# while param >= 5.0: # disgard network that larger than 5M
# 	block_cfg, size_cfg, ds_cfg = make_cfg(args.DNA_SIZE)
# 	model = nas.initial_network(block_cfg, size_cfg, ds_cfg)
# 	param = utils.count_parameters_in_MB(nas.net)
# 	logging.info('param size = {0:.2f}MB\n'.format(param))
#

block_cfg = [6, 5, 4, 7, 4, 1, 3, 5, 4, 7, 4, 1, 3, 5, 4, 7, 4]
size_cfg = [16, 24, 32, 512, 512, 32, 128, 128, 32, 512, 512, 32, 128, 128, 32, 512, 512, 32]
ds_cfg = [False, False, False, False, False, True, False, False, False, False, False, True, False, False, False, False, False]

model = nas.initial_network(block_cfg, size_cfg, ds_cfg)
nas.initialization()








train_acc = np.zeros([1, args.num_epoch])
test_acc = np.zeros([1, args.num_epoch])
accu_wNoise = np.zeros([1, args.num_epoch])

for epoch in range(args.num_epoch):
	train_acc[0, epoch] = nas.train(epoch, trainloader)
	test_acc[0, epoch] = nas.test(testloader)
	logging.info('epoch {0} lr {1} ========== train_acc {2:.4f} test_acc {3:.4f}'.format(
		epoch, nas.current_lr, train_acc[0, epoch], test_acc[0, epoch]))
	# nas.check_weight()
	if epoch == 4  or epoch == args.num_epoch - 1:
		# add noise and test
		logging.info('Add Noise within range -{:.5f}, {:.5f}'.format(args.alpha, args.alpha))
		nas.add_noise(args.alpha)
		accuracy_wNoise = nas.test(testloader)

		logging.info('Accuracy changes {:.4f} -> {:.4f} after adding Gaussian noise with alpha={}'.format(test_acc[0, epoch], accuracy_wNoise, args.alpha))
		accu_wNoise[0, epoch] = accuracy_wNoise
		scio.savemat('../../results/mode_{}/epoch{}_alpha{}_cfg{}_Acc{:.4f}to{:.4f}.mat'.format(args.mode, epoch, args.alpha, block_cfg, test_acc[0, epoch], accuracy_wNoise),
					 {'alpha': args.alpha, 'test_acc':test_acc[0, epoch], 'accuracy_wNoise': accuracy_wNoise})
		# load back previous weight without noise
		logging.info('Loading back weights without noise.....')
		nas.load_weight_back()
		print('Is accuracy the same?', nas.test(testloader))


torch.save(nas.net,'../../results/mode_{}/final_model_cfg{}_acc{:.3f}'.format(args.mode, block_cfg, sum(test_acc[0,-5:])/5))
logging.info('param size = {0:.2f}MB'.format(param))
scio.savemat('../../results/mode_{}/test_acc{:.3f}_param{:.2f}MB.mat'.format(args.mode, sum(test_acc[0,-5:])/5, param),
	{'block_cfg':block_cfg, 'ds_cfg':ds_cfg, 'size_cfg': size_cfg,
	 'lr_step_size': args.lr_step_size,  'lr_gamma': args.gamma, 'seed': args.seed,
	 'train_acc':train_acc, 'test_acc':test_acc, 'accu_wNoise':accu_wNoise,
	 'param':param, 'num_epoch': args.num_epoch, 'DNA_SIZE': args.DNA_SIZE})
logging.info('Results saved in ../../results/mode_{}/test_acc{:.3f}_param{:.1f}MB.mat'.format(args.mode, sum(test_acc[0,-5:])/5, param))

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
plt.savefig('../../results/mode_{}/plot_LC_acc{:.3f}_param{:.2f}MB.png'.format(args.mode, sum(test_acc[0,-5:])/5, param))
# plt.show()