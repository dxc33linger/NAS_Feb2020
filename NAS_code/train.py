'''
This is the codes for NAS project, starting from 03/2019
Author: XD

File description: main file
'''

import os
import sys

import matplotlib.pyplot as plt
import scipy.io as scio
import torch

import functions
import utils
from args import parser
import make_architecture

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu
log_format = '%(asctime)s   %(message)s'
make_architecture.logging.basicConfig(stream=sys.stdout, level=make_architecture.logging.INFO,
									  format=log_format, datefmt='%m/%d %I:%M%p')
fh = make_architecture.logging.FileHandler(os.path.join('../../results', 'log.txt'))
fh.setFormatter(make_architecture.logging.Formatter(log_format))
make_architecture.logging.getLogger().addHandler(fh)
make_architecture.logging.info("\n\n\n - - - - - NAS main.py - - - - - - - ")
make_architecture.logging.info("args = %s", args)

random.seed(args.seed)
nas = functions.NAS()
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

#
# param = 5.1
# while param >= 5.0: # disgard network that larger than 5M
# 	block_cfg, size_cfg, ds_cfg = make_cfg(args.DNA_SIZE)
# 	model = nas.initial_network(block_cfg, size_cfg, ds_cfg)
# 	param = utils.count_parameters_in_MB(nas.net)
# 	logging.info('param size = {0:.2f}MB\n'.format(param))
#

# block_cfg = [   4,  5,  0,  7,  2,  4,  6, 5,   6,  3,   8,   4,  5,  0, 7,   2, 4,   6,  5,  6, 3,   8,   4,   5, 0,   7, 2,   4,  6,   5, 6, 3]
# size_cfg =[16, 32, 24, 32, 32, 32, 32, 24, 24, 24, 128, 128, 32, 24, 32, 32, 32, 32, 24, 24, 24, 128, 128, 32, 24, 32, 32, 32, 32, 24, 24, 24, 128]
# ds_cfg =  [False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False]

ds_cfg =  [False, False, False,False, False, True, False, False,False, False, False, True, False, False,False, False, False]
block_cfg =     [0, 5, 1, 3, 3, 7, 0, 1, 5, 5, 4, 9, 0, 4, 7, 5, 6]
size_cfg = [16, 32, 64, 16, 256, 24, 24, 512, 512, 64, 32, 24, 24, 64, 16, 16, 16, 24]
model = nas.initial_network(block_cfg, size_cfg, ds_cfg)
nas.initialization()
param = utils.count_parameters_in_MB(nas.net)



train_acc = np.zeros([1, args.num_epoch])
test_acc = np.zeros([1, args.num_epoch])
accu_wNoise = np.zeros([1, args.num_epoch])

for epoch in range(args.num_epoch):
	train_acc[0, epoch] = nas.train(epoch, trainloader)
	test_acc[0, epoch] = nas.test(testloader)
	make_architecture.logging.info('epoch {0} lr {1} ========== train_acc {2:.4f} test_acc {3:.4f}'.format(
		epoch, nas.current_lr, train_acc[0, epoch], test_acc[0, epoch]))
	# nas.check_weight()
	if epoch % 10 == 0  or epoch == args.num_epoch - 1:
		# add noise and test
		make_architecture.logging.info('Add Noise within range -{:.5f}, {:.5f}'.format(args.alpha, args.alpha))
		nas.add_noise(args.alpha)
		accuracy_wNoise = nas.test(testloader)

		make_architecture.logging.info('Accuracy changes {:.4f} -> {:.4f} after adding Gaussian noise with alpha={}'.format(test_acc[0, epoch], accuracy_wNoise, args.alpha))
		accu_wNoise[0, epoch] = accuracy_wNoise
		scio.savemat('../../results/mode_{}/epoch{}_alpha{}_cfg{}_Acc{:.4f}to{:.4f}.mat'.format(args.mode, epoch, args.alpha, block_cfg, test_acc[0, epoch], accuracy_wNoise),
					 {'alpha': args.alpha, 'test_acc':test_acc[0, epoch], 'accuracy_wNoise': accuracy_wNoise})
		# load back previous weight without noise
		make_architecture.logging.info('Loading back weights without noise.....')
		nas.load_weight_back()
		print('Is accuracy the same?', nas.test(testloader))


torch.save(nas.net,'../../results/mode_{}/final_model_cfg{}_acc{:.3f}'.format(args.mode, block_cfg, sum(test_acc[0,-5:])/5))
make_architecture.logging.info('param size = {0:.2f}MB'.format(param))
scio.savemat('../../results/mode_{}/test_acc{:.3f}_param{:.2f}MB.mat'.format(args.mode, sum(test_acc[0,-5:])/5, param),
	{'block_cfg':block_cfg, 'ds_cfg':ds_cfg, 'size_cfg': size_cfg,
	 'lr_step_size': args.lr_step_size,  'lr_gamma': args.gamma, 'seed': args.seed,
	 'train_acc':train_acc, 'test_acc':test_acc, 'accu_wNoise':accu_wNoise,
	 'param':param, 'num_epoch': args.num_epoch, 'DNA_SIZE': args.DNA_SIZE})
make_architecture.logging.info('Results saved in ../../results/mode_{}/test_acc{:.3f}_param{:.1f}MB.mat'.format(args.mode, sum(test_acc[0, -5:]) / 5, param))

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
plt.show()