import os
import sys
import utils
import math
import logging
import argparse
import numpy as np
import train_search
from dataload import *
import scipy.io as scio
from block_library import *
from args import parser
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu


log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
	format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('../log_files', 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("args = %s", args)

# np.random.seed(args.seed)

num_block = {'plain': 6,
			'all': 8} 


size_cfg = [3]
block_cfg = []
ds_cfg = []

for i in range(args.num_module):	
	blocks = np.random.randint(1, args.max_block+1) # number of blocks in a module
	block_cfg += np.random.randint(0, num_block['plain'], size = blocks).tolist()  # random index from block library
	ds_cfg += [False]*blocks

	block_cfg += [np.random.randint(0, num_block['all'])] # ds_cfg
	ds_cfg += [True]	
	size_cfg += (2 ** np.random.randint(4, 10, size = blocks+1)).tolist()

for j,x in enumerate(block_cfg): 
	if x == 6 or x==7:
		size_cfg[j+1] = size_cfg[j]

logging.info('block_cfg = {} length: {}'.format(block_cfg, len(block_cfg)))
logging.info(' size_cfg = {} length: {}'.format( size_cfg, len( size_cfg)))
logging.info('   ds_cfg = {} length: {}'.format(   ds_cfg, len(   ds_cfg)))



trainloader, testloader, valloader = dataload()
nas = train_search.NAS()

model = nas.initial_network(block_cfg, size_cfg, ds_cfg)
nas.initialization()
torch.save(model,'../results/saved_model_cfg{}'.format(block_cfg))

train_acc = np.zeros([1, args.num_epoch])
test_acc = np.zeros([1, args.num_epoch])

for epoch in range(args.num_epoch):
	param = utils.count_parameters_in_MB(model)
	logging.info('param size = {0:.2f}MB'.format(param))

	train_acc[0, epoch] = nas.train(epoch, trainloader, valloader)
	test_acc[0, epoch] = nas.test(testloader)
		
	logging.info('epoch {0} lr {1} train_acc {2:.4f} test_acc {3:.4f}'.format( 
		epoch, args.lr, train_acc[0, epoch], test_acc[0, epoch]))

scio.savemat('../results/test_acc{0:.4f}_param{1:.1f}MB.mat'
	.format(test_acc[0,-1], param), 
	{'block_cfg':block_cfg, 'ds_cfg':ds_cfg, 'size_cfg': size_cfg, 
	'train_acc':train_acc, 'test_acc':test_acc})