from block_library import *
from dataload_regular import *
from dataload_continual import *
import logging

args = parser.parse_args()


def make_cfg(DNA_SIZE, num_reduction = 3):
	'''
	Args:
		DNA_SIZE: NA length
		num_reduction: Number of meta modules, conv - module1 - reduction - module2 - reduction - module3 -fc- output

	Returns: block_cfg, size_cfg, ds_cfg
	'''
	num_block, pool_blocks, densenet_num = number_of_blocks()

	block_cfg_module = []
	size_cfg_module = []  # initial channel = 16
	ds_cfg_module = []
	block_cfg_reduction = []
	size_cfg_reduction = []
	ds_cfg_reduction = []

	# module1
	block_cfg_module += np.random.randint(0, num_block['plain'], size=DNA_SIZE).tolist()  # random index from block library
	ds_cfg_module += [False] * DNA_SIZE  # False means
	size_cfg_module += (2 ** np.random.randint(4, 10, size=DNA_SIZE)).tolist()
	## reduction cell
	block_cfg_reduction += [np.random.randint(0, num_block['all'])]  # ds_cfg
	ds_cfg_reduction += [True]
	size_cfg_reduction += (2 ** np.random.randint(4, 10, size=1)).tolist()

	block_cfg = (block_cfg_module + block_cfg_reduction) * 2 + block_cfg_module
	size_cfg = [16] + (size_cfg_module + size_cfg_reduction) * 2 + size_cfg_module
	ds_cfg = (ds_cfg_module + ds_cfg_reduction) * 2 + ds_cfg_module

	for idx, block in enumerate(block_cfg):
		if block in pool_blocks:  ## if block is pooling layer, size is same with last layer
			size_cfg[idx + 1] = size_cfg[idx]
		elif block == densenet_num and block_cfg[idx - 1] != densenet_num:  # non-DB - DB
			size_cfg[idx + 1] = 24
		elif block == densenet_num and block_cfg[idx - 1] == densenet_num:  # DB - DB
			size_cfg[idx + 1] = 24

	logging.info('block_cfg = {} length: {}'.format(block_cfg, len(block_cfg)))
	logging.info(' size_cfg = {} length: {}'.format(size_cfg, len(size_cfg)))
	logging.info('   ds_cfg = {} length: {}'.format(ds_cfg, len(ds_cfg)))
	return block_cfg, size_cfg, ds_cfg


# size_cfg = [16]  # initial channel = 16
# block_cfg = []
# ds_cfg = []
# num_reduction = 3 # Number of meta modules, conv - module1 - reduction - module2 - reduction - module3 -fc- output
#
# for i in range(num_reduction):
# 	blocks = np.random.randint(1, args.max_block+1) # number of blocks in a module
# 	block_cfg += np.random.randint(0, num_block['plain'], size = blocks).tolist()  # random index from block library
# 	ds_cfg += [False]*blocks  # False means
# 	size_cfg += (2 ** np.random.randint(4, 8, size=blocks)).tolist()
#
# 	if i != num_reduction - 1:
# 		block_cfg += [np.random.randint(0, num_block['all'])] # ds_cfg
# 		ds_cfg += [True]
# 		size_cfg += (2 ** np.random.randint(4, 8, size=1)).tolist()
#
# for idx,x in enumerate(block_cfg):
# 	if x in [6, 7, 8]: ## if block is pooling layer, size is same with last layer
# 		size_cfg[idx+1] = size_cfg[idx]
#
# #

class architecture(nn.Module):
	def __init__(self, block_cfg, size_cfg, ds_cfg):
		super(architecture, self).__init__()
		self.block_cfg = block_cfg
		self.size_cfg = size_cfg
		self.ds_cfg = ds_cfg
		self.num_classes, self.in_feature = num_classes_in_feature()
		_, _, densenet_num = number_of_blocks()

		kwargs = {'in_planes': 3, 'out_planes': 16}
		if args.dataset in ['cifar10', 'SVHN', 'mnist', 'cifar100']:
			self.head = BlockFactory('head', downSample=False, **kwargs)
		elif args.dataset in ['imagenet-5k']:
			head_layer = []
			# head_layer.append(
			# 		BlockFactory('head', downSample=False, **{'in_planes': 3, 'out_planes': 16}),
			# 		BlockFactory('head', downSample=False, **{'in_planes': 16, 'out_planes': 16}),
			# 		BlockFactory('head', downSample=False, **{'in_planes': 16, 'out_planes': 16}),
			# )
			self.head = nn.Sequential(*head_layer)

		self.features = nn.Sequential(self._make_layer())
		kwargs = {'in_planes': self.in_feature*size_cfg[-1],
				  'out_planes': self.num_classes }

		self.classifier = BlockFactory('fc', downSample = False, **kwargs)  # FC

	def _make_layer(self):
		layers = []
		for lyr_idx, num in enumerate(self.block_cfg):
			kwargs = {'in_planes': self.size_cfg[lyr_idx],
					 'out_planes': self.size_cfg[lyr_idx+1]	}
			layers.append(BlockFactory(num, self.ds_cfg[lyr_idx], **kwargs))
		return nn.Sequential(*layers)


	def forward(self, x):
		x = self.head(x)
		x = self.features(x)
		x = x.view(x.size(0), -1)
		out = self.classifier(x)
		return out