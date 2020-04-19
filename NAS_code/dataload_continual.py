"""

"""

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from args import parser
from torch.utils.data import DataLoader, sampler
import logging
import random
# torch.backends.cudnn.deterministic = True

import random
args = parser.parse_args()
# np.random.seed(args.seed)  # Python random module.
# torch.manual_seed(args.seed)  # for reproducibility for the same run.
# Transforms object for trainset with augmentation

kwargs = {'root': '../../dataset_dxc', 'download': True}

if args.dataset == 'cifar10':
	transform_with_aug = transforms.Compose([
		transforms.ToPILImage(),
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	# Transforms object for testset with NO augmentation
	transform_no_aug = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	trainset = datasets.CIFAR10(train=True , transform = transform_with_aug, **kwargs)
	testset = datasets.CIFAR10(train=False,  transform = transform_no_aug, **kwargs)

elif args.dataset == 'cifar100':
	transform_with_aug = transforms.Compose([
		transforms.ToPILImage(),
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])
	# Transforms object for testset with NO augmentation
	transform_no_aug = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	trainset = datasets.CIFAR100(train=True , transform = transform_no_aug, **kwargs)
	testset = datasets.CIFAR100(train=False,  transform = transform_no_aug, **kwargs)

elif args.dataset == 'SVHN':
	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		# transforms.Scale(config.image_size),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])
	trainset = datasets.SVHN(root='./dataset_dxc', split='train', download=True, transform=transform_train)
	testset = datasets.SVHN(root='./dataset_dxc', split='test', download=True, transform=transform_test)

else:
	raise ValueError("Wrong dataset given!")

#
# if torch.__version__ == '0.4.0':
# # Separating trainset/testset data/label
# 	x_train  = trainset.train_data
# 	x_test   = testset.test_data
# 	y_train  = trainset.train_labels
# 	y_test   = testset.test_labels
# elif torch.__version__ == '1.0.1.post2' or torch.__version__ == '1.1.0':
x_train  = trainset.data
x_test   = testset.data
y_train  = trainset.targets
y_test   = testset.targets


class DatasetMaker(Dataset):
	def __init__(self, datasets, label_list, start_idx, transformFunc = transform_with_aug):
		"""
		datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
		"""
		self.datasets = datasets
		self.lengths  = [len(d) for d in self.datasets]
		self.transformFunc = transformFunc
		self.label_list = label_list
		self.start_idx = start_idx

	def __getitem__(self, i):
		class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)

		img = self.datasets[class_label][index_wrt_class]
		img = self.transformFunc(img)
		# print('Ground truth {}:{}'.format(self.label_list[class_label], class_label + self.start_idx))
		if args.shuffle:
			return img, class_label + self.start_idx # re-assign label
		else:
			return img, self.label_list[class_label]# + self.start_idx # Ground truth label
		# return img, class_label + self.start_idx # re-assign label
	#
	def __len__(self):
		return sum(self.lengths)
	
	def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
		"""
		Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
		"""
		# Which class/bin does i fall into?
		accum = np.add.accumulate(bin_sizes)
		if verbose:
			print("accum =", accum)
		bin_index  = len(np.argwhere(accum <= absolute_index))
		if verbose:
			print("class_label =", bin_index)
		# Which element of the fallent class/bin does i correspond to?
		index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
		if verbose:
			print("index_wrt_class =", index_wrt_class)

		return bin_index, index_wrt_class

def get_class_i(x, y, i):
	"""
	x: trainset.train_data or testset.test_data
	y: trainset.train_labels or testset.test_labels
	i: class label, a number between 0 to 9
	return: x_i
	"""
	# Convert to a numpy array
	y = np.array(y)
	# Locate position of labels that equal to i
	pos_i = np.argwhere(y == i)
	# Convert the result into a 1-D list
	pos_i = list(pos_i[:,0])
	# Collect all data that match the desired label
	x_i = [x[j] for j in pos_i]
	
	return x_i


def dataload_partial(label_list, start_idx, batch_size_=args.batch_size, shuffle_ = True):

	train_list = []
	test_list = []
	for i in label_list:
		# print(label_list)
		train_list.append(get_class_i(x_train, y_train, i))
		test_list.append(get_class_i(x_test , y_test , i))

	partial_trainset = DatasetMaker(train_list, label_list, start_idx, transform_with_aug)
	partial_testset  = DatasetMaker(test_list,	label_list, start_idx, transform_no_aug)

	# Create datasetLoaders from trainset and testset
	partial_trainsetLoader  = DataLoader(partial_trainset, batch_size = batch_size_, shuffle = shuffle_, drop_last=False, num_workers=1)#, worker_init_fn=_init_fn)
	partial_testsetLoader   = DataLoader(partial_testset , batch_size = batch_size_, shuffle = False, drop_last=False, num_workers=1)#, worker_init_fn=_init_fn)

	return partial_trainsetLoader, partial_testsetLoader


### different memory size for each class
def get_partial_dataset_cifar(start_idx, label_list, num_images):
	train_list = []
	test_list = []
	all_lable_list = []
	# logging.info('label_list {}'.format(label_list))
	for t in range(len(label_list)):
		all_lable_list += label_list[t]
		# logging.info('t {} label_list[t]'.format(t, label_list[t]))
		for i in label_list[t]:
			num_images_per_classes = int(num_images[t] / len(label_list[t]))
			# logging.info(' class {}, num_image/class={} '.format(i, num_images_per_classes))
			train_list.append(get_class_i(x_train, y_train, i)[0 : num_images_per_classes])  #num_image_class
			test_list.append(get_class_i(x_test , y_test , i))

	partial_trainset = DatasetMaker(train_list, all_lable_list, start_idx, transform_with_aug)
	partial_testset  = DatasetMaker(test_list,	all_lable_list, start_idx, transform_no_aug)

	# Create datasetLoaders from trainset and testset
	partial_trainsetLoader  = DataLoader(partial_trainset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=1)#, worker_init_fn=_init_fn)
	partial_testsetLoader   = DataLoader(partial_testset , batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=1)#, worker_init_fn=_init_fn)

	return partial_trainsetLoader, partial_testsetLoader

