import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, sampler
from math import pow
from args import parser

args = parser.parse_args()



def num_classes_in_feature():
	if args.dataset in ['cifar10','SVHN', 'mnist']:
		num_classes = 10
		in_feature = int(pow(32/(2**args.num_module),2)	)
		
	elif args.dataset in ['cifar100']:
		num_classes = 100
		in_feature = int(pow(32/(2**args.num_module),2)	)
	
	elif args.dataset in ['imagenet-5k']:
		num_classes = 5000
		in_feature = int(pow(224/(2**args.num_module),2))

	else:
		raise ValueError('Dataset not found!')

	return num_classes, in_feature



def dataload():
	kwargs = {'root': '../../dataset_dxc', 'download': True }
	print('\n===> Preparing {} dataset...'.format(args.dataset))

	if args.dataset == 'mnist':
		transform_train = transforms.Compose([
			transforms.ToTensor(),	])

		transform_test = transforms.Compose([
			transforms.ToTensor(),	])

		trainset =  datasets.MNIST(train=True, transform=transform_train, **kwargs)
		testset =  datasets.MNIST(train=False, transform=transform_test, **kwargs)

		full_training = 50000
		n_training_samples = int(50000 / args.K * (args.K-1))
		n_val_samples = int(50000 / args.K)


	elif args.dataset == 'cifar10':			
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		
		trainset = datasets.CIFAR10(train=True, transform=transform_train, **kwargs)
		testset = datasets.CIFAR10(train=False, transform=transform_test, **kwargs)
	
		full_training = 50000		
		n_training_samples = int(50000 / args.K * (args.K-1))
		n_val_samples = int(50000 / args.K)
	

	elif args.dataset == 'cifar100':
		transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		trainset = datasets.CIFAR100(train=True, transform=transform_train, **kwargs)
		testset = datasets.CIFAR100(train=False, transform=transform_test, **kwargs)

		full_training = 50000		
		n_training_samples = int(50000 / args.K * (args.K-1))
		n_val_samples = int(50000 / args.K)


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
		# # trainset = datasets.SVHN(root='./dataset_SVHN_dxc', split='extra', download=True, transform=transform_train)


		# 73257 26032 531131
		full_training = 73257
		n_training_samples = int( 73257 / args.K * (args.K-1))
		n_val_samples = int(73257 / args.K)


	else:
		raise ValueError("Wrong dataset given!")

	if args.CV is True:
		train_sampler = sampler.SubsetRandomSampler(range(n_training_samples))
		val_sampler =  sampler.SubsetRandomSampler(range(n_training_samples, n_training_samples+n_val_samples))
		valloader =	DataLoader(trainset, batch_size=args.batch_size, shuffle=False,  num_workers=2, sampler=val_sampler, drop_last=True)	
	else: 
		train_sampler =  sampler.SubsetRandomSampler(range(full_training))


	trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=2, sampler=train_sampler, drop_last=True)		
	testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
	valloader =	None	


	print('===> Dataset {} is ready.\n'.format(args.dataset))
	return trainloader, testloader, valloader
