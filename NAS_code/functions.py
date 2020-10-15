import random
from collections import OrderedDict

from torch.autograd import Variable

from dataload_regular import *
from make_architecture import *
from utils import progress_bar


class NAS(object):
	def __init__(self):
		self.batch_size = args.batch_size

	def initial_network(self, block_cfg, size_cfg, ds_cfg):
		self.net = architecture(block_cfg, size_cfg, ds_cfg)
		self.block_cfg = block_cfg
		# logging.info('network %s', self.net)
		return self.net

	def initialization(self, lr = args.lr, lr_step_size = args.lr_step_size):
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.net = self.net.to(self.device)
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
		self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_step_size, gamma=args.gamma)



	def train(self, epoch, trainloader):
		self.net.train()
		train_loss = 0.0
		correct = 0
		total = 0
		self.scheduler.step()
		self.current_lr = self.scheduler.get_lr()
		logging.info('\nEpoch: %d lr: %s' % (epoch, self.scheduler.get_lr()))

		for batch_idx, (inputs, targets) in enumerate(trainloader):
			inputs, targets = inputs.to(self.device), targets.to(self.device)

			inputs_var = torch.autograd.Variable(inputs)
			targets_var = torch.autograd.Variable(targets)

			self.optimizer.zero_grad()
			outputs = self.net(inputs_var)
			loss = self.criterion(outputs, targets_var)

			loss.backward()
			# nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
			self.optimizer.step()

			train_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			progress_bar(batch_idx, len(trainloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Train' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

		if not args.default_model and epoch == 89 or epoch == 49:
			self.save_checkpoint_t7(epoch,  correct/total, train_loss)
		return correct/total

	def save_checkpoint_t7(self, epoch, acc, loss, postfix = ''):
		self.save_folder = self.name_save_folder(args)
		state = {
			'acc': acc,
			'loss': loss,
			'epoch': epoch,
			'state_dict': self.net.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
		}
		opt_state = {
			'optimizer': self.optimizer.state_dict()
		}
		self.model_file = '../../results/NAS_'+self.save_folder+'_model_epoch'+ str(epoch) + postfix + str(acc)+'.t7'
		logging.info('Saving checkpiont to ' + self.model_file)
		torch.save(state, self.model_file)

	def test(self, testloader):
		self.net.eval()				
		test_loss = 0.0
		correct = 0
		total = 0
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(testloader):
				inputs, targets = inputs.to(self.device), targets.to(self.device)				
				outputs = self.net(inputs)
				loss = self.criterion(outputs, targets)
				test_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0)
				correct += predicted.eq(targets).sum().item()

				progress_bar(batch_idx, len(testloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Test/Validation' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
		# self.scheduler.step()		
		return correct/total



	def create_task(self):
		num_classes, _ = num_classes_in_feature()
		# random select label
		a = list(range(0, num_classes))
		if args.shuffle:
			random.seed(args.seed)
			random.shuffle(a)
		else:
			a = a
		task_list = []
		for i in range(0, len(a), 1):
			task_list.append(a[i:i + 1])
		self.task_list = task_list
		self.total_num_task = int(num_classes / 1)
		return self.task_list, self.total_num_task



	def add_noise(self, alpha):
		self.alpha = alpha

		param_w_noise = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
		param_clean = OrderedDict([(k, None) for k in self.net.state_dict().keys()])

		for layer_name, param in self.net.state_dict().items():
			param = param.type(torch.cuda.FloatTensor)
			print(layer_name, param.shape)
			if len(param.shape) == 4 or len(param.shape) == 2:
				std = param.std().item()
				noise = self.alpha * param.clone().normal_(0, std)
				param_w_noise[layer_name] = Variable(param.clone() + noise.type(torch.cuda.FloatTensor), requires_grad=False)
				assert param_w_noise[layer_name].get_device() == self.net.state_dict()[layer_name].get_device(), "parameter and net are not in same device"
			else:
				param_w_noise[layer_name] = Variable(param.clone(), requires_grad=False)
			param_clean[layer_name] = Variable(param.clone(), requires_grad=True)
		self.net.load_state_dict(param_w_noise)
		self.param_clean = param_clean


	def load_weight_back(self):
		self.net.load_state_dict(self.param_clean)


	def name_save_folder(self, args):
		if args.default_model:
			save_folder = args.default_model
		else:
			save_folder = 'cfg_' + str(self.block_cfg)
		save_folder += '_lr=' + str(args.lr) + '_bs=' + str(args.batch_size)
		return save_folder



	def check_weight(self):
		for layer_name, param in self.net.state_dict().items():
			logging.info('layer_name {}, min {:.4f} max {:.4f}'.format(layer_name, torch.min(param), torch.max(param)) )
