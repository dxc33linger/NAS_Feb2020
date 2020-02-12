from dataload import *
from utils import progress_bar
from make_architecture import *



class NAS(object):
	def __init__(self):

		self.batch_size = args.batch_size

	

	def initial_network(self, block_cfg, size_cfg, ds_cfg):
		self.net = architecture(block_cfg, size_cfg, ds_cfg)
		print(self.net)
		return self.net

	def initialization(self):
		
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		self.net = self.net.to(self.device)
		if self.device == 'cuda':
			self.net = nn.DataParallel(self.net)
			cudnn.benchmark = True
		
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.SGD(self.net.parameters(), lr = args.lr, momentum=0.9, weight_decay=5e-4) 
		self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = args.lr_step_size, gamma= args.lr_gamma) 
		# self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        # self.optimizer, float(args.num_epoch), eta_min=args.learning_rate_min)

	def train(self, epoch, trainloader, valloader):

		lr_list = self.scheduler.get_lr()
		print('\nEpoch: %d lr: %s' % (epoch, self.scheduler.get_lr()))
		self.net.train()
		train_loss = 0.0
		correct = 0
		total = 0
		for batch_idx, (inputs, targets) in enumerate(trainloader):
			inputs, targets = inputs.to(self.device), targets.to(self.device)

			inputs_var = torch.autograd.Variable(inputs)
			targets_var = torch.autograd.Variable(targets)

			outputs = self.net(inputs_var)

			loss = self.criterion(outputs, targets_var)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			train_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
		
			progress_bar(batch_idx, len(trainloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Train' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
		return correct/total

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