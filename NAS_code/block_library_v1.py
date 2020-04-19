"""
Block library
03/27/2019
dxc

"""
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def BlockFactory(number, downSample, **kwargs):
	in_planes = kwargs['in_planes']
	out_planes = kwargs['out_planes']
	# expansion = kwargs['expansion']

	if downSample: 
		stride = 2
	else:
		stride = 1

	block_dict = {
			'0': Block_mobilenet(in_planes, out_planes, stride),
			'1': Block_mobilenetV2(in_planes, out_planes, stride),
			'2': Block_resnetV2(in_planes, out_planes, stride),  # 9x9
			'3': Block_conv(in_planes, out_planes, stride),
			'4': Block_resnet(in_planes, out_planes, stride),
			'5': Block_resnext(in_planes, out_planes, stride),
			'6': Block_DenseNet(in_planes, stride),
			'7': Identity() if stride == 1 else Block_reduction(in_planes, out_planes),

			'8': PoolBN('avg', out_planes, 3, stride, 1),
			'9': PoolBN('max', out_planes, 3, stride, 1),

			'head': StdConv(in_planes, out_planes, 3, stride, 1),
			'fc':Block_fc(in_planes, out_planes)
	}

	if number == 'fc':
		return block_dict[number]
	else:
		return block_dict[str(number)]



def number_of_blocks():
	num_block = {'plain': 8,
				 'all': 10}
	pool_blocks = [7, 8, 9]
	densenet_num = 6
	return num_block, pool_blocks, densenet_num

class StdConv(nn.Module):
	""" Standard conv
	ReLU - Conv - BN
	"""
	def __init__(self, C_in, C_out, kernel_size, stride, padding):
		super().__init__()
		self.net = nn.Sequential(
			nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
			nn.BatchNorm2d(C_out),
			nn.ReLU(),
		)

	def forward(self, x):
		out = self.net(x)
		# logging.info('input.shape {}, output.shape {}'.format(x.shape, out.shape))
		return out



class Block_mobilenet(nn.Module): #Block_MobileNet
	'''Depthwise conv + Pointwise conv'''
	def __init__(self, in_planes, out_planes, stride=1):
		super(Block_mobilenet, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn2 = nn.BatchNorm2d(out_planes)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		return out





class Block_mobilenetV2(nn.Module): #Block_MobileNet_v2
	'''expand + depthwise + pointwise'''
	def __init__(self, in_planes, out_planes, stride):
		super(Block_mobilenetV2, self).__init__()
		self.stride = stride
	
		expansion = 2 if in_planes!=3 else 1
		planes = expansion * in_planes
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn3 = nn.BatchNorm2d(out_planes)

		self.shortcut = nn.Sequential()
		if stride == 1 and in_planes != out_planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(out_planes),
			)
	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out = out + self.shortcut(x) if self.stride==1 else out
		return out




# When stride = 2, dimention decreases
class Block_resnet(nn.Module): #ResNetv1
	expansion = 1
	def __init__(self, in_planes, planes, stride):
		super(Block_resnet, self).__init__()
		self.blocks = nn.Sequential(
		 nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
		 nn.BatchNorm2d(planes),
		 nn.ReLU(inplace=True),
		 nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
		 nn.BatchNorm2d(planes),
		)

		self.shortcut = nn.Sequential()
		if  in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)
	def forward(self, x):
		out = self.blocks(x) 
		out += self.shortcut(x)
		out = F.relu(out)
		return out



# When stride = 2, dimention decreases
class Block_resnetV2(nn.Module): #ResNetv2
	expansion = 1
	def __init__(self, in_planes, planes, stride):
		super(Block_resnetV2, self).__init__()
		self.blocks = nn.Sequential(
		 nn.BatchNorm2d(in_planes),  
		 nn.ReLU(inplace=True),    
		 nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
		 nn.BatchNorm2d(planes),
		 nn.ReLU(inplace=True),  
		 nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
		)

		self.shortcut = nn.Sequential()
		if in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)
	def forward(self, x):
		out = self.blocks(x) 
		out += self.shortcut(x)
		return out



class Block_resnext(nn.Module): # ResNext
	'''Grouped convolution block.'''
	expansion = 2
	def __init__(self, in_planes, planes, stride=1):
		super(Block_resnext, self).__init__()
		bottleneck_width = 4
		cardinality = int(planes / self.expansion / bottleneck_width)
		group_width = cardinality * bottleneck_width
		self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(group_width)
		self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
		self.bn2 = nn.BatchNorm2d(group_width)
		self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*group_width:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*group_width)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out



class Block_conv(nn.Module): # plain conv network stride = 1
	def __init__(self, in_planes, planes, stride=1):
		super(Block_conv, self).__init__()
		self.blocks = nn.Sequential(             
			nn.Conv2d(in_planes, planes, stride=stride, kernel_size=3, padding=1,  bias=False), 
			nn.BatchNorm2d(planes,eps=1e-05, momentum=0.1, affine=True), 
			nn.ReLU(inplace=True)
			)
	def forward(self, x):
		out = self.blocks(x) 
		return out


class Identity(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x

class PoolBN(nn.Module):
	"""
	AvgPool or MaxPool - BN
	"""
	def __init__(self, pool_type, C_out, kernel_size, stride, padding):
		"""
		Args:
			pool_type: 'max' or 'avg'
		"""
		super().__init__()
		if pool_type.lower() == 'max':
			self.pool = nn.MaxPool2d(kernel_size, stride, padding)
		elif pool_type.lower() == 'avg':
			self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
		else:
			raise ValueError()

		self.bn = nn.BatchNorm2d(C_out)

	def forward(self, x):
		out = self.pool(x)
		out = self.bn(out)
		# logging.info('input.shape {}, output.shape {}'.format(x.shape, out.shape))
		return out


class Block_reduction(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    ref: https://github.com/khanrc/pt.darts/blob/48e71375c88772daac376829fb4bfebc4fb78144/models/ops.py#L165
    """
    def __init__(self, in_planes, planes, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_planes, planes // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_planes, planes // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(planes, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out



class Block_pool(nn.Module): # pooling
	def __init__(self, stride = 2):
		super(Block_pool, self).__init__()
		self.blocks = nn.Sequential(             
			nn.MaxPool2d(stride=stride, kernel_size=2)
			)
	def forward(self, x):
		out = self.blocks(x) 
		return out



class Block_fc(nn.Module): # FC
	def __init__(self, in_planes, num_classes):
		super(Block_fc, self).__init__()
		self.blocks = nn.Sequential(             
			nn.Linear(in_planes, int(1.2*num_classes))
		)
	def forward(self, x):
		out = self.blocks(x) 
		return out






class Bottleneck(nn.Module):
	def __init__(self, in_planes, growth_rate):
		super(Bottleneck, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(4*growth_rate)
		self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = self.conv2(F.relu(self.bn2(out)))
		out = torch.cat([out,x], 1)
		return out

#
class Transition(nn.Module):
	def __init__(self, in_planes, out_planes, stride):
		super(Transition, self).__init__()
		self.bn = nn.BatchNorm2d(in_planes)
		self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
		self.stride = stride
	def forward(self, x):
		out = self.conv(F.relu(self.bn(x)))
		if self.stride == 2:
			out = F.avg_pool2d(out, 2)
		return out
class Block_DenseNet(nn.Module):
	def __init__(self, in_planes, stride, block = Bottleneck, nblocks = 2, growth_rate=12, reduction=0.5):
		super(Block_DenseNet, self).__init__()
		self.growth_rate = growth_rate

		num_planes = 2*growth_rate
		self.conv1 = nn.Conv2d(in_planes, num_planes, kernel_size=3, padding=1, bias=False)

		self.dense1 = self._make_dense_layers(block, num_planes, nblocks)
		num_planes += nblocks*growth_rate
		out_planes = int(math.floor(num_planes*reduction))
		self.trans1 = Transition(num_planes, out_planes, stride)
		num_planes = out_planes
		self.bn = nn.BatchNorm2d(num_planes)

	def _make_dense_layers(self, block, in_planes, nblock):
		layers = []
		for i in range(nblock):
			layers.append(block(in_planes, self.growth_rate))
			in_planes += self.growth_rate
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.conv1(x)
		out = self.trans1(self.dense1(out))
		# logging.info('input.shape {}, output.shape {}'.format(x.shape, out.shape))

		return out
