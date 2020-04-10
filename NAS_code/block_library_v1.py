"""
Block library
03/27/2019
dxc

"""
import abc
import torch
import torch.nn as nn
import torch.nn.functional as F

def BlockFactory(number, downSample, **kwargs):
	in_planes = kwargs['in_planes']
	out_planes = kwargs['out_planes']
	# expansion = kwargs['expansion']
	
	
	if downSample: 
		stride = 2
	else:
		stride = 1


	if number == 0:
		return Block_mobilenet(in_planes, out_planes, stride)

	elif number == 1:
		# return Block_mobilenetV2(in_planes, out_planes, expansion, stride)
		return Block_mobilenetV2(in_planes, out_planes, stride)

	elif number == 2:
		return Block_resnet(in_planes, out_planes, stride)
	
	elif number == 3:
		return Block_resnetV2(in_planes, out_planes, stride)

	elif number == 4:
		return Block_resnext(in_planes, out_planes, stride)
	
	elif number == 5:
		return Block_conv(in_planes, out_planes, stride)

	elif number == 6:
		return Block_pool(stride)

	elif number == 7:
		return Block_pool(stride)
	
	elif number == 'fc':
		return Block_fc(in_planes, out_planes) # out_planes=num_classes

	else:
		raise ValueError('Block indexc not found!')






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


