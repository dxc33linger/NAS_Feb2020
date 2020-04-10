"""
Block library
03/27/2019
dxc

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
# """
# OPS = {
#     'none': Zero(stride),
#     'avg_pool_3x3': PoolBN('avg', C, 3, stride, 1),
#     'max_pool_3x3': PoolBN('max', C, 3, stride, 1),
#     'skip_connect': Identity() if stride == 1 else FactorizedReduce(C, C),
#     'sep_conv_3x3': SepConv(C, C, 3, stride, 1),
#     'sep_conv_5x5': SepConv(C, C, 5, stride, 2),
#     'sep_conv_7x7': SepConv(C, C, 7, stride, 3),
#     'dil_conv_3x3': DilConv(C, C, 3, stride, 2, 2), # 5x5
#     'dil_conv_5x5': DilConv(C, C, 5, stride, 4, 2), # 9x9
#     'conv_7x1_1x7': FacConv(C, C, 7, stride, 3)
# }
# """
def BlockFactory(number, downSample, **kwargs):
	in_planes = kwargs['in_planes']
	out_planes = kwargs['out_planes']
	# expansion = kwargs['expansion']

	if downSample:
		stride = 2
	else:
		stride = 1

	block_dict = {
			'0': SepConv(in_planes, out_planes, 3, stride, 1),
			'1': SepConv(in_planes, out_planes, 5, stride, 2),
			'2': DilConv(in_planes, out_planes, 3, stride, 2, 2),  # 5x5
			'3': DilConv(in_planes, out_planes, 5, stride, 4, 2),  # 9x9
			'4': StdConv(in_planes, out_planes, 3, stride, 1),
			'5': Block_resnet(in_planes, out_planes, stride),

			'6': Identity() if stride == 1 else FactorizedReduce(in_planes, out_planes),
			'7': PoolBN('avg', out_planes, 3, stride, 1),
			'8': PoolBN('max', out_planes, 3, stride, 1),

			'head': StdConv(in_planes, out_planes, 3, stride, 1),
			'fc':Block_fc(in_planes, out_planes)
	}

	if number == 'fc':
		return block_dict[number]
	else:
		return block_dict[str(number)]


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


class StdConv(nn.Module):
	""" Standard conv
	ReLU - Conv - BN
	"""
	def __init__(self, C_in, C_out, kernel_size, stride, padding):
		super().__init__()
		self.net = nn.Sequential(
			nn.ReLU(),
			nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
			nn.BatchNorm2d(C_out)
		)

	def forward(self, x):
		out = self.net(x)
		# logging.info('input.shape {}, output.shape {}'.format(x.shape, out.shape))
		return out

#
# class FacConv(nn.Module):
#     """ Factorized conv
#     ReLU - Conv(Kx1) - Conv(1xK) - BN
#     """
#     def __init__(self, C_in, C_out, kernel_length, stride, padding):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
#             nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
#             nn.BatchNorm2d(C_out)
#         )
#
#     def forward(self, x):
#         return self.net(x)
#

class DilConv(nn.Module):
	""" (Dilated) depthwise separable conv
	ReLU - (Dilated) depthwise separable - Pointwise - BN
	If dilation == 2, 3x3 conv => 5x5 receptive field
					  5x5 conv => 9x9 receptive field
	"""
	def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
		super().__init__()
		self.net = nn.Sequential(
			nn.ReLU(),
			nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
					  bias=False),
			nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(C_out)
		)

	def forward(self, x):
		out = self.net(x)
		## # logging.info('input.shape {}, output.shape {}'.format(x.shape, out.shape))
		return out


class SepConv(nn.Module):
	""" Depthwise separable conv
	DilConv(dilation=1) * 2
	"""
	def __init__(self, C_in, C_out, kernel_size, stride, padding):
		super().__init__()
		self.net = nn.Sequential(
			DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1),
			DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1)
		)

	def forward(self, x):
		out = self.net(x)
		# logging.info('input.shape {}, output.shape {}'.format(x.shape, out.shape))
		return out


class Block_resnet(nn.Module):
	expansion = 1
	def __init__(self, in_planes, planes, stride=1):
		super(Block_resnet, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1   = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2   = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		# logging.info('input.shape {}, output.shape {}'.format(x.shape, out.shape))
		return out

class FactorizedReduce(nn.Module):
	"""
	Reduce feature map size by factorized pointwise(stride=2).
	"""
	def __init__(self, C_in, C_out):
		super().__init__()
		self.relu = nn.ReLU()
		self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
		self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
		self.bn = nn.BatchNorm2d(C_out)

	def forward(self, x):
		x = self.relu(x)
		out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
		out = self.bn(out)
		# logging.info('input.shape {}, output.shape {}'.format(x.shape, out.shape))
		return out


class Block_fc(nn.Module): # FC
	def __init__(self, in_planes, num_classes):
		super(Block_fc, self).__init__()
		self.blocks = nn.Sequential(
			nn.Linear(in_planes, int(1.2*num_classes))
		)
	def forward(self, x):
		out = self.blocks(x)
		# logging.info('input.shape {}, output.shape {}'.format(x.shape, out.shape))
		return out

