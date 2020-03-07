global num_classes
global in_feature
import numpy as np
import torch.nn as nn
from dataload import *
from block_library import *
from args import parser
args = parser.parse_args()



class architecture(nn.Module):
	def __init__(self, block_cfg, size_cfg, ds_cfg):
		super(architecture, self).__init__()
		self.block_cfg = block_cfg
		self.size_cfg = size_cfg
		self.ds_cfg = ds_cfg
		self.num_classes, self.in_feature = num_classes_in_feature()

		self.features = nn.Sequential(
			self._make_layer())

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
		x = self.features(x)
		x = x.view(x.size(0), -1)
		out = self.classifier(x)
		return out