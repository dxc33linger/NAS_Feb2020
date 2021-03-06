import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace
import matplotlib.pyplot as plt
from torchvision.models import AlexNet


model = torch.load('../../results/50,10,10,10,10,10_model_with_best_fitness')


x = torch.randn(1, 3, 32, 32).requires_grad_(False)
y = model(x.cuda())

params = list(model.parameters())
k = 0
for i in params:
        l = 1
        print("Layer structure：" + str(list(i.size())))
        for j in i.size():
                l *= j
        print("Layer #Param：" + str(l) + ' B')
        k = k + l
print("Total #Param：" + str(k)+ ' B')


vis_graph = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
vis_graph.view()


