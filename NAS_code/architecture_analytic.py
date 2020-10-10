import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as scio

from args import parser
args = parser.parse_args()

start_generation = 0
end_generation = 6
marker = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']

for gen in range(start_generation, end_generation):
    file_name = 'generation{}_finish.mat'.format(gen)
    content = scio.loadmat('../../results/mode_continual/'+file_name)['pop'][0]
    block_pop = content['block_pop'][0].flatten()
    num = np.arange(block_pop.shape[0])
    block_pop = np.vstack((block_pop, num))
    print(block_pop.shape)
    plt.scatter(block_pop[1], block_pop[0], marker = marker[gen])

plt.show()