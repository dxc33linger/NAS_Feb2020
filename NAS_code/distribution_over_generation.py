import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as scio

from args import parser
args = parser.parse_args()

file_name = 'POPsize50_DNA5_maxAccu0.684_minDeltaAccu-0.036.mat'
content = scio.loadmat('../../results/mode_continual/'+file_name)

record_generation = content['record_generation']

start_generation = 0
end_generation = 8
for gen in range(start_generation, end_generation):
    pop_accuracy = record_generation[gen, 1, :]
    plt.hist(pop_accuracy, bins= 'auto', alpha=0.5, label='Accuracy distribution in Generation {}'.format(gen))  # arguments are passed to np.histogram

plt.title('Accuracy distribution')
plt.xticks(np.arange(0, 1.1, step=0.1))
plt.yticks(np.arange(0, args.POP_SIZE+1, step=10))
plt.xlabel('Accuracy')
plt.ylabel('Count')
plt.legend(loc='best')
plt.savefig('../../results/mode_{}/plot_hist_accuracy_{}.png'.format(args.mode,file_name))
# plt.show()

plt.figure()
for gen in range(start_generation, end_generation):
    pop_delta_accu = record_generation[gen, 2, :]
    plt.hist(pop_delta_accu, bins= 'auto', alpha=0.5, label=r'$\Delta$Accuracy distribution in Generation {}'.format(gen))  # arguments are passed to np.histogram

plt.title(r'$\Delta$Accuracy distribution')
plt.xlabel(r'$\Delta$Accuracy')
plt.ylabel('Count')
plt.xticks(np.arange(0, 1.1, step=0.1))
plt.yticks(np.arange(0, args.POP_SIZE+1, step=10))
plt.legend(loc='best')
plt.savefig('../../results/mode_{}/plot_hist_DeltaAccuracy_{}.png'.format(args.mode,file_name))
# plt.show()

print('Done')