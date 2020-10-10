import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as scio

from args import parser
args = parser.parse_args()
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots,
               # expressed as a fraction of the average axis width
hspace = 0.35
# the amount of height reserved for white space between subplots,
               # expressed as a fraction of the average axis height

file_name = 'POPsize60_DNA5_maxfit_Accu0.362_DeltaAccu0.074.mat'
content = scio.loadmat('../../results/mode_continual/'+file_name)
print(file_name)
record_generation = content['record_generation']

#
start_generation = 0
end_generation = 6
col = 0
row = 0
fig, axs = plt.subplots(2, int((end_generation - start_generation) / 2) , sharey='row')
for gen in range(start_generation, end_generation):
    delta_accuracy = record_generation[gen, 2, :]
    print(np.average(delta_accuracy, axis=0),np.std(delta_accuracy, axis=0))
    # delta_accuracy = np.delete(delta_accuracy, np.where(delta_accuracy == (1.0,)), axis=0)
    print(delta_accuracy.shape)
    axs[row, col].hist(delta_accuracy, bins=30, alpha=0.7, facecolor='b', label=r'$\Delta$Accuracy distribution in Generation {}'.format(gen))  # arguments are passed to np.histogram
    axs[row, col].set_ylim(0,20)
    axs[row, col].set_xlim(0.0, 1.0)
    axs[row, col].set_ylabel('Count')
    axs[row, col].set_title(r'Gen. {}'.format(gen))
    col += 1
    if col == int((end_generation - start_generation)/2):
        row += 1
        col = 0
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
plt.savefig('../../results/plot_generation_Deltaaccuracy_{}.jpg'.format(file_name))

# plt.show()

col = 0
row = 0
fig, axs = plt.subplots(2, int((end_generation - start_generation) / 2) , sharey='row')
for gen in range(start_generation, end_generation):
    pop_accuracy = record_generation[gen, 1, :]
    print(np.average(pop_accuracy, axis=0), np.std(pop_accuracy, axis=0))

    # delta_accuracy = record_generation[gen, 2, :]
    # pop_accuracy = np.delete(pop_accuracy, np.where(delta_accuracy == (1.0,)), axis=0)
    print(pop_accuracy.shape)
    axs[row, col].hist(pop_accuracy, bins=30, alpha=0.7, facecolor='g',  label='Accuracy distribution in Generation {}'.format(gen))  # arguments are passed to np.histogram
    axs[row, col].set_ylim(0,20)
    axs[row, col].set_xlim(0.0, 0.5)
    axs[row, col].set_ylabel('Count')
    axs[row, col].set_title(r'Gen. {}'.format(gen))
    col += 1
    if col == int((end_generation - start_generation)/2):
        row += 1
        col = 0
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
plt.savefig('../../results/plot_generation_accuracy_{}.jpg'.format(file_name))
# plt.show()

col = 0
row = 0
fig, axs = plt.subplots(2, int((end_generation - start_generation) / 2) , sharey='row')
for gen in range(start_generation, end_generation):
    pop_accuracy = record_generation[gen, 1, :]
    delta_accuracy = record_generation[gen, 2, :]
    fitness = abs(1.0/(abs(delta_accuracy + 0.00001)))* pow(pop_accuracy, 1)

    print(np.average(fitness, axis=0), np.std(fitness, axis=0))
    print(fitness.shape)
    # delta_accuracy = record_generation[gen, 2, :]
    # pop_accuracy = np.delete(pop_accuracy, np.where(delta_accuracy == (1.0,)), axis=0)
    axs[row, col].hist(fitness, bins=20, alpha=0.7, facecolor='r',  label='Accuracy distribution in Generation {}'.format(gen))  # arguments are passed to np.histogram
    axs[row, col].set_ylim(0,30)
    # axs[row, col].set_xlim(0.0, 0.5)
    axs[row, col].set_ylabel('Count')
    axs[row, col].set_title(r'Gen. {}'.format(gen))
    col += 1
    if col == int((end_generation - start_generation)/2):
        row += 1
        col = 0
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
plt.savefig('../../results/plot_generation_fitness_{}.jpg'.format(file_name))

# for gen in range(start_generation, end_generation):
#     pop_accuracy = record_generation[gen, 1, :]
#     plt.hist(pop_accuracy, bins= 'auto', alpha=0.5, label='Accuracy distribution in Generation {}'.format(gen))  # arguments are passed to np.histogram
#
# plt.title('Accuracy distribution')
# plt.xticks(np.arange(0, 1.1, step=0.1))
# plt.yticks(np.arange(0, args.POP_SIZE+1, step=10))
# plt.xlabel('Accuracy')
# plt.ylabel('Count')
# plt.legend(loc='best')
# plt.savefig('../../results/plot_hist_accuracy_{}.png'.format(file_name))
# # plt.show()
#
# plt.figure()
# for gen in range(start_generation, end_generation):
#     pop_delta_accu = record_generation[gen, 2, :]
#     plt.hist(pop_delta_accu, bins= 'auto', alpha=0.5, label=r'$\Delta$Accuracy distribution in Generation {}'.format(gen))  # arguments are passed to np.histogram
#
# plt.title(r'$\Delta$Accuracy distribution')
# plt.xlabel(r'$\Delta$Accuracy')
# plt.ylabel('Count')
# plt.xticks(np.arange(0, 1.1, step=0.1))
# plt.yticks(np.arange(0, args.POP_SIZE+1, step=10))
# plt.legend(loc='best')
# plt.savefig('../../results/plot_hist_DeltaAccuracy_{}.png'.format(file_name))
# # plt.show()
# print('start_generation = {}, end_generation = {}'.format(start_generation, end_generation))
print('Done')