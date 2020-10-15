import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as scio
import collections
import pandas as pd
from args import parser
args = parser.parse_args()

start_generation = 0
end_generation = 6
marker = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']
record_all = collections.defaultdict()
record1 = []
record2 = []
record3 = []
record4 = []
record5 = []
record6 = []

for gen in range(start_generation, end_generation):
    file_name = 'generation{}.mat'.format(gen)
    record = collections.defaultdict(int)
    content = scio.loadmat('../../results/mode_continual/'+file_name)['pop'][0]
    block_pop = content['block_pop'][0].flatten()
    # ds_pop =  content['ds_pop'][0].flatten()
    for idx, num in enumerate(block_pop):
        if num == 0:
            record['depthwise'] += 1
            record['conv11'] += 2
            record['layer'] +=3/60
        elif num == 1:
            record['skip_connection'] +=1
            record['conv33'] += 2
            record['layer'] +=2/60

        elif num == 2:
            record['concat'] += 3
            record['conv33'] += 3
            record['conv11'] += 3
            record['layer'] +=9/60

        elif num == 3:
            record['identity'] += 1

        #     if ds_pop ==  '1':
        #         record['reduction'] += 1
        #     elif ds_pop == '0':
        #         record['identity'] += 1
        # df = pd.DataFrame.from_dict(record)
    print(record)
    record1.append(record['depthwise'])
    record2.append(record['conv11'])
    record3.append(record['conv33'])
    record4.append(record['skip_connection'])
    record5.append(record['identity'])
    record6.append(record['concat'])

x = np.arange(start_generation, end_generation)
plt.plot(x, record1, 'o--', color='grey', alpha=0.3, label = 'Depthwise')
plt.plot(x, record2, 'x-', color='r', alpha=0.3, label ='conv11')
plt.plot(x, record3, '-', color='b', alpha=0.3, label ='conv33')
plt.plot(x, record4, '^-', color='k', alpha=0.3, label ='skip_connection')
plt.plot(x, record5, 'o-', color='y', alpha=0.3, label ='identity')
plt.plot(x, record6, 'x-',  alpha=0.3, label ='concat')

    # record_all['generation{}'.format(gen)] = record

# scio.savemat('../../results/mode_{}/analytic.mat'.format(args.mode), {'record_all': record_all})

#     print(record_all['generation{}'.format(gen)])
#     fig, ax = plt.subplots()
#     for key, item in record_all['generation{}'.format(gen)].item():
#         if key == 'depthwise':
#             rects1 = ax.bar(x - width / 2, men_means, width, label=key)
#
#         rects1 = ax.bar(x - width / 2, men_means, width, label=key)
#         rects2 = ax.bar(x + width / 2, women_means, width, label='Women')
#
# plt.scatter(block_pop[1], block_pop[0], marker = marker[gen])
#
plt.grid(axis='x', color='0.95')
plt.legend(title='Block cells')
plt.show()