
import logging
import os
import pickle
import sys
import scipy.io as scio
import functions_continual
import functions
from dataload_continual import *
from dataload_regular import *
from utils import *
import matplotlib.pyplot as plt

from args import parser
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu


log_path = 'log_add_noise.txt'.format()
log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('../../results',log_path))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("*************************************************************************************************")
logging.info("                                         add_noise.py                                             ")
logging.info("*************************************************************************************************")
logging.info("args = %s", args)

nas = functions.NAS()

# -----------------------------------------
# Prepare dataset
# -----------------------------------------

file_name = '../../results/best_model_cfg.mat'
content = scio.loadmat(file_name)
block_cfg = content['block_cfg'][0].tolist()
size_cfg = content['size_cfg'][0].tolist()
ds_cfg = content['ds_cfg'][0].tolist()
nas.net = torch.load('../../results/model_with_best_fitness')
model = nas.net
param = count_parameters_in_MB(model)
logging.info('param size = {0:.2f}MB'.format(param))
nas.initialization()


num_classes, in_feature = num_classes_in_feature()

if args.mode == 'regular': # full dataset
	raise ValueError('This function is designed for continual learning')
elif args.mode == 'continual':  ## continual learning
	task_list, total_num_task = nas.create_task()
	logging.info('Task list %s: ', task_list)
	task_division = list(map(int, args.task_division.split(",")))
	cloud_list = task_list[0: task_division[0]]
	train_cloud, test_cloud = dataload_partial(cloud_list, 0)
	total_task = len(task_division)
	all_data_list = []
	all_data_list.append(cloud_list)

for batch_idx, (data, target) in enumerate(train_cloud):
	logging.info('CLOUD re-assigned label: %s\n', np.unique(target))
	break
for batch_idx, (data, target) in enumerate(test_cloud):
	logging.info('Batch: {} CLOUD re-assigned test label: {}\n'.format(batch_idx, np.unique(target)))
	break


#------------
#  Load cloud model
# -----------------------------------------
task_id = 0 ## cloud



accu = nas.test(test_cloud)
logging.info('\n\ntest on cloud data {0:.4f}\n'.format(accu))

model_wNoise = nas.add_noise(args.alpha)
accuracy_wNoise = nas.test(test_cloud)
logging.info('Accuracy changes {:.4f} -> {:.4f} after adding Gaussian noise with alpha={}'.format(accu, accuracy_wNoise, args.alpha))

logging.info('\n\nLoading back weights without noise.....')
nas.load_weight_back()
print('Is accuracy the same?', nas.test(test_cloud))

# scio.savemat('../../results/{}/addNoise_model{}_{:.4f}to{:.4f}_alpha{}.mat'.format(args.model, args.model,accu, accuracy_wNoise, args.alpha),
#              {'model':args.model, 'NA_C0':args.NA_C0, 'epoch': args.epoch, 'epoch_edge': args.epoch_edge,
#             'lr': args.lr, 'lr_step_size':args.lr_step_size, 'classes_per_task': args.classes_per_task, 'weight_decay': args.weight_decay,  'score': args.score,
#             'dataset':args.dataset, 'task_list': task_list, 'seed':args.seed, 'shuffle':args.shuffle})