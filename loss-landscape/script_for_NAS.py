import os
import logging
import sys
import matplotlib
matplotlib.use('pdf')


log_path = 'log_script.txt'.format()
log_format = '%(asctime)s   %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M%p')
fh = logging.FileHandler(os.path.join('./',log_path))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("---------------------------------------------------------------------------------------------")
logging.info("                           script.py                                             ")
logging.info("---------------------------------------------------------------------------------------------")



string = '_lr=0.1_bs=64_model_epoch'

for model in ['resnet56', 'resnet56_noshort']:

	batch_size = 64

	epoch =

	"""
	------------------------------------
	Run landscape
	------------------------------------
	"""

	file = model+string+ str(epoch-1)

	command1 = 'mpirun -n 4 python plot_surface.py --mpi --cuda --model '+ model+' --x=-0.5:1.5:401 --dir_type states --model_file cifar10/trained_nets/' + file + '.t7 --plot --batch_size '+str(batch_size)
	logging.info('command: %s\n', command1)
	os.system(command1)

	command2 = 'mpirun -n 4 python plot_surface.py --mpi --cuda --model '+model+' --x=-1:1:51 --model_file cifar10/trained_nets/' + file + '.t7 --dir_type weights --xnorm filter --xignore biasbn --plot --batch_size '+str(batch_size)
	logging.info('command: %s\n', command2)
	os.system(command2)

	command3 = 'mpirun -n 4 python plot_surface.py --mpi --cuda --model '+model+' --x=-1:1:51 --y=-1:1:51 --model_file cifar10/trained_nets/' + file + '.t7 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot --batch_size '+str(batch_size)
	logging.info('command: %s\n', command3)
	os.system(command3)


	postfix = '.t7_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_[-1.0,1.0,51]x[-1.0,1.0,51].h5'

	command4 = 'python plot_2D.py --surf_file cifar10/trained_nets/' + file + postfix +' --surf_name train_loss'
	logging.info('command: %s\n', command4)
	os.system(command4)

	command5 = 'python h52vtp.py --surf_file cifar10/trained_nets/' + file + postfix + ' --surf_name train_loss --zmax  10 --log'
	logging.info('command: %s\n', command5)
	os.system(command5)



