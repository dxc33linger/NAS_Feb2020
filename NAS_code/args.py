import argparse
parser = argparse.ArgumentParser(description='NAS dxc')
parser.add_argument("--gpu", type=str, default = '0', help='GPU')
parser.add_argument("--seed", type=int, default = 1, help='random seed')
parser.add_argument("--resume", type=bool, default = False, help='random seed')

# training
parser.add_argument('--lr', default = 0.1, type=float, help='learning rate')
parser.add_argument('--lr_step_size', default = 25, type=int, help='learning rate decay step')
parser.add_argument('--gamma', default = 0.5, type=float, help='learning rate decay rate')
parser.add_argument("--batch_size", type=int, default = 128, help='batch size')
parser.add_argument("--num_epoch", type=int, default = 5, help='training epochs')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')

# dataset
parser.add_argument('--dataset', default = 'cifar100', type=str, choices=['cifar10','cifar100', 'SVHN', 'imagenet', 'mnist'])
parser.add_argument("--mode", type=str, default = 'continual', choices=['regular','continual'], help='regular means full-dataset training; continual means continual laearning')
parser.add_argument('--shuffle', type=bool, default = True, help='dataset shuffle')
parser.add_argument('--classes_per_task', type=int, default = 1, choices=[1, 10], help='#class per task')
parser.add_argument('--task_division', type=str, default = '10,10')
parser.add_argument("--K", type=int, default = 10,  help="K-fold validation")
parser.add_argument("--CV", type=bool, default = False,  help="turn on cross-validation?")

# noise
parser.add_argument('--alpha', type=float, default = 0.1, help='noise distribution range')

# genetic algorithm
# parser.add_argument("--max_block", type=int, default = 6, help='max number of blocks inside a module')
parser.add_argument("--DNA_SIZE", type=int, default = 5, help='DNA size, i.e., number of layers within one module')  # #layers = DNA_SIZE * 3 + 2(reduction)+ 2(head-conv + FC)
parser.add_argument("--POP_SIZE", type=int, default = 50, help='population size')
parser.add_argument("--N_GENERATIONS", type=int, default = 2, help='number of generations')
parser.add_argument('--CROSS_RATE', type=float, default = 0.7, help='crossover rate')
parser.add_argument('--MUTATION_RATE', type=float, default = 0.05, help='mutation rate')

parser.add_argument("--cfg", type=int, default = 55)
