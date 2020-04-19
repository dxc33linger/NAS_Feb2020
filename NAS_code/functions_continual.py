import logging
import re
from collections import OrderedDict

from torch import optim

from utils import progress_bar
import torch
import torch.nn as nn
import numpy as np
import logging

import numpy as np
import os
import torch.nn as nn
from torch.autograd import Variable

from args import parser
args = parser.parse_args()


class ContinualNN(object):
    def __init__(self, net):
        self.net = net

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, args.lr_step_size, gamma=args.gamma)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists('../../mask_library/'):
        os.mkdir('../../mask_library')
    if not os.path.exists('../../results/'):
        os.mkdir('../../results')
    def sensitivity_rank_taylor_filter(self, threshold):
        self.net.eval()
        mask_list_4d = []
        mask_list_R_4d = []
        threshold_list = []
        gradient_list = []
        weight_list = []
        taylor_list = []
        i = 0
        logging.info("Obtain top {} position according to {} ........".format(threshold, args.score))

        for m in self.net.modules():
            # print(m)
            if type(m) != nn.Sequential and i != 0:
                if isinstance(m, nn.Conv2d):
                    total_param = m.weight.data.shape[0]
                    weight_copy = m.weight.data.abs().clone().cpu().numpy()
                    if args.score == 'abs_w':
                        taylor = np.sum(weight_copy,  axis=(1 ,2 ,3))
                    elif args.score == 'abs_grad':
                        grad_copy = m.weight.grad.data.abs().clone().cpu().numpy()
                        taylor = np.sum(grad_copy, axis=(1 ,2 ,3))
                    elif args.score == 'grad_w':
                        grad_copy = m.weight.grad.data.abs().clone().cpu().numpy()
                        taylor = np.sum(weight_copy *grad_copy, axis=(1, 2, 3))

                    num_keep = int(total_param * threshold)
                    arg_max = np.argsort(taylor) # Returns the indices sort an array. small->big
                    arg_max_rev = arg_max[::-1][:num_keep]
                    thre = taylor[arg_max_rev[-1]]
                    mask = np.zeros(weight_copy.shape)
                    mask_R = np.ones(weight_copy.shape)
                    mask[arg_max_rev.tolist(), :, :, :] = 1.0 ## mask = 0 means postions to be updated
                    mask_R[arg_max_rev.tolist(), :, :, :] = 0.0  ## mask = 0 means postions to be updated

                    mask_list_4d.append(mask)  # 0 is more
                    mask_list_R_4d.append(mask_R)  # 1 is more
                    threshold_list.append(thre)
                    if args.score in ['abs_grad', 'grad_w']:
                        gradient_list.append(m.weight.grad.data.clone().cpu().numpy())
                    weight_list.append(m.weight.data.clone().cpu().numpy())
                    taylor_list.append(taylor)

                elif isinstance(m, nn.BatchNorm2d):
                    # bn weight
                    total_param = m.weight.data.shape[0]
                    weight_copy = m.weight.data.abs().clone().cpu().numpy()
                    if args.score == 'abs_w':
                        taylor = weight_copy# * weight_copy
                    elif args.score == 'abs_grad':
                        grad_copy = m.weight.grad.data.abs().clone().cpu().numpy()
                        taylor = grad_copy  # * weight_copy
                    elif args.score == 'grad_w':
                        grad_copy = m.weight.grad.data.abs().clone().cpu().numpy()
                        taylor = weight_copy *grad_copy  #
                    num_keep = int(total_param * threshold)
                    arg_max = np.argsort(taylor)  # Returns the indices sort an array. small->big
                    arg_max_rev = arg_max[::-1][:num_keep]
                    thre = taylor[arg_max_rev[-1]]
                    mask = np.zeros(weight_copy.shape)
                    mask_R = np.ones(weight_copy.shape)
                    mask[arg_max_rev.tolist()] = 1.0
                    mask_R[arg_max_rev.tolist()] = 0.0
                    mask_list_4d.append(mask)  # 0 is more
                    mask_list_R_4d.append(mask_R)
                    threshold_list.append(thre)
                    if args.score in ['abs_grad', 'grad_w']:
                        gradient_list.append(m.weight.grad.data.clone().cpu().numpy())
                    weight_list.append(m.weight.data.clone().cpu().numpy())
                    taylor_list.append(taylor)

                    ##bn bias
                    total_param = m.bias.data.shape[0]
                    weight_copy = m.bias.data.abs().clone().cpu().numpy()
                    if args.score == 'abs_w':
                        taylor = weight_copy# * weight_copy
                    elif args.score == 'abs_grad':
                        grad_copy = m.bias.grad.data.abs().clone().cpu().numpy()
                        taylor = grad_copy  # * weight_copy
                    elif args.score == 'grad_w':
                        grad_copy = m.bias.grad.data.abs().clone().cpu().numpy()
                        taylor = weight_copy *grad_copy  #
                    num_keep = int(total_param * threshold)
                    arg_max = np.argsort(taylor)  # Returns the indices sort an array. small->big
                    arg_max_rev = arg_max[::-1][:num_keep]
                    thre = taylor[arg_max_rev[-1]]
                    mask = np.zeros(weight_copy.shape)
                    mask_R = np.ones(weight_copy.shape)
                    mask[arg_max_rev.tolist()] = 1.0
                    mask_R[arg_max_rev.tolist()] = 0.0
                    mask_list_4d.append(mask)
                    mask_list_R_4d.append(mask_R)
                    threshold_list.append(thre)
                    if args.score in ['abs_grad', 'grad_w']:
                        gradient_list.append(m.bias.grad.data.clone().cpu().numpy())
                    weight_list.append(m.bias.data.clone().cpu().numpy())
                    taylor_list.append(taylor)

                    # # running_mean
                    total_param = m.running_mean.data.shape[0]
                    weight_copy = m.running_mean.data.abs().clone().cpu().numpy()
                    num_keep = int(total_param * threshold)
                    arg_max = np.argsort(taylor)  # Returns the indices sort an array. small->big
                    arg_max_rev = arg_max[::-1][:num_keep]
                    thre = taylor[arg_max_rev[-1]]
                    mask = np.zeros(weight_copy.shape)
                    mask_R = np.ones(weight_copy.shape)
                    mask[arg_max_rev.tolist()] = 1.0
                    mask_R[arg_max_rev.tolist()] = 0.0
                    mask_list_4d.append(mask)
                    mask_list_R_4d.append(mask_R)
                    threshold_list.append(thre)
                    if args.score in ['abs_grad', 'grad_w']:
                        gradient_list.append(m.bias.grad.data.clone().cpu().numpy())
                    weight_list.append(m.bias.data.clone().cpu().numpy())
                    taylor_list.append(taylor)

                    total_param = m.running_var.data.shape[0]
                    weight_copy = m.running_var.data.abs().clone().cpu().numpy()
                    taylor = weight_copy  # * weight_copy
                    num_keep = int(total_param * threshold)
                    arg_max = np.argsort(taylor)  # Returns the indices sort an array. small->big
                    arg_max_rev = arg_max[::-1][:num_keep]
                    thre = taylor[arg_max_rev[-1]]
                    mask = np.zeros(weight_copy.shape)
                    mask_R = np.ones(weight_copy.shape)
                    mask[arg_max_rev.tolist()] = 1.0
                    mask_R[arg_max_rev.tolist()] = 0.0
                    mask_list_4d.append(mask)
                    mask_list_R_4d.append(mask_R)
                    threshold_list.append(thre)
                    if args.score in ['abs_grad', 'grad_w']:
                        gradient_list.append(m.bias.grad.data.clone().cpu().numpy())
                    weight_list.append(m.bias.data.clone().cpu().numpy())
                    taylor_list.append(taylor)

                    # if torch.__version__ == '1.0.1.post2': # torch 1.0 bn.num_tracked
                    mask_list_4d.append(np.zeros(1))
                    mask_list_R_4d.append(np.zeros(1))
                    threshold_list.append(np.zeros(1))
                    gradient_list.append(np.zeros(1))
                    weight_list.append(np.zeros(1))
                    taylor_list.append(taylor)

                elif isinstance(m, nn.Linear): # neuron-wise
                    # print('linear', m)
                    # linear weight
                    weight_copy = m.weight.data.abs().clone().cpu().numpy()
                    if args.score == 'abs_w':
                        taylor = np.sum(weight_copy, axis = 1)
                    elif args.score == 'abs_grad':
                        grad_copy = m.weight.grad.data.abs().clone().cpu().numpy()
                        taylor = np.sum(grad_copy, axis = 1)
                    elif args.score == 'grad_w':
                        grad_copy = m.weight.grad.data.abs().clone().cpu().numpy()
                        taylor = np.sum(weight_copy *grad_copy, axis = 1)
                    num_keep = int(m.weight.data.shape[0] * threshold)
                    arg_max = np.argsort(taylor)  # Returns the indices that would sort an array. small->big
                    arg_max_rev = arg_max[::-1][:num_keep]
                    thre = taylor[arg_max_rev[-1]]
                    mask = np.zeros(weight_copy.shape)
                    mask_R = np.ones(weight_copy.shape)
                    mask[arg_max_rev.tolist(), :] = 1.0
                    mask_R[arg_max_rev.tolist(), :] = 0.0
                    mask_list_4d.append(mask)  # 0 is more
                    mask_list_R_4d.append(mask_R)  # 1 is more
                    threshold_list.append(thre)
                    if args.score in ['abs_grad', 'grad_w']:
                        gradient_list.append(m.weight.grad.data.clone())
                    weight_list.append(m.weight.data.clone())
                    taylor_list.append(taylor)

                    # linear bias
                    weight_copy = m.bias.data.abs().clone().cpu().numpy()
                    if args.score == 'abs_w':
                        taylor = weight_copy# * weight_copy
                    elif args.score == 'abs_grad':
                        grad_copy = m.bias.grad.data.abs().clone().cpu().numpy()
                        taylor = grad_copy  # * weight_copy
                    elif args.score == 'grad_w':
                        grad_copy = m.bias.grad.data.abs().clone().cpu().numpy()
                        taylor = weight_copy *grad_copy  #
                    arg_max = np.argsort(taylor)
                    arg_max_rev = arg_max[::-1][:num_keep]
                    thre = taylor[arg_max_rev[-1]]
                    mask = np.zeros(weight_copy.shape[0])
                    mask_R = np.ones(weight_copy.shape[0])
                    mask[arg_max_rev.tolist()] = 1.0
                    mask_R[arg_max_rev.tolist()] = 0.0
                    mask_list_4d.append(mask)
                    mask_list_R_4d.append(mask_R)
                    threshold_list.append(thre)
                    if args.score in ['abs_grad', 'grad_w']:
                        gradient_list.append(m.bias.grad.data.clone())
                    weight_list.append(m.bias.data.clone())
                    taylor_list.append(taylor)
            i += 1
        all_mask = []
        all_mask.append(mask_list_4d)
        all_mask.append(mask_list_R_4d)
        logging.info('Got some lists: mask/maskR/threshold/gradient/weight/{}'.format(args.score))
        logging.info \
            ('mask length: {} // threshold_list length:{} // gradient list: length {} // weight list: length {} // taylor_list: length {}'.
                     format(len(mask_list_4d), len(threshold_list), len(gradient_list), len(weight_list), len(taylor_list)))  # 33

        gradient_dict, threshold_dict, mask_dict, mask_R_dict, taylor_dict = self.convert_list_to_dict(gradient_list, threshold_list, all_mask, taylor_list)
        return all_mask, threshold_dict, mask_dict, mask_R_dict, taylor_dict


    def convert_list_to_dict(self, gradient_list, threshold_list, mask_file,
                             taylor_list):  # test drift range of the rest parameters
        threshold_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
        gradient_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
        mask_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
        mask_R_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
        taylor_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
        # print(threshold_dict.keys())
        # print(len(threshold_dict))
        assert len(threshold_list) == len(threshold_dict), 'Dictionary <-> list does not match'

        idx = 0

        mask_list = []
        mask_list_R = []
        for i in range(len(mask_file[0])):
            mask_list.append(torch.from_numpy(mask_file[0][i]).type(torch.cuda.FloatTensor))
            mask_list_R.append(torch.from_numpy(mask_file[1][i]).type(torch.cuda.FloatTensor))

        for layer_name, param in self.net.state_dict().items():
            # print(layer_name, param.shape)
            # print(idx)
            # print(threshold_list[idx])
            threshold_dict[layer_name] = threshold_list[idx]
            if args.score in ['abs_grad', 'grad_w']:
                gradient_dict[layer_name] = gradient_list[idx]
            mask_dict[layer_name] = mask_list[idx]
            mask_R_dict[layer_name] = mask_list_R[idx]
            taylor_dict[layer_name] = taylor_list[idx]
            idx += 1

        logging.info('Several lists are converted into dictionaries (in torch.cuda)\n\n')
        return gradient_dict, threshold_dict, mask_dict, mask_R_dict, taylor_dict


    def train_with_frozen_filter(self, epoch, trainloader, mask_dict, mask_dict_R, path_postfix=''):
        param_old_dict = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
        for layer_name, param in self.net.state_dict().items():
            param_old_dict[layer_name] = param.clone()

        self.net.train()
        logging.info('\nEpoch: %d lr: %s' % (epoch, self.scheduler.get_lr()))
        train_loss = 0.0
        correct = 0
        total = 0
        self.optimizer.step()
        self.scheduler.step()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs_var = Variable(inputs)
            targets_var = Variable(targets)

            self.optimizer.zero_grad()
            outputs = self.net(inputs_var)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total
            # apply mask
            param_processed = OrderedDict([(k, None) for k in self.net.state_dict().keys()])
            for layer_name, param_new in self.net.state_dict().items():
                param_new = param_new.type(torch.cuda.FloatTensor)
                param_old_dict[layer_name] = param_old_dict[layer_name].type(torch.cuda.FloatTensor)
                # print(layer_name)
                if re.search('conv', layer_name):
                    param_processed[layer_name] = Variable(torch.mul(param_old_dict[layer_name], mask_dict[layer_name]) +
                                                           torch.mul(param_new, mask_dict_R[layer_name]),
                                                           requires_grad=True)



                # print('new\n', param_new[0:3, 0, :, :])

                elif re.search('shortcut', layer_name):
                    if len(param_new.shape) == 4:  # conv in shortcut
                        param_processed[layer_name] = Variable(
                            torch.mul(param_old_dict[layer_name], mask_dict[layer_name]) +
                            torch.mul(param_new, mask_dict_R[layer_name]), requires_grad=True)
                    else:
                        param_processed[layer_name] = Variable(param_new, requires_grad=True)
                elif re.search('linear', layer_name):
                    param_processed[layer_name] = Variable(torch.mul(param_old_dict[layer_name], mask_dict[layer_name]) +
                                                           torch.mul(param_new, mask_dict_R[layer_name]),
                                                           requires_grad=True)

                else:
                    param_processed[layer_name] = Variable(param_new, requires_grad=True)  # num_batches_tracked
            self.net.load_state_dict(param_processed)
            progress_bar(batch_idx, len(trainloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Train' % (
                train_loss / (batch_idx + 1), acc, correct, total))

        return correct / total


    def mask_frozen_weight(self, maskR):
        param_processed = OrderedDict([(k, None) for k in self.net.state_dict().keys()])

        for layer_name, param in self.net.state_dict().items():
            param_processed[layer_name] = Variable(torch.mul(param, maskR[layer_name]), requires_grad=False)
        self.net.load_state_dict(param_processed)


    def AND_twomasks(self, mask_dict_1, mask_dict_2, maskR_dict_1, maskR_dict_2):
        maskR_processed = OrderedDict([(k, None) for k in maskR_dict_1.keys()])
        mask_processed = OrderedDict([(k, None) for k in maskR_dict_1.keys()])
        for layer_name, mask in maskR_dict_1.items():
            maskR_processed[layer_name] = torch.mul(maskR_dict_1[layer_name], maskR_dict_2[layer_name])
            mask_processed[layer_name] = torch.add(mask_dict_1[layer_name], mask_dict_2[layer_name])
        return mask_processed, maskR_processed

