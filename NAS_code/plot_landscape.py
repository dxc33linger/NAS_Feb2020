
import copy
import h5py
import torch
import time
import os
import sys
import numpy as np
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import re
import os
import h5_util




class landscape():
    def __init__(self,net, model_name='current'):

        self.model_name = model_name
        self.net= net
        self.net.eval()
        if not os.path.isdir('../../results/landscape'):
            os.system('mkdir ../../results/landscape')


        self.dir_file    = '../../results/landscape/{}_dir.h5'.format(self.model_name)
        self.surf_file= '../../results/landscape/{}_surface.h5'.format(self.model_name)
    
    def generate_h5(self, trainloader=None, testloader=None, useTraindata= True):
        w = self.get_weights()
        s = copy.deepcopy(self.net.state_dict())
     
        self.setup_direction()
 
        self.setup_surface_file()
        d = self.load_directions() 
        if useTraindata:
            assert trainloader is not None
            self.crunch( w, s, d, trainloader, 'train_loss', 'train_acc')
        else:
            assert testloader is not None
            self.crunch( w, s, d, testloader, 'test_loss', 'test_acc')

        f = h5py.File(self.surf_file, 'r')

        if useTraindata:
            os.system('python h52vtp.py --surf_file {} --surf_name train_loss --output {} --zmax {} --log'.format(self.surf_file,self.model_name,100))
        else:
            os.system('python h52vtp.py --surf_file {} --surf_name test_loss  --output {} --zmax {} --log'.format(self.surf_file,self.model_name,100))




    def eval_loss(self,net, criterion, loader, use_cuda=False):

        correct = 0
        total_loss = 0
        total = 0 # number of samples
        num_batch = len(loader)

        if use_cuda:
            net.cuda()
        net.eval()

        with torch.no_grad():
            if isinstance(criterion, nn.CrossEntropyLoss):
                for batch_idx, (inputs, targets) in enumerate(loader):
                    batch_size = inputs.size(0)
                    total += batch_size
                    inputs = Variable(inputs)
                    targets = Variable(targets)
                    if use_cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()
                    outputs = net(inputs)
                    #outputs,_ = net(inputs)

                    loss = criterion(outputs, targets)



                    total_loss += loss.item()*batch_size
                    _, predicted = torch.max(outputs.data, 1)
                    correct += predicted.eq(targets).sum().item()

        return total_loss/total, 100.*correct/total




    def crunch(self, w, s, d, dataloader, loss_key, acc_key):


        f = h5py.File(self.surf_file, 'r+')
        losses, accuracies = [], []
        xcoordinates = f['xcoordinates'][:]
        ycoordinates = f['ycoordinates'][:] 

        if loss_key not in f.keys():
            shape = (len(xcoordinates),len(ycoordinates))
            losses = -np.ones(shape=shape)
            accuracies = -np.ones(shape=shape)
            f[loss_key] = losses
            f[acc_key] = accuracies
        else:
            losses = f[loss_key][:]
            accuracies = f[acc_key][:]
        

        inds, coords, inds_nums = self.get_job_indices(losses, xcoordinates, ycoordinates)
        criterion = nn.CrossEntropyLoss()


        # Loop over all uncalculated loss values
        for count, ind in enumerate(inds):
            coord = coords[count]
            self.set_weights(w, d, coord)

            loss, acc = self.eval_loss(self.net, criterion, dataloader, True)

            losses.ravel()[ind] = loss
            accuracies.ravel()[ind] = acc          
            f[loss_key][:] = losses
            f[acc_key][:] = accuracies
            f.flush()

            print('Evaluating   %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f' % (
                    count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                    acc_key, acc))

        f.close()






    def get_job_indices(self,vals, xcoordinates, ycoordinates):
       
        inds, coords = self.get_unplotted_indices(vals, xcoordinates, ycoordinates)
        return inds, coords, len(inds)



    def get_unplotted_indices(self, vals, xcoordinates, ycoordinates):

        inds = np.array(range(vals.size))
        inds = inds[vals.ravel() <= 0]
        xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
        s1 = xcoord_mesh.ravel()[inds]
        s2 = ycoord_mesh.ravel()[inds]
        return inds, np.c_[s1,s2]





    def set_weights(self, weights, directions, step):

    
        assert step is not None, 'If a direction is specified then step must be specified as well'
        dx = directions[0]
        dy = directions[1]
        changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        for (p, w, d) in zip(self.net.parameters(), weights, changes):

            #p.data = w + torch.Tensor(d).type(type(w))
            p.data= w + torch.Tensor(d).to('cuda')

    def load_directions(self):

        f = h5py.File(self.dir_file, 'r')
        xdirection = h5_util.read_list(f, 'xdirection')
        ydirection = h5_util.read_list(f, 'ydirection')
        directions = [xdirection, ydirection]
        return directions

    def setup_surface_file(self):
        if os.path.exists(self.surf_file):
            os.system('rm {}'.format(self.surf_file))

        f = h5py.File(self.surf_file, 'a')
        f['dir_file'] = self.dir_file
        xcoordinates = np.linspace(-1, 1, num=51)
        f['xcoordinates'] = xcoordinates
        ycoordinates = np.linspace(-1, 1, num=51)
        f['ycoordinates'] = ycoordinates
        f.close()
        return self.surf_file


    def get_weights(self):
        """ Extract parameters from net, and return a list of tensors"""
        return [p.data for p in self.net.parameters()]

    def get_random_weights(self,weights):
        """
            Produce a random direction that is a list of random Gaussian tensors
            with the same shape as the network's weights, so one direction entry per weight.
        """
        return [torch.randn(w.size()) for w in weights]
        
    def create_random_direction(self, dir_type='weights', ignore='biasbn', norm='filter'):
    


        weights = self.get_weights() # a list of parameters.
        direction = self.get_random_weights(weights)
        self.normalize_directions_for_weights(direction, weights, norm, ignore)

        return direction



    def normalize_direction(self, direction, weights, norm='filter'):

        if norm == 'filter':
            for d, w in zip(direction, weights):           
                d.mul_(w.norm()/(d.norm() + 1e-10))



    def normalize_directions_for_weights(self,direction, weights, norm='filter', ignore='biasbn'):
        """
            The normalization scales the direction entries according to the entries of weights.
        """
        assert(len(direction) == len(weights))
        for d, w in zip(direction, weights):
            if d.dim() <= 1:
                if ignore == 'biasbn':
                    d.fill_(0) # ignore directions for weights with 1 dimension        
                else:
                    d.copy_(w) # keep directions for weights/bias that are only 1 per node
            else:
                self.normalize_direction(d, w, norm)
            


    def setup_direction(self):

        print('-------------------------------------------------------------------')
        print('setup_direction')
        print('-------------------------------------------------------------------')
        
        if os.path.exists(self.dir_file):
            os.system('rm {}'.format(self.dir_file))

        f = h5py.File(self.dir_file,'w') # create file, fail if exists
        print("Setting up the plotting directions...")            
        xdirection = self.create_random_direction(self.net)   
        ydirection = self.create_random_direction(self.net)
        h5_util.write_list(f, 'xdirection', xdirection)
        h5_util.write_list(f, 'ydirection', ydirection)

        f.close()
        print ("direction file created: %s" % self.dir_file)



