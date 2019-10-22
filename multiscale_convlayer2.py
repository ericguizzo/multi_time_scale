from __future__ import print_function
import numpy as np
import torch
import sys
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as utils
import matplotlib.pyplot as plt

torch.set_printoptions(threshold=1000000000000)

class MultiscaleConv2d(nn.Module):
    def __init__(self, in_depth, out_depth, kernel_size, padding=0, stride=1, scale_factors=[1.],
                 output_type='pooled_map', stretch_penality_lambda=0., training_mode='train_and_eval'):
        super(MultiscaleConv2d, self).__init__()
        self.training_mode = training_mode
        self.scale_factors = scale_factors
        self.layer_names = ['self.conv_shifted_0']
        self.stretch_penality_lambda = stretch_penality_lambda
        self.output_type = output_type
        #original layer creation
        self.conv_shifted_0 = nn.Conv2d(in_depth, out_depth, kernel_size=kernel_size, stride=stride, padding=padding)
        #dynamic creation of stretched layers
        self.layer_names = ['self.conv_shifted_0']
        for i in range(len(self.scale_factors)):
            name = 'self.conv_shifted_'+ str(i+1)
            self.layer_names.append(name)
        i=0
        dummy_weights = torch.zeros(size=kernel_size)
        dummy_weights = dummy_weights.reshape((1,1,dummy_weights.shape[-2], dummy_weights.shape[-1]))
        for layer in self.layer_names:
            #define layer parameters
            if layer != self.layer_names[0]:
                temp_weights = nn.functional.interpolate(dummy_weights, scale_factor=self.scale_factors[i], mode='bilinear')
                temp_kernel_size = temp_weights.shape[-2], temp_weights.shape[-1]
                #create_layer
                create_layer = layer + ' = nn.Conv2d(' + str(in_depth) + ',' + str(out_depth) + ', stride=' + str(stride) + ', kernel_size=' + str(temp_kernel_size) + ', padding=' + str(padding) + ')'
                exec(create_layer)
                i += 1


    def forward(self, x, training_state=True):
        #index 0 is always the original layer
        #trainable conv layer (only for computing weights and biases)
        #f_map0 = locals()
        f_map0 = self.conv_shifted_0(x)

        orig_dim = f_map0.shape
        input_weights = self.conv_shifted_0.weight
        input_bias = self.conv_shifted_0.bias

        #apply stretched weights
        self.layer_names = ['self.conv_shifted_0']
        for i in range(len(self.scale_factors)):
            name = 'self.conv_shifted_'+ str(i+1)
            self.layer_names.append(name)

        i=0
        for layer in self.layer_names:
            #define layer parameters
            if layer != self.layer_names[0]:
                #depth = input_weights.shape[1]
                temp_weights = nn.functional.interpolate(input_weights, scale_factor=self.scale_factors[i], mode='bilinear')  #up/downsampling
                temp_kernel_size = temp_weights.shape[-2], temp_weights.shape[-1]
                #apply resampled kernel weights weights
                apply_weights = layer + '.weight.data = temp_weights'
                exec(apply_weights)
                #apply bias
                apply_bias = layer + '.bias.data = input_bias'
                exec(apply_bias)
                #apply retain graph **probably useless**
                apply_grad = layer + '.retain_graph = True'
                #exec(apply_grad)
                i += 1

        #dynamic forward computation
        i = 0
        fmap_names = []
        for layer in self.layer_names:
            if layer == 'self.conv_shifted_0':
                temp_fmap_name = 'f_map' + str(i)
                fmap_names.append(temp_fmap_name)
            else:
                temp_fmap_name = 'f_map' + str(i+1)
                fmap_names.append(temp_fmap_name)
                temp_computation = temp_fmap_name + ' = ' + layer + '(x)'
                exec(temp_computation)
                i += 1

        #resize feature_maps **original feature map dim is kept**
        for i in range(len(fmap_names)):
            if i >= 1:
                resize_string = 'locals()[fmap_names[' + str(i) + ']]' + ' = nn.functional.interpolate(locals()[fmap_names['+ str(i) + ']], size=' + str((f_map0.shape[-2], f_map0.shape[-1])) + ', mode=\'bilinear\')'
                exec(resize_string)

        #reshape feature maps ** from (batch, channels, time, freq) to (batch, channel*time, freq)
        #this permits to use standard 3d pooling layer when there is more than one channel
        for i in range(len(fmap_names)):
            if i != 0:
                #with relu
                reshape_string = 'locals()[fmap_names[' + str(i) + ']]' + ' = locals()[fmap_names[' + str(i) + ']].view(f_map0.shape[0], 1, f_map0.shape[1] * f_map0.shape[2], f_map0.shape[3])'
                #reshape_string = 'locals()[fmap_names[' + str(i) + ']]' + ' = locals()[fmap_names[' + str(i) + ']].view(f_map0.shape[0], 1, f_map0.shape[1] * f_map0.shape[2], f_map0.shape[3])'
                exec(reshape_string)
        f_map0 = f_map0.view(f_map0.shape[0], 1, f_map0.shape[1] * f_map0.shape[2], f_map0.shape[3])

        #penalize stretched fmaps: multiply pixels * 1-abs(log(stretchfactor)) * lambda
        #this means that the more the stretchfactor is distant from 1, the more the fmap is penalized
        #lambda is a fixed constant
        if self.stretch_penality_lambda != 0.:
            for i in range(len(fmap_names)-1):
                curr_scale = torch.tensor(self.scale_factors[i][0]).float()
                curr_penality = 1. - torch.abs(torch.log(curr_scale)) * self.stretch_penality_lambda
                penality_string = 'locals()[fmap_names[' + str(i+1) + ']]' + ' = torch.mul(locals()[fmap_names[' + str(i+1) + ']], curr_penality)'
                exec(penality_string)

        #concatenate feature maps **create one channel for every stretched fmap**
        i = 0
        cat_tuple = '('
        for fmap in fmap_names:
            cat_tuple += 'locals()[fmap_names[' + str(i) + ']]'
            if i < len(fmap_names)-1:
                cat_tuple += ','
            i += 1
        cat_tuple += ')'
        cat_tuple = eval(cat_tuple)
        try:
            x = torch.cat(cat_tuple, 1)
        except TypeError:
            raise ValueError('Only one stretch factor found: for this behavior use regular 2dConv layer')

        #3d pooling (select best stretch for every pixel in feature maps)
        self.pool = nn.MaxPool3d(kernel_size=[len(self.layer_names),1,1])
        pool_matrix = self.pool(x)

        #reshape again matrices to original shape: RESHAPE feature maps ** from (batch, channel*time, freq) to (batch, channels, time, freq)
        pool_matrix = pool_matrix.view(orig_dim)
        for i in range(len(fmap_names)):
            if i != 0:
                reshape_string = 'locals()[fmap_names[' + str(i) + ']]' + ' = locals()[fmap_names[' + str(i) + ']].view(orig_dim)'
                exec(reshape_string)
        f_map0 = f_map0.view(orig_dim)

        #compute compare matrices ** bool '==' between pooled fmap and the original maps**
        cmp_names = []
        for i in range(len(fmap_names)):
            tmp_cmpname = 'compare_matrix_' + str(i)
            cmp_names.append(tmp_cmpname)
            cmp_build_string1 = tmp_cmpname + ' = locals()[fmap_names[' + str(i) + ']] == pool_matrix'
            cmp_build_string2 = tmp_cmpname + ' = ' + tmp_cmpname + '.clone().float()'
            exec(cmp_build_string1)
            exec(cmp_build_string2)

        #compute perc of used stretch factors
        self.perc_stretches = []
        tot_pixels = torch.prod(torch.tensor(locals()[cmp_names[0]].shape))
        for i in range(len(cmp_names)):
            curr_perc = torch.sum(locals()[cmp_names[i]]) / tot_pixels
            self.perc_stretches.append(curr_perc)

        #multiply cmp matrices by stretch_factor
        for i in range(len(cmp_names)-1):
            curr_scale = self.scale_factors[i][0]
            mul_string = 'locals()[cmp_names[' + str(i+1) + ']] = torch.mul(locals()[cmp_names['+ str(i+1) + ']], torch.tensor(' + str(curr_scale) + '))'
            exec(mul_string)
            printstring = 'print (locals()[cmp_names[' + str(i+1) + ']])'

        #max between cmp matrices to obtain final index matrix
        #index matrix has the same dimension of the pooled fmap
        #contains maps the stretch factor values taken in every pixel of the pooled matrix
        for i in range(len(cmp_names)-1):
            if i == 0:
                index_matrix = torch.max(locals()[cmp_names[i]],locals()[cmp_names[i+1]])
            else:
                index_matrix = torch.max(locals()[cmp_names[i+1]], index_matrix)
        index_matrix = torch.log(index_matrix)

        #interleaving pooled feature maps and and stretch maps
        output_matrix = []
        #SELECT OUTPUT TYPE
        if self.output_type == 'pooled_map':
            output_matrix = pool_matrix

        if self.output_type == 'concat_fmaps':
            #concatenate feature maps along time dimension
            i = 0
            cat_tuple = '('
            for fmap in fmap_names:
                cat_tuple += 'locals()[fmap_names[' + str(i) + ']]'
                if i < len(fmap_names)-1:
                    cat_tuple += ','
                i += 1
            cat_tuple += ')'
            cat_tuple = eval(cat_tuple)
            output_matrix = torch.cat(cat_tuple, 2)

        if self.output_type == 'interleave_chdim':
            merged_matrix = torch.cat((pool_matrix, index_matrix), dim=2)

            merged_matrix = merged_matrix.view((pool_matrix.shape[0],
                                                pool_matrix.shape[1] * 2,
                                                pool_matrix.shape[2],
                                                pool_matrix.shape[3]))
            output_matrix = merged_matrix

        #look at eval or training mode
        if self.training_mode == 'train_and_eval':
            #use all feature maps both in train and eval
            #!!! to be coupled with update_kernels() at the end of training loop
            pass
        if self.training_mode == 'only_eval':
            #use original feature map in training and all ones in eval
            #!!! update_kernels() should be DISABLED in trainin loop
            if training_state == True:
                output_matrix = f_map0
            else:
                output_matrix = output_matrix
        if self.training_mode == 'only_train':
            #use original feature map in training and all ones in eval
            #!!! update_kernels() should be DISABLED in trainin loop
            if training_state == True:
                output_matrix = output_matrix
            else:
                output_matrix = f_map0
        if self.training_mode == 'only_gradient':
            #use always the only original feature map
            #BUT compute the gradients for the stretched ones
            #!!! to be coupled with update_kernels at the end of training loop
            output_matrix = f_map0()

            '''
            plt.figure(1)
            plt.pcolormesh(merged_matrix[0,0,:,:].detach().numpy().reshape(pool_matrix.shape[2],pool_matrix.shape[3]))
            plt.figure(2)
            plt.pcolormesh(merged_matrix[0,1,:,:].detach().numpy().reshape(pool_matrix.shape[2],pool_matrix.shape[3]))
            plt.figure(3)
            plt.pcolormesh(merged_matrix[0,2,:,:].detach().numpy().reshape(pool_matrix.shape[2],pool_matrix.shape[3]))
            plt.figure(4)
            plt.pcolormesh(merged_matrix[0,3,:,:].detach().numpy().reshape(pool_matrix.shape[2],pool_matrix.shape[3]))
            '''
            '''
            n_maps = pool_matrix.shape[1]
            merged_matrix = torch.empty(size=(0,0,0,0))
            for i in range(n_maps):
                merged_matrix = torch.cat((merged_matrix, pool_matrix[:,i,:,:]), dim=1)
                merged_matrix = torch.cat((merged_matrix, index_matrix[:,i,:,:]), dim=1)
                print (merged_matrix.shape)
            '''
        '''
        kr = self.conv_shifted_0.weight.detach().numpy().reshape(self.conv_shifted_0.weight.shape[2],self.conv_shifted_0.weight.shape[3])
        kr1 = self.conv_shifted_1.weight.detach().numpy().reshape(self.conv_shifted_1.weight.shape[2],self.conv_shifted_1.weight.shape[3])
        kr2 = self.conv_shifted_2.weight.detach().numpy().reshape(self.conv_shifted_2.weight.shape[2],self.conv_shifted_2.weight.shape[3])


        plt.figure(1)
        plt.subplot(331)
        plt.pcolormesh(kr.T)
        plt.subplot(332)
        plt.pcolormesh(kr1.T)
        plt.subplot(333)
        plt.pcolormesh(kr2.T)

        plt.subplot(334)
        plt.pcolormesh(locals()[cmp_names[0]].detach().numpy().reshape(locals()[fmap_names[0]].shape[-2], 118).T)
        plt.subplot(335)
        plt.pcolormesh(locals()[cmp_names[1]].detach().numpy().reshape(locals()[fmap_names[1]].shape[-2], 118).T)
        plt.subplot(336)
        plt.pcolormesh(locals()[cmp_names[2]].detach().numpy().reshape(locals()[fmap_names[2]].shape[-2], 118).T)
        plt.subplot(337)
        plt.pcolormesh(locals()[fmap_names[0]].detach().numpy().reshape(locals()[fmap_names[0]].shape[-2], 118).T)
        plt.subplot(338)
        plt.pcolormesh(locals()[fmap_names[1]].detach().numpy().reshape(locals()[fmap_names[1]].shape[-2], 118).T)
        plt.subplot(339)
        plt.pcolormesh(locals()[fmap_names[2]].detach().numpy().reshape(locals()[fmap_names[2]].shape[-2], 118).T)


        plt.figure(2)
        plt.pcolormesh(pool_matrix.detach().numpy().reshape(locals()[fmap_names[2]].shape[-2], 118).T)
        plt.figure(3)

        plt.pcolormesh(index_matrix.detach().numpy().reshape(locals()[fmap_names[2]].shape[-2], 118).T)
        '''

        return output_matrix

    def update_kernels(self):
        if self.training_mode != 'only_eval':
            i=0
            weights_0 = self.conv_shifted_0.weight.clone()  #load original kernels
            bias_0 = self.conv_shifted_0.bias.clone()
            original_shape = (weights_0.shape[-2], weights_0.shape[-1])

            #load shifted kernels and up/downsample to the shape of original kernels
            i = 0
            weight_names = ['wirghts_0']
            bias_names = ['bias_0']
            for layer in self.layer_names:
                if layer != 'self.conv_shifted_0':
                    weight_names.append('weights_' + str(i))
                    bias_names.append('bias_' + str(i))
                    load_resample_weights_string = 'weights_' + str(i) + ' = nn.functional.interpolate(' + layer + '.weight.clone(), size= ' + str(original_shape) + ', mode=\'bilinear\')'
                    load_bias_string = 'bias_' + str(i) + ' = ' + layer + '.bias.clone()'
                    exec(load_resample_weights_string)
                    exec(load_bias_string)
                i += 1

            #compute average weights and bias
            for i in range(len(self.layer_names)):
                if i == 0:
                    w_sum_mtx = weights_0
                    b_sum_mtx = bias_0
                else:
                    w_sum_mtx = torch.add(w_sum_mtx, locals()[weight_names[i]])
                    b_sum_mtx = torch.add(b_sum_mtx, locals()[bias_names[i]])

            #divide by number of stretch columns
            n_stretches = float(len(self.layer_names))
            new_weights = w_sum_mtx / n_stretches
            new_bias = b_sum_mtx / n_stretches

            #update weights and bias
            self.conv_shifted_0.weight.data = new_weights
            self.conv_shifted_0.bias.data = new_bias

    def get_stretch_percs(self):
        perc_stretches = torch.tensor(self.perc_stretches).numpy()
        return perc_stretches
