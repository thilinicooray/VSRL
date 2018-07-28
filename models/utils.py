import torch.nn as nn
import math
from torch.nn import init
import torch
import pdb
import numpy as np

'''def init_weight(module):
    if isinstance(module, nn.Linear):
        init.kaiming_normal_(module.weight)
        #bais cannot be handle by kaiming
        if module.bias is not None:
            init.kaiming_normal_(module.bias)'''

def init_weight(linear, pad_idx=None):
    if isinstance(linear, nn.Conv2d):
        init.xavier_normal(linear.weight)
        '''n = linear.kernel_size[0] * linear.kernel_size[1] * linear.out_channels
        linear.weight.data.normal_(0, math.sqrt(2. / n))'''
    if isinstance(linear, nn.Linear):
        init.xavier_normal(linear.weight)
    if isinstance(linear, nn.Embedding):
        init.xavier_normal(linear.weight)
        linear.weight.data[pad_idx] = 0

def init_gru_cell(input):

    weight = eval('input.weight_ih')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform(weight, -bias, bias)
    weight = eval('input.weight_hh')
    bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
    nn.init.uniform(weight, -bias, bias)

    if input.bias:
        weight = eval('input.bias_ih' )
        weight.data.zero_()
        weight.data[input.hidden_size: 2 * input.hidden_size] = 1
        weight = eval('input.bias_hh')
        weight.data.zero_()
        weight.data[input.hidden_size: 2 * input.hidden_size] = 1


def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_non_linearity=True):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_non_linearity:
            if batch_norm == 'batch':
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())
        #print '[Saved]: {}'.format(k)


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    try:
        for k, v in net.state_dict().items():
            if k in h5f:
                param = torch.from_numpy(np.asarray(h5f[k]))
                v.copy_(param)
                print ('[Copied]: {}'.format(k))
            else:
                print ('[Missed]: {}'.format(k))
    except Exception as e:
        pdb.set_trace()
        print('[Loaded net not complete] Parameter[{}] Size Mismatch...'.format(k))

def format_dict(d, s, p):
    rv = ""
    for (k,v) in d.items():
        if len(rv) > 0: rv += " , "
        rv+=p+str(k) + ":" + s.format(v*100)
    return rv

def get_only_relevant_roles(gt_labels):

    actual_ids = []

    for role in gt_labels:
        val = role[role > 0]
        if len(val) > 0:
            _, id = torch.max(torch.unsqueeze(role,1), 0)
            actual_ids.append(id.item())
    return torch.tensor(actual_ids)