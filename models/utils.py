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

def init_weight(linear, val = None, pad_idx=None):
    if isinstance(linear, nn.Conv2d):
        n = linear.kernel_size[0] * linear.kernel_size[1] * linear.out_channels
        linear.weight.data.normal_(0, math.sqrt(2. / n))
    if isinstance(linear, nn.Linear):
        if val is None:
            fan = linear.in_features +  linear.out_features
            spread = math.sqrt(2.0) * math.sqrt( 2.0 / fan )
        else:
            spread = val
        linear.weight.data.uniform_(-spread,spread)
        if linear.bias is not None:
            linear.bias.data.uniform_(-spread,spread)
    if isinstance(linear, nn.Embedding):
        bias = np.sqrt(3.0 / linear.weight.size(1))
        nn.init.uniform(linear.weight, -bias, bias)
        linear.weight.data[pad_idx] = 0

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