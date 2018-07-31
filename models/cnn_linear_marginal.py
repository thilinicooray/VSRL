import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
from . import utils
import math
import torch.nn.init as init
import random

from .action_graph import action_graph
from .faster_rcnn.utils.config import cfg

class resnet_modified_small(nn.Module):
    def __init__(self):
        super(resnet_modified_small, self).__init__()
        self.resnet = tv.models.resnet34(pretrained=True)

        #finetune last conv later set
        for p in self.resnet.layer4.parameters():
            p.requires_grad = True



        #probably want linear, relu, dropout
        self.linear = nn.Linear(7*7*512, 512)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        init.xavier_normal_(self.linear.weight)

        #self.conv1 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        #self.conv2 = nn.Conv2d(265, 256, 3, stride=2, padding=1)
        #self.conv3 = nn.Conv2d(256, 256, 3, stride=2, padding=1)

        #utils.init_weight(self.conv1)
        #utils.init_weight(self.conv2)
        #utils.init_weight(self.conv3)

        self.linear1 = nn.Linear(14*14, 512)
        self.dropout2d1 = nn.Dropout2d(.5)
        self.dropout1 = nn.Dropout(.5)
        self.relu1 = nn.LeakyReLU()
        init.xavier_normal(self.linear1.weight)

    def base_size(self): return 512
    def segment_count(self): return 128
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        #x = self.dropout2d(x)

        x_full = self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))

        x_full_segment = self.dropout1(self.relu1(self.linear1(x.view(-1, 14*14))))
        x_full_segment = x_full_segment.view(-1,self.segment_count(),self.base_size())


        return torch.cat((torch.unsqueeze(x_full,1), x_full_segment), 1)

class resnet_modified_small1(nn.Module):
    def __init__(self):
        super(resnet_modified_small1, self).__init__()
        self.resnet = tv.models.resnet34(pretrained=True)
        #probably want linear, relu, dropout
        self.linear = nn.Linear(7*7*512, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        utils.init_weight(self.linear)

    def base_size(self): return 512
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))

class resnet_modified_large(nn.Module):
    def __init__(self):
        super(resnet_modified_large, self).__init__()
        self.resnet = tv.models.resnet101(pretrained=True)
        #probably want linear, relu, dropout
        self.linear = nn.Linear(7*7*2048, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        utils.init_weight(self.linear)

    def base_size(self): return 2048
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        #print x.size()
        return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))

class baseline(nn.Module):

    def __init__(self, encoder, gpu_mode,cnn_type='resnet_34'):
        super(baseline, self).__init__()

        self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.RandomCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.dev_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.encoder = encoder
        self.gpu_mode = gpu_mode

        self.max_role_count = self.encoder.get_max_role_count()
        self.num_verbs = self.encoder.get_num_verbs()
        self.num_roles = self.encoder.get_num_roles()
        self.vocab_size = self.encoder.get_num_labels() #todo:how to decide this? original has 2000 only
        self.embedding_size = 1024 #user argument
        self.num_graph_steps = 3

        #get the vision module
        '''if cnn_type == 'vgg16' :
            self.cnn = vgg_modified()
        elif cnn_type == 'rn_conv':
            self.cnn = ConvInputModel()'''
        if cnn_type == 'resnet_34':
            self.cnn = resnet_modified_small1()
        elif cnn_type == "resnet_101" :
            self.cnn = resnet_modified_large()
        else:
            print('unknown base network')
            exit()
        self.img_size = self.cnn.base_size()

        #self.graph = action_graph(self.cnn.segment_count(), self.num_graph_steps, self.gpu_mode)

        self.verb_module = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.num_verbs),
            #nn.ReLU()
        )

        self.verb_module.apply(utils.init_weight)

        self.role_lookup_table = nn.Embedding(self.num_roles + 1, self.embedding_size, padding_idx=self.num_roles)
        utils.init_weight(self.role_lookup_table, pad_idx=self.num_roles)

        '''self.lstm = nn.LSTM(self.embedding_size, self.embedding_size, num_layers=2, bidirectional=True)
        utils.init_lstm(self.lstm)'''

        self.role_module = nn.ModuleList([ nn.Linear(self.embedding_size, len(self.encoder.role2_label[role_cat])) for role_cat in self.encoder.role_cat])

        '''self.role_module = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.vocab_size),
            #nn.ReLU()
        )'''
        #self.hidden = self.init_hidden()

        self.role_module.apply(utils.init_weight)

    def train_preprocess(self): return self.train_transform
    def dev_preprocess(self): return self.dev_transform

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
        tensor = torch.zeros(4, 6, self.embedding_size)
        tensor1 = torch.zeros(4, 6, self.embedding_size)
        if self.gpu_mode >= 0:
            tensor = tensor.to(torch.device('cuda'))
            tensor1 = tensor1.to(torch.device('cuda'))
        return (torch.autograd.Variable(tensor),
                torch.autograd.Variable(tensor1))

    def forward(self, img, verbs, roles, hidden=None):
        #print('input size', im_data.size())
        batch_size = img.size(0)

        img_embedding_batch = self.cnn(img)

        verb_predict = self.verb_module(img_embedding_batch)


        pred_list = []

        for i,module in enumerate(self.role_module):
            if i == 0:
                pred = module(img_embedding_batch)
                pred_list.append(pred)
                role_label_predict = pred
            else:
                pred = module(img_embedding_batch)
                pred_list.append(pred)
                role_label_predict = torch.cat((role_label_predict.clone(), pred), 1)


        role_label_pred = []
        role_label_marginal = []
        role_label_max = []
        role_label_maxi = []

        for element in pred_list:
            _rl_maxi, _rl_max ,_rl_marginal = self.log_sum_exp(element)
            role_label_maxi.append(_rl_maxi)
            role_label_max.append(_rl_max)
            role_label_pred.append(element)
            role_label_marginal.append(_rl_marginal)

        role_label_marginal = torch.stack(role_label_marginal, 1)
        role_label_max = torch.stack(role_label_max,1)
        role_label_maxi = torch.stack(role_label_maxi,1)

        label_marginal_sum = torch.sum(role_label_marginal, 1)
        label_marginal_sum = torch.unsqueeze(label_marginal_sum,1)
        label_marginal_sum_expanded = label_marginal_sum.expand_as(verb_predict)

        #v_marginal = label_marginal_sum_expanded + verb_predict

        _, _ , norm  = self.log_sum_exp(verb_predict)

        #print('out from forward :', role_label_predict.size())

        return verb_predict, role_label_predict, norm, verb_predict, role_label_maxi

        #computes log( (1 - exp(x)) * (1 - exp(y)) ) =  1 - exp(y) - exp(x) + exp(y)*exp(x) = 1 - exp(V), so V=  log(exp(y) + exp(x) - exp(x)*exp(y))
    #returns the the log of V
    def logsumexp_nx_ny_xy(self, x, y):
        #_,_, v = self.log_sum_exp(torch.cat([x, y, torch.log(torch.exp(x+y))]).view(1,3))
        if x.item() > y.item():
            return torch.log(torch.exp(y-x) + 1 - torch.exp(y) + 1e-8) + x
        else:
            return torch.log(torch.exp(x-y) + 1 - torch.exp(x) + 1e-8) + y

    #expects a list of vectors, BxD
    #returns the max index of every vector, max value of each vector and the log_sum_exp of the vector
    def log_sum_exp(self,vec):
        max_score, max_i = torch.max(vec,1)
        max_score_broadcast = max_score.view(-1,1).expand(vec.size())
        return (max_i , max_score,  max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast),1)))

    def calculate_loss(self, verb_pred, gt_verbs, norm, role_label_pred, gt_labels):
        batch_size = verb_pred.size()[0]
        start_idx = self.encoder.role_start_idx
        end_idx = self.encoder.role_end_idx
        for i in range(0,batch_size):
            _norm = norm[i]
            _v = verb_pred[i]
            _ref = gt_labels[i]
            for r in range(0,_ref.size(0)):
                v = gt_verbs[i]
                likelihood = _v[v]

                for j in range(0, self.encoder.get_max_role_count()):
                    if j == 0:
                        likelihood += utils.likelihood(role_label_pred[i][start_idx[j]:end_idx[j]], gt_labels[i,r,j], len(self.encoder.role2_label['agent']))
                    elif j == 1:
                        likelihood += utils.likelihood(role_label_pred[i][start_idx[j]:end_idx[j]], gt_labels[i,r,j], len(self.encoder.role2_label['place']))
                    elif j == 2:
                        likelihood += utils.likelihood(role_label_pred[i][start_idx[j]:end_idx[j]], gt_labels[i,r,j], len(self.encoder.role2_label['tool']))
                    elif j == 3:
                        likelihood += utils.likelihood(role_label_pred[i][start_idx[j]:end_idx[j]], gt_labels[i,r,j], len(self.encoder.role2_label['item']))
                    else:
                        likelihood += utils.likelihood(role_label_pred[i][start_idx[j]:end_idx[j]], gt_labels[i,r,j], len(self.encoder.role2_label['other']))

                if likelihood.item() > _norm.item():
                    print ("inference error")
                    print (likelihood)
                    print (_norm)
                if r == 0: _tot = likelihood-_norm
                else :
                    _tot = self.logsumexp_nx_ny_xy(_tot, likelihood-_norm)
                    #print('tot', _tot)
            if i == 0: loss = _tot
            else: loss = loss + _tot
        return -loss/batch_size




