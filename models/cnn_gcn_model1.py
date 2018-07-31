import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
from . import utils
import math
import torch.nn.init as init
import random
from .pygcn import gcn
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

class parallel_table(nn.Module):
    def __init__(self, embedding_size, num_verbs, num_roles):
        super(parallel_table,self).__init__()
        self.verb_lookup_table = nn.Embedding(num_verbs, embedding_size)
        #org code has size num_role + 1 x embedding
        #how to use embeddings here? what is the gain?
        self.role_lookup_table = nn.Embedding(num_roles+1, embedding_size)

        self.verb_lookup_table.weight.clone().fill_(0)
        self.role_lookup_table.weight.clone().fill_(0)


    def forward(self,x):
        #todo: what is the proper way to make batchx1024 -> batchx6x1024
        image_embed = x[0]
        verb_embed = self.verb_lookup_table(x[1])
        role_embed = self.role_lookup_table(x[2])
        role_embed_reshaped = role_embed.transpose(0,1)
        max_role_count = x[2].size()[1]
        image_embed_expand = image_embed.expand(max_role_count, image_embed.size(0), image_embed.size(1))
        verb_embed_expand = verb_embed.expand(max_role_count, verb_embed.size(0), verb_embed.size(1))
        '''final_role_init = torch.empty(role_embed.size(), requires_grad=False)
        for i in range(max_role_count):
            final_role_init[:,i, :] = image_embed * verb_embed * role_embed[:,i, :]
        out3 = self.role_lookup_table(x[2])
        out_size = out3.size()[1]
        out1 = torch.unsqueeze(x[0].repeat(1,out_size),1)
        out1 = out1.view(out3.size())
        out2 = torch.unsqueeze(self.verb_lookup_table(x[1]).repeat(1,out_size),1)
        out2 = out2.view(out3.size())
        y = [out1, out2,out3 ]
        #print('parallel size',final_role_init.size())
        #print('parallel ',final_role_init)'''
        final_role_init = image_embed_expand * verb_embed_expand * role_embed_reshaped
        return final_role_init.transpose(0,1)


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

        self.parallel = parallel_table(self.embedding_size, self.num_verbs, self.num_roles)
        self.role_graph_init_module = nn.Sequential(
            self.parallel,
            nn.ReLU()
        )
        #nhid and dropout, user arg
        #in GCN, they don't define #of nodes in init. they pass an adj matrix in forward.
        self.role_graph = gcn.GCN(
            nfeat=self.embedding_size,
            nhid=1024,
            nclass=1024,
            dropout=0.5
        )

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



    def forward(self, img, verbs, roles, hidden=None):
        #print('input size', im_data.size())
        batch_size = img.size(0)

        img_embedding_batch = self.cnn(img)

        verb_predict = self.verb_module(img_embedding_batch)

        role_init_embedding = self.role_graph_init_module([img_embedding_batch, verbs, roles])
        #print('role init: ', role_init_embedding.size())

        #graph forward
        #adjacency matrix for fully connected undirected graph
        #set only available roles to 1. every verb doesn't have 6 roles.

        adj_matrx = self.encoder.get_adj_matrix(verbs)
        if self.gpu_mode >= 0:
            adj_matrx = torch.autograd.Variable(adj_matrx.cuda())
        else:
            adj_matrx = torch.autograd.Variable(adj_matrx)
        role_predict = self.role_graph(role_init_embedding, adj_matrx)



        for i,module in enumerate(self.role_module):
            if i == 0:
                role_label_predict = module(role_predict[:,i])
            else:
                role_label_predict = torch.cat((role_label_predict.clone(), module(role_predict[:,i])), 1)

        #print('out from forward :', role_label_predict.size())

        return verb_predict, role_label_predict


    def calculate_loss(self, verb_pred, gt_verbs, role_label_pred, gt_labels):
        verb_criterion = nn.CrossEntropyLoss()
        target = gt_verbs
        verb_loss = verb_criterion(verb_pred, target)
        batch_size = verb_pred.size()[0]
        loss = 0
        start_idx = self.encoder.role_start_idx
        end_idx = self.encoder.role_end_idx
        for i in range(batch_size):
            for index in range(gt_labels.size()[1]):
                frame_loss = 0
                for j in range(0, self.encoder.get_max_role_count()):
                    if j == 0:
                        frame_loss += utils.cross_entropy_loss(role_label_pred[i][start_idx[j]:end_idx[j]], gt_labels[i,index,j], len(self.encoder.role2_label['agent']))
                    elif j == 1:
                        frame_loss += utils.cross_entropy_loss(role_label_pred[i][start_idx[j]:end_idx[j]], gt_labels[i,index,j], len(self.encoder.role2_label['place']))
                    elif j == 2:
                        frame_loss += utils.cross_entropy_loss(role_label_pred[i][start_idx[j]:end_idx[j]], gt_labels[i,index,j], len(self.encoder.role2_label['tool']))
                    elif j == 3:
                        frame_loss += utils.cross_entropy_loss(role_label_pred[i][start_idx[j]:end_idx[j]], gt_labels[i,index,j], len(self.encoder.role2_label['item']))
                    else:
                        frame_loss += utils.cross_entropy_loss(role_label_pred[i][start_idx[j]:end_idx[j]], gt_labels[i,index,j], len(self.encoder.role2_label['other']))

                loss += frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])


        final_loss = verb_loss + loss/batch_size
        #print('loss :', final_loss)
        return final_loss




