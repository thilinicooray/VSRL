import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
from . import utils
import math

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
        utils.init_weight(self.linear)

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
        utils.init_weight(self.linear1)

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

class baseline(nn.Module):
    def __init__(self, encoder, gpu_mode,cnn_type='resnet_34'):
        super(baseline, self).__init__()
        self.encoder = encoder
        self.gpu_mode = gpu_mode

        self.max_role_count = self.encoder.get_max_role_count()
        self.num_verbs = self.encoder.get_num_verbs()
        self.num_roles = self.encoder.get_num_roles()
        self.vocab_size = self.encoder.get_num_labels() #todo:how to decide this? original has 2000 only
        self.embedding_size = 512 #user argument
        self.num_graph_steps = 3

        #get the vision module
        '''if cnn_type == 'vgg16' :
            self.cnn = vgg_modified()
        elif cnn_type == 'rn_conv':
            self.cnn = ConvInputModel()'''
        if cnn_type == 'resnet_34':
            self.cnn = resnet_modified_small()
        else:
            print('unknown base network')
            exit()
        self.img_size = self.cnn.base_size()

        self.graph = action_graph(self.cnn.segment_count(), self.num_graph_steps, self.gpu_mode)

        self.verb_module = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.num_verbs)
        )

        self.verb_module.apply(utils.init_weight)

        self.role_lookup_table = nn.Linear(self.num_roles, self.embedding_size)

        self.role_att = nn.Sequential(
            nn.Linear(self.embedding_size * 2, 1),
            nn.Tanh(),
            nn.LogSoftmax()
        )

        self.role_module = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.vocab_size)
        )

    def forward(self, img, verbs, roles):
        #print('input size', im_data.size())

        img_embedding_batch = self.cnn(img)
        #img_embedding_adjusted = self.img_embedding_layer(img_embedding)
        #print('cnn out size', img_embedding_batch.size())

        #initialize verb node with summation of all region feature vectors
        verb_init = img_embedding_batch[:,0]
        #print('verb_init', verb_init.size(), torch.unsqueeze(verb_init, 1).size())

        vert_init = img_embedding_batch
        #print('vert_init :', vert_init.size())
        #initialize each edge with verb + respective region feature vector
        verb_init_expand = verb_init.expand(img_embedding_batch.size(1)-1, verb_init.size(0), verb_init.size(1))
        verb_init_expand = verb_init_expand.transpose(0,1)
        edge_init = img_embedding_batch[:,1:] + verb_init_expand

        #print('input to graph :', vert_init.size(), edge_init.size())

        vert_states, edge_states = self.graph((vert_init,edge_init))
        #print('out from graph :', vert_states.size(), edge_states.size())

        verb_predict = self.verb_module(vert_states[:,0])

        #original code use gold verbs to insert to role predict module (only at training )

        role_embedding = self.role_lookup_table(roles)
        #print('role_embedding :', role_embedding.size())

        role_label_embd_list = []

        #for attention, first try with node only
        #todo: use edge for this calculation
        role_expanded_state = role_embedding.expand(edge_states.size(1),role_embedding.size(0), role_embedding.size(1),
                                                    role_embedding.size(2))
        role_expanded_state = role_expanded_state.permute(1,2,0,3)
        vert_state_expanded = vert_states.expand(role_embedding.size(1),vert_states.size(0), vert_states.size(1),
                                                 vert_states.size(2))
        vert_state_expanded = vert_state_expanded.transpose(0,1)
        #print('expand :', role_expanded_state.size(), vert_state_expanded.size())
        role_concat = torch.cat((role_expanded_state, vert_state_expanded[:,:,1:]), 3)
        #print('cat :', role_concat.size())

        att_weighted_role_per_region = torch.mul(self.role_att(role_concat), vert_state_expanded[:,:,1:])
        #print('att :', att_weighted_role_per_region.size())
        att_weighted_role_embd = torch.sum(att_weighted_role_per_region, 2)
        #print('weighted sum :',  att_weighted_role_embd.size())

        '''for role_embd in role_embedding:
            #print('role embed size :', role_embd.size())
            role_expanded_state = role_embd.expand(edge_states.size(0), role_embd.size(0))
            #print('expand :', role_expanded_state.size(), vert_states[1:].size())
            role_concat = torch.cat((role_expanded_state, vert_states[1:]), 1)
            #print('concat :', role_concat.size())
            att_weighted_role_per_region = torch.mul(self.role_att(role_concat), vert_states[1:])
            att_weighted_role = torch.sum(att_weighted_role_per_region, 0)
            role_label_embd_list.append(att_weighted_role)

        label_embed = torch.stack(role_label_embd_list)'''
        role_label_predict = self.role_module(att_weighted_role_embd)

        #print('out from forward :', verb_predict.size(), role_label_predict.size())

        return verb_predict, role_label_predict

    def calculate_loss(self, verb_pred, gt_verbs, role_label_pred, gt_labels):
        '''criterion = nn.CrossEntropyLoss()
        #loss = verb_loss + c.entropy for roles, for all 3 ann per image.
        verb_tensor = torch.unsqueeze(gt_verbs, 0)
        #print('v tensor', verb_tensor.size())
        target = torch.max(verb_tensor, 1)[1]
        #print('v gold', target)
        verb_loss = criterion(torch.unsqueeze(verb_pred, 0), target)
        #this is a multi label classification problem
        loss = 0
        for index in range(gt_labels.size()[0]):
            loss += criterion(role_label_pred, torch.max(gt_labels[index,:,:],1)[1])

        final_loss = verb_loss + loss'''

        criterion = nn.CrossEntropyLoss()


        target = torch.max(gt_verbs,1)[1]
        #print(target)
        verb_loss = criterion(verb_pred, target)
        #this is a multi label classification problem
        batch_size = verb_pred.size()[0]
        loss = 0
        for i in range(batch_size):
            sub_loss = 0
            for index in range(gt_labels.size()[1]):
                actual_ids = utils.get_only_relevant_roles(gt_labels[i,index,:,:])
                if self.gpu_mode >= 0:
                    actual_ids = actual_ids.to(torch.device('cuda'))
                sub_loss += criterion(role_label_pred[i, :len(actual_ids)], actual_ids)
            loss += sub_loss


        final_loss = verb_loss + loss/batch_size
        return final_loss



