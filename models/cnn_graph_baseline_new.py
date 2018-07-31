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
        self.linear = nn.Linear(7*7*512, 512)
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
        #img_embedding_adjusted = self.img_embedding_layer(img_embedding)
        #print('cnn out size', img_embedding_batch.size())

        #initialize verb node with summation of all region feature vectors
        '''verb_init = img_embedding_batch[:,0]
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

        #original code use gold verbs to insert to role predict module (only at training )
        #print('roles', roles)'''
        '''for i in range(roles.size(0)):
            for j in range(0,6):
                embd = self.role_lookup_table(roles[i][j].type(torch.LongTensor))
                print('role embd' , embd)
                break
            break'''
        '''roles = roles.type(torch.LongTensor)

        if self.gpu_mode >= 0:
            roles = roles.to(torch.device('cuda'))

        role_embedding = self.role_lookup_table(roles)'''
        #mask = self.encoder.
        #print('role embedding', role_embedding[0][3])

        '''vert_no_verb = vert_states[:,1:]
        verb_expand = vert_states[:,0].expand(self.max_role_count, vert_states.size(0),vert_states.size(-1))
        verb_expand = verb_expand.transpose(1,0)
        role_verb = torch.mul(role_embedding, verb_expand)
        role_mul = torch.matmul(role_embedding, vert_no_verb.transpose(-2, -1))#torch.mul(role_embedding, vert_state_expanded)
        #print('cat :', role_mul[0,-1])
        role_mul = role_mul.masked_fill(role_mul == 0, -1e9)

        p_attn = F.softmax(role_mul, dim = -1)
        mask = self.encoder.apply_mask(roles, p_attn)
        p_attn = mask * p_attn



        att_weighted_role = torch.matmul(p_attn, vert_no_verb)
        #print('attention :', att_weighted_role.size(), att_weighted_role)
        #print('check', att_weighted_role[:,-1])
        combined_role_val = att_weighted_role[:,0] * att_weighted_role[:,1] * att_weighted_role[:,2] * att_weighted_role[:,3] *att_weighted_role[:,4] *att_weighted_role[:,5]
        #verb_expanded = torch.mul(vert_states[:,0], torch.sum(att_weighted_role,1))
        verb_predict = self.verb_module(vert_states[:,0])
        #verb_predict = self.verb_module(vert_states[:,0])'''
        '''hidden = self.init_hidden()

        lstm_out, hidden = self.lstm(att_weighted_role, hidden)

        role_label_predict = self.role_module(lstm_out)'''
        verb_predict = self.verb_module(img_embedding_batch)
        for i,module in enumerate(self.role_module):
            if i == 0:
                role_label_predict = module(img_embedding_batch)
            else:
                role_label_predict = torch.cat((role_label_predict.clone(), module(img_embedding_batch)), 1)


        #print('out from forward :', role_label_predict.size())

        return verb_predict, role_label_predict

    '''def forward(self, img, verbs, roles):
        #print('input size', im_data.size())

        img_embedding_batch = self.cnn(img)
        #img_embedding_adjusted = self.img_embedding_layer(img_embedding)
        #print('cnn out size', img_embedding_batch.size())

        #initialize verb node with summation of all region feature vectors
        verb_init = img_embedding_batch[:,0]

        roles = roles.type(torch.LongTensor)

        if self.gpu_mode >= 0:
            roles = roles.to(torch.device('cuda'))

        role_embedding = self.role_lookup_table(roles)
        verb_expand = verb_init.expand(self.max_role_count, verb_init.size(0),verb_init.size(-1))
        verb_expand = verb_expand.transpose(1,0)
        role_verb = torch.mul(role_embedding, verb_expand)

        verb_predict = self.verb_module(verb_init)
        #verb_predict = self.verb_module(vert_states[:,0])

        role_label_predict = self.role_module(role_verb)

        #print('out from forward :', verb_predict.size(), role_label_predict.size())

        return verb_predict, role_label_predict'''

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

        #verb_criterion = nn.CrossEntropyLoss()
        target = gt_verbs
        #print('verb pred vs gt', pred_best, target)
        #verb_loss = verb_criterion(verb_pred, target)
        #this is a multi label classification problem
        batch_size = verb_pred.size()[0]

        '''agent_criterion = nn.CrossEntropyLoss(ignore_index=len(self.encoder.role2_label['agent']), size_average=False)
        place_criterion = nn.CrossEntropyLoss(ignore_index=len(self.encoder.role2_label['place']), size_average=False)
        tool_criterion = nn.CrossEntropyLoss(ignore_index=len(self.encoder.role2_label['tool']), size_average=False)
        item_criterion = nn.CrossEntropyLoss(ignore_index=len(self.encoder.role2_label['item']), size_average=False)
        other_criterion = nn.CrossEntropyLoss(ignore_index=len(self.encoder.role2_label['other']), size_average=False)'''

        loss = 0
        start_idx = self.encoder.role_start_idx
        end_idx = self.encoder.role_end_idx
        for i in range(batch_size):
            for index in range(gt_labels.size()[1]):
                frame_loss = 0
                verb_loss = utils.cross_entropy_loss(verb_pred[i],gt_verbs[i], len(self.encoder.role2_label['agent']))
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

                loss += verb_loss + frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss




