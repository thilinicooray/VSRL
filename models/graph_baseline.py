import torch
import torch.nn as nn
import torchvision as tv
from . import utils
from .pygcn import gcn
from .faster_rcnn.faster_rcnn.vgg16_modified import vgg16
from .action_graph import action_graph
from .faster_rcnn.utils.config import cfg


class frcnn_pretrained_vgg_modified(nn.Module):
    def __init__(self, pretrained_model):
        super(frcnn_pretrained_vgg_modified,self).__init__()
        self.fasterRCNN = vgg16(pretrained=False)
        self.fasterRCNN.create_architecture()
        checkpoint = torch.load(pretrained_model)

        #getting only the required keys from original pretrained model
        updated_frcnn_dict = self.fasterRCNN.state_dict()
        filtered_pretrained_dict = {k: v for k, v in checkpoint['model'].items() if k in updated_frcnn_dict}
        updated_frcnn_dict.update(filtered_pretrained_dict)

        self.fasterRCNN.load_state_dict(updated_frcnn_dict)
        self.fasterRCNN.eval()
        #print('model', self.frcnn_vgg)

        #self.frcnn_vgg_features = self.frcnn_vgg.features
        #self.vgg_features = self.vgg.features
        #self.classifier = nn.Sequential(
        #nn.Dropout(),
        self.lin1 = nn.Linear(4096, 1024)
        self.relu1 = nn.ReLU(True)
        #self.dropout1 = nn.Dropout()
        self.lin2 =  nn.Linear(1024, 512)
        self.relu2 = nn.ReLU(True)
        #self.dropout2 = nn.Dropout()

        utils.init_weight(self.lin1)
        utils.init_weight(self.lin2)

    def rep_size(self): return 512

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        #print('frcnn original size:',self.fasterRCNN(im_data, im_info, gt_boxes, num_boxes).size())
        #frcnn_out
        with torch.no_grad():
            batch_size = im_data.size(0)
            batch_region_list = []
            for i in range(batch_size):
                img_embedding = self.fasterRCNN(torch.unsqueeze(im_data[i],0), im_info[i], gt_boxes[i], num_boxes[i])#200x512
                batch_region_list.append(torch.squeeze(img_embedding, 0))
            frcnn_out = torch.stack(batch_region_list,0)

        return self.relu2(self.lin2(self.relu1(self.lin1(frcnn_out))))


class baseline(nn.Module):
    def __init__(self, encoder, gpu_mode, pretrained_cnn_path, cnn_type='faster_rcnn_vgg'):
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
        if cnn_type == 'faster_rcnn_vgg' :
            self.cnn = frcnn_pretrained_vgg_modified(pretrained_cnn_path)
        else:
            print('unknown base network')
            exit()
        self.img_size = self.cnn.rep_size()

        self.graph = action_graph(cfg.TRAIN.RPN_POST_NMS_TOP_N, self.num_graph_steps, self.gpu_mode)

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

    def forward(self, im_data, im_info, gt_boxes, num_boxes, verbs, roles):
        print('input size', im_data.size())

        img_embedding_batch = self.cnn(im_data, im_info, gt_boxes, num_boxes)
        #img_embedding_adjusted = self.img_embedding_layer(img_embedding)
        print('cnn out size', img_embedding_batch.size())

        #initialize verb node with summation of all region feature vectors
        verb_init = torch.sum(img_embedding_batch,1)
        print('verb_init', verb_init.size(), torch.unsqueeze(verb_init, 1).size())

        vert_init = torch.cat((torch.unsqueeze(verb_init, 1),img_embedding_batch),1)
        print('vert_init :', vert_init.size())
        #initialize each edge with verb + respective region feature vector
        verb_init_expand = verb_init.expand(img_embedding_batch.size(1), verb_init.size(0), verb_init.size(1))
        verb_init_expand = verb_init_expand.transpose(0,1)
        edge_init = img_embedding_batch + verb_init_expand

        print('input to graph :', vert_init.size(), edge_init.size())

        vert_states, edge_states = self.graph((vert_init,edge_init))
        print('out from graph :', vert_states.size(), edge_states.size())

        verb_predict = self.verb_module(vert_states[0])

        #original code use gold verbs to insert to role predict module (only at training )

        role_embedding = self.role_lookup_table(roles)
        role_embedding = torch.unsqueeze(role_embedding,3)
        print('role_embedding :', role_embedding.size())

        role_label_embd_list = []

        #for attention, first try with node only
        #todo: use edge for this calculation
        role_expanded_state = role_embedding.expand(role_embedding.size(0), role_embedding.size(1), role_embedding.size(2),
                                                    edge_states.size(1))
        role_expanded_state = role_expanded_state.transpose(2,3)
        vert_state_expanded = vert_states.expand(vert_states.size(0), role_embedding.size(1),role_embedding.size(2),
                                                 vert_states.size(1))
        vert_state_expanded = vert_state_expanded.transpose(2,3)
        print('expand :', role_expanded_state.size(), vert_state_expanded.size())
        role_concat = torch.cat((role_expanded_state, vert_state_expanded[:,:,1:]), 3)
        print('cat :', role_concat.size())

        att_weighted_role_per_region = torch.mul(self.role_att(role_concat), vert_states[:,:,1:])
        att_weighted_role_embd = torch.sum(att_weighted_role_per_region, 3)
        print('out from forward :', att_weighted_role_per_region.size(), att_weighted_role_embd.size())

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

        print('out from forward :', verb_predict.size(), role_label_predict.size())

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
        verb_loss = criterion(verb_pred, target)
        #this is a multi label classification problem
        batch_size = verb_pred.size()[0]
        loss = 0
        for i in range(batch_size):
            sub_loss = 0
            for index in range(gt_labels.size()[1]):
                sub_loss += criterion(role_label_pred[i], torch.max(gt_labels[i,index,:,:],1)[1])
            loss += sub_loss


        final_loss = verb_loss + loss/batch_size
        return final_loss



