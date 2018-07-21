import torch
import torch.nn as nn
import torchvision as tv
from . import utils
from .pygcn import gcn
from .faster_rcnn.faster_rcnn.vgg16_modified import vgg16


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
        #print('model', self.frcnn_vgg)

        '''self.frcnn_vgg_features = self.frcnn_vgg.features
        #self.vgg_features = self.vgg.features
        #self.classifier = nn.Sequential(
        #nn.Dropout(),
        self.lin1 = nn.Linear(512 * 7 * 7, 1024)
        self.relu1 = nn.ReLU(True)
        self.dropout1 = nn.Dropout()
        self.lin2 =  nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU(True)
        self.dropout2 = nn.Dropout()

        utils.init_weight(self.lin1)
        utils.init_weight(self.lin2)'''

    def rep_size(self): return 1024

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        print('frcnn original size:',self.fasterRCNN(im_data, im_info, gt_boxes, num_boxes).size())
        return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features()))))))

class parallel_table(nn.Module):
    def __init__(self, embedding_size, num_verbs, num_roles):
        super(parallel_table,self).__init__()
        self.verb_lookup_table = nn.Linear(num_verbs, embedding_size)
        #org code has size num_role + 1 x embedding
        #how to use embeddings here? what is the gain?
        self.role_lookup_table = nn.Linear(num_roles, embedding_size)


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

class mul_table(nn.Module):
    def __init__(self):
        super(mul_table, self).__init__()

    def forward(self, x1):
        x_mul = x1[0] * x1[1] * x1[2]
        return x_mul

class baseline(nn.Module):
    def __init__(self, encoder, gpu_mode, pretrained_cnn_path, cnn_type='faster_rcnn_vgg'):
        super(baseline, self).__init__()
        self.encoder = encoder
        self.gpu_mode = gpu_mode

        #get the CNN
        if cnn_type == 'faster_rcnn_vgg' :
            self.cnn = frcnn_pretrained_vgg_modified(pretrained_cnn_path)
            self.cnn.eval()
        else:
            print('unknown base network')
            exit()
        self.img_size = self.cnn.rep_size()

        self.max_node_count = self.encoder.get_max_role_count()
        self.num_verbs = self.encoder.get_num_verbs()
        self.num_roles = self.encoder.get_num_roles()
        self.vocab_size = self.encoder.get_num_labels() #todo:how to decide this? original has 2000 only
        self.embedding_size = 1024 #user argument

        self.verb_module = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.num_verbs)
        )

        #self.verb_module = nn.Linear(self.img_size, self.num_verbs)
        self.verb_module.apply(utils.init_weight)



        self.parallel = parallel_table(self.embedding_size, self.num_verbs, self.num_roles)
        self.role_graph_init_module = nn.Sequential(
            self.parallel,
            nn.ReLU()
        )
        self.role_graph_init_module.apply(utils.init_weight)

        #nhid and dropout, user arg
        #in GCN, they don't define #of nodes in init. they pass an adj matrix in forward.
        self.role_graph = gcn.GCN(
            nfeat=self.embedding_size,
            nhid=1024,
            nclass=self.vocab_size,
            dropout=0.5
        )


    def forward(self, im_data, im_info, gt_boxes, num_boxes, verbs, roles):
        #print('input size', images.size())

        img_embedding = self.cnn(im_data, im_info, gt_boxes, num_boxes)
        #img_embedding_adjusted = self.img_embedding_layer(img_embedding)
        #print('cnn out size', img_embedding.size())
        verb_predict = self.verb_module(img_embedding)
        #print('verb module out ', verb_predict.size())
        #get argmax(verb is) from verb predict
        #todo: check which is the most correct way
        '''
        original code use gold verbs to insert to role predict module
        _, verb_id = torch.max(verb_predict, 1)
        verbs = self.encoder.get_verb_encoding(verb_id)
        roles = self.encoder.get_role_encoding(verb_id)

        if self.gpu_mode >= 0: 
            #if torch.cuda.is_available():
            verbs = verbs.to(torch.device('cuda'))
            roles = roles.to(torch.device('cuda'))'''
        #expected size = 6 x embedding size
        role_init_embedding = self.role_graph_init_module([img_embedding, verbs, roles])
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
        #print('role predict size :', role_predict.size())

        return verb_predict, role_predict
        #return verb_predict

    def calculate_loss(self, verb_pred, gt_verbs, roles_pred, gt_labels):
        '''

        :param verb_pred: write sizes
        :param gt_verbs:
        :param roles_pred:
        :param gt_labels:
        :return:
        '''
        #as per paper, loss is sum(i) sum(3) (cross_entropy(verb) + 1/6sum(all roles)cross_entropy(label)

        criterion = nn.CrossEntropyLoss()


        target = torch.max(gt_verbs,1)[1]
        verb_loss = criterion(verb_pred, target)
        #this is a multi label classification problem
        batch_size = verb_pred.size()[0]
        loss = 0
        for i in range(batch_size):
            sub_loss = 0
            for index in range(gt_labels.size()[1]):
                sub_loss += criterion(roles_pred[i], torch.max(gt_labels[i,index,:,:],1)[1])
            loss += sub_loss


        final_loss = verb_loss + loss/batch_size


        return final_loss

