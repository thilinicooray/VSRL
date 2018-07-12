import torch
import torch.nn as nn
import torchvision as tv
from . import utils, pygcn
from .pygcn import gcn


class resnet152_pretrained(nn.Module):
    def __init__(self, embed_size=1024):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(resnet152_pretrained, self).__init__()
        resnet = tv.models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

    def rep_size(self): return 1024

class vgg_modified(nn.Module):
    def __init__(self):
        super(vgg_modified,self).__init__()
        self.vgg = tv.models.vgg16(pretrained=True)
        self.vgg_features = self.vgg.features
        #self.classifier = nn.Sequential(
        #nn.Dropout(),
        self.lin1 = nn.Linear(512 * 7 * 7, 1024)
        self.relu1 = nn.ReLU(True)
        self.dropout1 = nn.Dropout()
        self.lin2 =  nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU(True)
        self.dropout2 = nn.Dropout()

        utils.init_weight(self.lin1)
        utils.init_weight(self.lin2)

    def rep_size(self): return 1024

    def forward(self,x):
        return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features(x)))))))

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

class resnet_modified_medium(nn.Module):
    def __init__(self):
        super(resnet_modified_medium, self).__init__()
        self.resnet = tv.models.resnet50(pretrained=True)
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


class resnet_modified_small(nn.Module):
    def __init__(self):
        super(resnet_modified_small, self).__init__()
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
    def __init__(self, encoder, gpu_mode,cnn_type='resnet_34'):
        super(baseline, self).__init__()
        self.encoder = encoder
        self.gpu_mode = gpu_mode

        #get the CNN
        if cnn_type == 'resnet152' : self.cnn = resnet152_pretrained()
        elif cnn_type == 'resnet_101' : self.cnn = resnet_modified_large()
        elif cnn_type == 'resnet_50': self.cnn = resnet_modified_medium()
        elif cnn_type == 'resnet_34': self.cnn = resnet_modified_small()
        else:
            print('unknown base network')
            exit()
        self.img_size = self.cnn.rep_size()

        self.max_node_count = self.encoder.get_max_role_count()
        self.num_verbs = self.encoder.get_num_verbs()
        self.num_roles = self.encoder.get_num_roles()
        self.vocab_size = self.encoder.get_num_labels() #todo:how to decide this? original has 2000 only
        self.embedding_size = 1024 #user argument

        self.img_embedding_layer = nn.Linear(self.img_size, self.embedding_size)
        utils.init_weight(self.img_embedding_layer)

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


    def forward(self, images, verbs, roles):
        #print('input size', images.size())
        img_embedding = self.cnn(images)
        img_embedding_adjusted = self.img_embedding_layer(img_embedding)
        #print('cnn out size', img_embedding.size())
        verb_predict = self.verb_module(img_embedding_adjusted)
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
        role_init_embedding = self.role_graph_init_module([img_embedding_adjusted, verbs, roles])
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

