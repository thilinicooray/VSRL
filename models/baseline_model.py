import torch.nn as nn
import torchvision as tv
from utils import initLinear

from pygcn import models

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

        initLinear(self.lin1)
        initLinear(self.lin2)

    def rep_size(self): return 1024

    def forward(self,x):
        return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features(x).view(-1, 512*7*7)))))))

class resnet_modified_large(nn.Module):
    def __init__(self):
        super(resnet_modified_large, self).__init__()
        self.resnet = tv.models.resnet101(pretrained=True)
        #probably want linear, relu, dropout
        self.linear = nn.Linear(7*7*2048, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        initLinear(self.linear)

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
        initLinear(self.linear)

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
        initLinear(self.linear)

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
    def __init__(self, mod1, mod2, mod3):
        super(parallel_table,self).__init__()
        self.mod1 = mod1
        self.mod2 = mod2
        self.mod3 = mod3

    def forward(self,x):
        y = [self.mod1(x[0]), self.mod2(x[1]), self.mod3(x[2])]
        return y

class mul_table(nn.Module):
    def __init__(self):
        super(mul_table, self).__init__()

    def forward(self, x1, x2, x3):
        x_mul = x1 * x2 * x3
        return x_mul

class baseline(nn.Module):
    def __init__(self, encoder, cnn_type = 'vgg_net'):

        #get the CNN
        if cnn_type == 'vgg_net' : self.cnn = vgg_modified()
        elif cnn_type == 'resnet_101' : self.cnn = resnet_modified_large()
        elif cnn_type == 'resnet_50': self.cnn = resnet_modified_medium()
        elif cnn_type == 'resnet_34': self.cnn = resnet_modified_small()
        else:
            print('unknown base network')
            exit()
        self.img_size = self.cnn.rep_size()

        self.max_node_count = encoder.get_max_role_count()
        self.num_verbs = encoder.get_num_verbs()
        self.num_roles = encoder.get_num_roles()
        self.vocab_size = 2000 #how to decide this?
        self.embedding_size = 1024 #user argument

        self.verb_embedding_module = nn.Sequential(
                                        nn.Linear(self.img_size, self.embedding_size),
                                        nn.ReLU()
                                    )
        self.img_embedding_module = nn.Sequential(
                                        nn.Linear(self.img_size, self.embedding_size)
                                    )
        #for embeddings, check without nn.sequential as well
        self.verb_lookup_table = nn.Sequential(
                                    nn.Embedding(self.num_verbs, self.embedding_size)
                                )
        self.role_lookup_table = nn.Sequential(
                                    nn.Embedding(self.num_roles + 1, self.embedding_size)
                                )
        self.parallel = parallel_table(self.img_embedding_module, self.verb_lookup_table, self.role_lookup_table)
        self.role_embedding_module = nn.Sequential(
                                        self.parallel,
                                        mul_table(),
                                        nn.ReLU()
                                    )

        self.verb_output = nn.Sequential(
                            nn.Dropout(0.5),
                            nn.Linear(self.embedding_size, self.num_verbs),
                            nn.LogSoftmax()
                        )

        #nhid and dropout, user arg
        #in GCN, they don't define #of nodes in init. they pass an adj matrix in forward.
        self.graph = models.GCN(
                            nfeat=self.embedding_size,
                            nhid=1024,
                            nclass=self.vocab_size,
                            dropout=0.5
                        )
        self.role_output = nn.Sequential(
            self.graph
        )

        '''
        todo: make them all .cuda() to run in GPU mode
        '''

        #init weights ....
        #check model code again. can remove sequential i think
