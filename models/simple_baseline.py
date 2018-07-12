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

        self.verb_module = nn.Sequential(
            nn.Linear(self.embedding_size, self.num_verbs)
        )

        #self.verb_module = nn.Linear(self.img_size, self.num_verbs)
        self.verb_module.apply(utils.init_weight)



        self.role_module =  nn.ModuleList(
            [ nn.Linear(self.embedding_size, self.vocab_size) for i in range(self.num_roles)]
        )
        self.role_module.apply(utils.init_weight)



    def forward(self, images, verbs, roles):
        img_embedding = self.cnn(images)
        verb_predict = self.verb_module(img_embedding)

        role_list = []
        for i, l in enumerate(self.role_module):
            role_list.append(l(img_embedding))

        role_predict = torch.stack(role_list).transpose(0,1)

        return verb_predict, role_predict

    def calculate_loss(self, verb_pred, gt_verbs, roles_pred, gt_labels):

        criterion = nn.CrossEntropyLoss()

        target = torch.max(gt_verbs,1)[1]
        verb_loss = criterion(verb_pred, target)

        #get correct roles for each gt verb from roles pred
        target_role_encoding = self.encoder.get_role_encoding(target)

        role_pred_for_target = torch.bmm(target_role_encoding, roles_pred)

        batch_size = gt_verbs.size(0)

        loss = 0
        for i in range(batch_size):
            sub_loss = 0
            for index in range(gt_labels.size()[1]):
                sub_loss += criterion(role_pred_for_target[i], torch.max(gt_labels[i,index,:,:],1)[1])
            loss += sub_loss

        final_loss = verb_loss + loss/batch_size

        return final_loss

