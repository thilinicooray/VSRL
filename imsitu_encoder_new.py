import torch
import random
from collections import OrderedDict
import csv

#This is the class which encodes training set json in the following structure
#todo: the structure

class imsitu_encoder():
    def __init__(self, train_set):
        # json structure -> {<img_id>:{frames:[{<role1>:<label1>, ...},{}...], verb:<verb1>}}
        print('imsitu encoder initialization started.')
        self.verb_list = []
        self.role_list = []
        self.max_label_count = 3
        self.verb2_role_dict = {}
        self.label_list = ['#UNK#']
        label_frequency = {}
        self.max_role_count = 0
        self.role2_label = {}
        self.main_roles = ['agent', 'place', 'tool', 'item']
        self.role_cat = ['agent', 'place', 'tool', 'item', 'other', 'other', 'other']

        for img_id in train_set:
            img = train_set[img_id]
            current_verb = img['verb']
            if current_verb not in self.verb_list:
                self.verb_list.append(current_verb)
                self.verb2_role_dict[current_verb] = []

            for frame in img['frames']:
                for role,label in frame.items():
                    if role not in self.role_list:
                        self.role_list.append(role)
                    if role not in self.verb2_role_dict[current_verb]:
                        self.verb2_role_dict[current_verb].append(role)
                    if role not in self.role2_label:
                        self.role2_label[role] = []
                    if label not in self.role2_label[role]:
                        self.role2_label[role].append(label)
                    '''if role not in self.role2_verb:
                        self.role2_verb[role] = []
                    if current_verb not in self.role2_verb[role]:
                        self.role2_verb[role].append(current_verb)'''
                    if label not in self.label_list:
                        if label not in label_frequency:
                            label_frequency[label] = 1
                        else:
                            label_frequency[label] += 1
                        #only labels occur at least 5 times are considered
                        if label_frequency[label] == 20:
                            self.label_list.append(label)

        for (k,v) in self.verb2_role_dict.items():
            i = 0
            for role in v:
                if role not in self.main_roles:
                    i += 1
            if i > self.max_role_count:
                self.max_role_count = i

        self.max_role_count += 4

        updated_role2_label = {}
        for (k,v) in self.role2_label.items():
            new_list = ['#UNK#']
            for label in v:
                if label in self.label_list and label not in new_list:
                    new_list.append(label)
            updated_role2_label[k] = new_list
            #print('label count', k, len(v))

        self.role2_label = updated_role2_label
        updated_role2_label = {}
        for (k,v) in self.role2_label.items():
            if k in self.main_roles:
                updated_role2_label[k] = v
            else:
                if 'other' not in updated_role2_label:
                    updated_role2_label['other'] = v
                else:
                    updated_role2_label['other'].extend(x for x in v if x not in updated_role2_label['other'])

        self.role2_label = updated_role2_label

        self.role_start_idx = []
        self.role_end_idx = []

        for p in range(0, self.max_role_count):
            if p == 0:
                self.role_start_idx.append(0)
                self.role_end_idx.append(len(self.role2_label['agent']))
            elif p == 1:
                self.role_start_idx.append(self.role_end_idx[-1])
                self.role_end_idx.append(self.role_end_idx[-1] + len(self.role2_label['place']))
            elif p == 2:
                self.role_start_idx.append(self.role_end_idx[-1])
                self.role_end_idx.append(self.role_end_idx[-1] + len(self.role2_label['tool']))
            elif p == 3:
                self.role_start_idx.append(self.role_end_idx[-1])
                self.role_end_idx.append(self.role_end_idx[-1] + len(self.role2_label['item']))
            else:
                self.role_start_idx.append(self.role_end_idx[-1])
                self.role_end_idx.append(self.role_end_idx[-1] + len(self.role2_label['other']))

        self.verb2role_encoding = self.get_verb2role_encoding()

        print('train set stats: \n\t verb count:', len(self.verb_list), '\n\t role count:',len(self.role_list),
              '\n\t label count:', len(self.label_list) ,
              '\n\t max role count:', self.max_role_count)


    def save_stat(self, dict, filename):
        newdict = {}
        for y in dict.items():
            newdict[y[0]] = len(y[1])

        d_sorted_by_value = OrderedDict(sorted(newdict.items(), key=lambda x: x[1], reverse=True))

        import pandas as pd

        (pd.DataFrame.from_dict(data=d_sorted_by_value, orient='index')
         .to_csv(filename+'.csv', header=False))

    def get_verb2role_encoding(self):
        verb2role_embedding_list = []

        for verb_id in range(len(self.verb_list)):
            current_role_list = self.verb2_role_dict[self.verb_list[verb_id]]

            role_embedding_verb = []
            for element in self.main_roles:
                if element in current_role_list:
                    role_embedding_verb.append(1)
                else:
                    role_embedding_verb.append(0)

            for role in current_role_list:
                if not role in self.main_roles:
                    role_embedding_verb.append(1)


            padding_count = self.max_role_count - len(role_embedding_verb)

            for i in range(padding_count):
                role_embedding_verb.append(0)

            verb2role_embedding_list.append(torch.tensor(role_embedding_verb))

        return verb2role_embedding_list

    def encode(self, item):
        verb = self.verb_list.index(item['verb'])
        roles = self.get_role_ids(item['verb'])
        labels = self.get_label_ids(item['frames'])

        #print('item encoding size : v r l', verb.size(), roles.size(), labels.size())
        #assuming labels are also in order of roles in encoder
        return verb, roles, labels

    def save_encoder(self):
        return None

    def load_encoder(self):
        return None

    def get_max_role_count(self):
        return self.max_role_count

    def get_num_verbs(self):
        return len(self.verb_list)

    def get_num_roles(self):
        return len(self.role_list)

    def get_num_labels(self):
        return len(self.label_list)

    def get_verb_encoding(self, verb_id):
        verbs = []

        for id in verb_id:
            verb = self.verb_embedding[id]
            verbs.append(verb)

        return torch.stack(verbs)

    def get_role_encoding(self, verb_id):
        verb_roles = []

        for id in verb_id:
            verb_r = self.verb2role_embedding[id]
            verb_roles.append(verb_r)

        return torch.stack(verb_roles)

    def get_role_count(self, verb_id):
        return len(self.verb2_role_dict[self.verb_list[verb_id]])

    def get_adj_matrix(self, verb_ids):
        adj_matrix_list = []

        for id in verb_ids:
            encoding = self.verb2role_encoding[id]
            encoding_tensor = torch.unsqueeze(torch.tensor(encoding),0)
            expanded = encoding_tensor.expand(self.max_role_count, encoding_tensor.size(1))
            transpose = torch.t(expanded)
            adj = expanded*transpose
            adj_matrix_list.append(adj)

        return torch.stack(adj_matrix_list).type(torch.FloatTensor)

    def apply_mask(self, roles, val):
        mask = val.clone().fill_(1)
        batch_size = roles.size(0)
        for i in range(batch_size):
            for j in range(self.get_max_role_count()):
                role = roles[i][j]
                if role == 190:
                    #embedding[i][j] = embedding[i][j].fill_(0)
                    mask[i,j:] = 0
                    break

        return mask

    def get_role_ids(self, verb):
        #print('verb', verb)
        roles = self.verb2_role_dict[verb]
        #print('roles', len(roles))
        #add main roles for all
        role_id = []
        main_role_ids = []
        for element in self.main_roles:
            main_role_ids.append(self.role_list.index(element))
        role_id = main_role_ids
        #print('role is list', len(main_role_ids))

        for role in roles:
            if self.role_list.index(role) not in role_id:
                role_id.append(self.role_list.index(role))

        pad = self.max_role_count - len(role_id)
        for j in range(pad):
            role_id.append(190)

        return torch.tensor(role_id)

    def get_label_ids(self, frames):

        all_frame_id_list = []
        for frame in frames:
            label_id_list = []

            for main_role in self.main_roles:
                found = False
                for role,label in frame.items():
                    if role == main_role:
                        #use UNK when unseen labels come
                        found = True
                        if label in self.role2_label[main_role]:
                            label_id = self.role2_label[main_role].index(label)
                        else:
                            label_id = self.role2_label[main_role].index('#UNK#')

                        label_id_list.append(label_id)
                if not found:
                    label_id_list.append(len(self.role2_label[main_role]))

            for role,label in frame.items():
                if role not in self.main_roles:
                    if label in self.role2_label['other']:
                        label_id = self.role2_label['other'].index(label)
                    else:
                        label_id = self.role2_label['other'].index('#UNK#')

                    label_id_list.append(label_id)

            role_padding_count = self.max_role_count - len(label_id_list)

            for i in range(role_padding_count):
                label_id_list.append(len(self.role2_label['other']))

            all_frame_id_list.append(torch.tensor(label_id_list))

        labels = torch.stack(all_frame_id_list,0)

        return labels
