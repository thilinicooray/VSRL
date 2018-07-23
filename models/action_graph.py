'''SRL graph has one verb node and #number of regions region nodes. each region node connects with verb
from an edge. region nodes are not connected with each other directly
'''

import torch.nn as nn
import torch
import utils

class action_graph(nn.Module):
    def __init__(self, num_regions, num_steps):
        self.num_regions = num_regions
        self.num_steps = num_steps
        self.vert_state_dim = 512
        self.edge_state_dim = 512

        self.vert_gru = nn.GRUCell(self.vert_state_dim, self.vert_state_dim)
        self.edge_gru = nn.GRUCell(self.edge_state_dim, self.edge_state_dim)
        #todo: check gru param init. code resets, but not sure

        self.edge_att = nn.Sequential(
            nn.Linear(self.edge_state_dim * 2, 1),
            nn.Sigmoid(),
            nn.LogSoftmax()

        )

        self.vert_att = nn.Sequential(
            nn.Linear(self.vert_state_dim * 2, 1),
            nn.Sigmoid(),
            nn.LogSoftmax()
        )

        self.edge_att.apply(utils.init_weight)#actually pytorch init does reset param
        self.vert_att.apply(utils.init_weight)


    def forward(self, input):
        vert_state = self.vert_gru(input[0])
        edge_state = self.edge_gru(input[1])

        #todo: check whether this way is correct, TF code uses a separate global var to keep hidden state
        for i in range(self.num_steps):
            edge_context = self.get_edge_context(edge_state, vert_state)
            vert_context = self.get_vert_context(vert_state, edge_state)

            edge_state = self.edge_gru(edge_context, edge_state)
            vert_state = self.vert_gru(vert_context, vert_state)


        return vert_state, edge_state

    def get_edge_context(self, edge_state, vert_state):
        #todo: implement for undirectional, not only have verb-> region direction.
        # however i dont use 2 independent linear layers like them

        '''
        here we do not consider the direction as ours is undirectional.
        :param edge_state: 200x512
        :param vert_state: 201 x 512
        :return:
        '''

        verb_vert_state = vert_state[0]
        region_vert_state = vert_state[1:]
        verb_expanded_state = verb_vert_state.expand(region_vert_state.size(0), verb_vert_state.size(0))

        print('vert shapes', verb_vert_state.size(), region_vert_state.size(), verb_expanded_state.size())

        verb_concat = torch.cat((verb_expanded_state, edge_state), 1)
        region_concat = torch.cat((region_vert_state, edge_state), 1)

        att_weighted_verb = torch.mul(self.edge_att(verb_concat), verb_expanded_state)
        att_weighted_region = torch.mul(self.edge_att(region_concat), region_vert_state)

        return att_weighted_verb + att_weighted_region

    def get_vert_context(self, vert_state, edge_state):
        verb_vert_state = vert_state[0]
        region_vert_state = vert_state[1:]
        verb_expanded_state = verb_vert_state.expand(region_vert_state.size(0), verb_vert_state.size(0))

        print('vert shapes', verb_vert_state.size(), region_vert_state.size(), verb_expanded_state.size())

        verb_concat = torch.cat((verb_expanded_state, edge_state), 1)
        region_concat = torch.cat((region_vert_state, edge_state), 1)

        att_weighted_verb_per_edge = torch.mul(self.vert_att(verb_concat), edge_state)
        att_weighted_region = torch.mul(self.edge_att(region_concat), edge_state)
        att_weighted_verb = torch.sum(att_weighted_verb_per_edge, 0)

        vert_ctx = torch.cat((att_weighted_verb,att_weighted_region), 0)

        print('vert context :', vert_ctx.size())
        return vert_ctx


    '''
    in graph_baseline
    init
    cnn
    linear layer for 1024->512 (or update faster rcnn itself
    action_graph
    verb_layer
    attention layer
    role_layer
    
    forward
    get region x features from faster rcnn
    get vert_date, edge_state from action graph
    get verb softmax from verb_layer
    get role based attention for 6 roles
    get label_rep using attention
    get label softmax using role layer
    
    '''