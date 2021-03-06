'''SRL graph has one verb node and #number of regions region nodes. each region node connects with verb
from an edge. region nodes are not connected with each other directly
'''

import torch.nn as nn
import torch
from . import utils
import torch.nn.functional as F

class action_graph(nn.Module):
    def __init__(self, num_regions, num_steps, gpu_mode):
        super(action_graph,self).__init__()

        self.num_regions = num_regions
        self.num_steps = num_steps
        self.vert_state_dim = 1024
        self.edge_state_dim = 1024
        self.gpu_mode= gpu_mode

        self.vert_gru = nn.GRUCell(self.vert_state_dim, self.vert_state_dim)
        self.edge_gru = nn.GRUCell(self.edge_state_dim, self.edge_state_dim)

        utils.init_gru_cell(self.vert_gru)
        utils.init_gru_cell(self.edge_gru)

        #todo: check gru param init. code resets, but not sure

        self.edge_att = nn.Sequential(
            #nn.Linear(self.edge_state_dim * 2, 1),
            #nn.Tanh(),
            #nn.LogSoftmax()
        )

        self.vert_att = nn.Sequential(
            #nn.Linear(self.vert_state_dim * 2, 1),
            nn.Tanh(),
            nn.LogSoftmax()
        )

        self.out = nn.Linear(self.vert_state_dim, self.vert_state_dim)
        self.m = nn.Tanh()

        '''self.edge_att.apply(utils.init_weight)#actually pytorch init does reset param
        self.vert_att.apply(utils.init_weight)'''


    def forward(self, input_):
        #init hidden state with xavier
        vert_state = torch.zeros(input_[0].size(1), self.vert_state_dim)
        edge_state = torch.zeros(input_[1].size(1), self.edge_state_dim)

        if self.gpu_mode >= 0:
            vert_state = vert_state.to(torch.device('cuda'))
            edge_state = edge_state.to(torch.device('cuda'))

        batch_size = input_[0].size(0)
        vert_input = input_[0]
        edge_input = input_[1]
        #print('vert and edge input', vert_input.size(), edge_input.size())
        vert_state_list = []
        edge_state_list = []
        #todo: can this be parallelized?
        for i in range(batch_size):
            torch.nn.init.xavier_uniform_(vert_state)
            torch.nn.init.xavier_uniform_(edge_state)
            vert_state = self.vert_gru(vert_input[i], vert_state)
            edge_state = self.edge_gru(edge_input[i], edge_state)

            #todo: check whether this way is correct, TF code uses a separate global var to keep hidden state
            for j in range(self.num_steps):
                edge_context = self.get_edge_context(edge_state, vert_state)
                vert_context = self.get_vert_context(vert_state, edge_state)

                edge_state = self.edge_gru(edge_context, edge_state)
                vert_state = self.vert_gru(vert_context, vert_state)

            vert_edge = vert_state[1:].clone() * edge_state.clone()
            full_vert = torch.cat((torch.unsqueeze(vert_state[0],0), vert_edge),0)
            final = torch.unsqueeze(self.m(self.out(full_vert)),0)

            if i == 0:
                vert_state_batch = final
            else:
                vert_state_batch = torch.cat((vert_state_batch.clone(), final), 0)
        return vert_state_batch

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

        #print('vert shapes', verb_vert_state.size(), region_vert_state.size(), verb_expanded_state.size())

        verb_mul = torch.mul(verb_expanded_state, edge_state)
        region_mul = torch.mul(region_vert_state, edge_state)

        att_weighted_verb = torch.mul(self.edge_att(verb_mul), verb_expanded_state)
        att_weighted_region = torch.mul(self.edge_att(region_mul), region_vert_state)

        return att_weighted_verb + att_weighted_region

    def get_vert_context(self, vert_state, edge_state):
        verb_vert_state = vert_state[0]
        region_vert_state = vert_state[1:]
        verb_expanded_state = verb_vert_state.expand(region_vert_state.size(0), verb_vert_state.size(0))

        #print('vert shapes', verb_vert_state.size(), region_vert_state.size(), verb_expanded_state.size())

        verb_weighted = torch.mul(verb_expanded_state, edge_state)
        region_concat = torch.mul(region_vert_state, edge_state)
        att_over_edges = F.softmax(verb_weighted)
        att_weighted_verb = torch.matmul(att_over_edges, edge_state)
        #att_weighted_region = torch.mul(self.edge_att(region_concat), edge_state)
        #att_weighted_verb = torch.sum(att_weighted_verb_per_edge, 0)

        vert_ctx = torch.cat((torch.unsqueeze(att_weighted_verb,0),region_concat), 0)

        #print('vert context :', vert_ctx.size())
        return vert_ctx
