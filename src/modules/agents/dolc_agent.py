import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
import torch as th
import copy


class DOLCAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(DOLCAgent, self).__init__()
        self.args = args
        self.r_dim = args.rnn_hidden_dim
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.n_heads = 4

        self.n_experts = args.n_experts # 4
        self.topk = args.topk # 2

        self.gate = nn.Linear(input_shape, self.n_experts)

        self.bottoms = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(input_shape, self.r_dim),
                nn.ReLU())
             for _ in range(args.n_experts)])

        self.rnns = nn.ModuleList(
            [nn.GRUCell(self.r_dim, self.r_dim)
             for _ in range(args.n_experts)])
        self.hyper_w_1 = nn.ModuleList([nn.Linear(input_shape,self.r_dim * self.r_dim) 
            for _ in range(self.n_experts)])
        self.hyper_w_final = nn.ModuleList([nn.Linear(input_shape,self.r_dim * self.n_actions) 
            for _ in range(self.n_experts)])
        
        self.Att = SelfAttention(input_shape*self.n_agents, self.n_heads, self.n_actions)

        self.V = nn.Sequential(nn.Linear(input_shape * self.n_agents, self.r_dim),
                               nn.ReLU(),
                               nn.Linear(self.r_dim, self.r_dim))       
        
        self.mixer = nn.Linear(input_shape, self.n_experts)
        self.previousObs = None
        self.meanObs = None

    def init_hidden(self,i):
        # make hidden states on same device as model
        return self.bottoms[i][0].weight.new(1, self.r_dim).zero_()

    def forward(self, inputs, hidden_state,t_ep):
        b, a, e = inputs.size()

        if t_ep == 0:
            self.previousObs = th.zeros_like(inputs)
            self.meanObs = copy.deepcopy(inputs) # [b,a,e]
        else:
            self.meanObs = (self.meanObs / t_ep) + ((t_ep - 1) / t_ep) * inputs
        diff = inputs - self.previousObs # [b,a,e]
        
        g = F.sigmoid(self.gate(inputs.view(-1, e))) #[b*a,n_experts]
        indices = th.topk(g,self.topk)[1]
        mask = th.zeros_like(g)
        mask.scatter_(dim=-1, index=indices, value=1)
        g = (g * mask).unsqueeze(-1)
        
        x = [bot(inputs.view(-1, e)) for bot in self.bottoms] #[b*a,r_dim]
        h_out = [h.view(-1, self.r_dim) for h in hidden_state] # [b*a,r_dim]
        h_out = [self.rnns[i](x[i],h_out[i]) for i in range(self.n_experts)] # [b*a,r_dim]
        hh = th.stack(h_out,dim=1) # [b*a,n_experts,r_dim]
        hh = (hh * g).mean(dim=1) #[b*a,self.r_dim]

        # output layers bias
        b1 = self.V(inputs.view(b,-1)) #[b,self.r_dim]
        b1 = b1.unsqueeze(1).expand((b,a,-1)).reshape(b*a,-1) #[b*a,self.r_dim]
        queries = diff.view(b,-1).unsqueeze(1)
        keys = copy.deepcopy(self.meanObs.view(b,-1).unsqueeze(1))
        values = inputs.view(b,-1).unsqueeze(1)
        b2 = self.Att(queries, keys, values) #[b,1,n_heads*n_actions]
        b2 = b2.squeeze(1).view(b,self.n_heads,self.n_actions).mean(dim=1) #[b,n_actions]
        b2 = b2.unsqueeze(1).expand((b,a,-1)).reshape(b*a,-1) #[b*a,n_actions]
        # output layers weights
        outs = []
        for i in range(self.n_experts):
            w1 = self.hyper_w_1[i](inputs.view(-1,e)) #[b*a,r_dim**2]
            w1 = w1.view(-1,self.r_dim,self.r_dim) #[b*a,r_dim,r_dim]
            o1 = F.elu(th.bmm(hh.unsqueeze(1), w1).squeeze(1) + b1) #[b*a,self.r_dim]
            w2 = self.hyper_w_final[i](inputs.view(-1,e)) #[b*a,r_dim*n_actions]
            w2 = w2.view(-1,self.r_dim,self.n_actions) #[b*a,r_dim,n_actions]
            o2 = th.bmm(o1.unsqueeze(1), w2).squeeze(1) + b2 #[b*a,n_actions]
            outs.append(o2) 
        q = th.stack(outs,dim=0) #[n_experts,b*a,n_actions]
        # mix 
        mix_coff = F.sigmoid(self.mixer(inputs.view(-1,e))) #[b*a,n_experts]  
        mix_coff = mix_coff.T.unsqueeze(-1) #[n_experts,b*a,1]
        mix_coff = mix_coff.expand(q.size()) #[n_experts,b*a,n_actions]  

        q = (q * mix_coff).mean(dim=0) #[b*a,n_actions]

        self.previousObs=inputs

        return q.view(b, a, -1), h_out, g.view(b, a, -1)


class SelfAttention(nn.Module):
    def __init__(self, input_size, heads, embed_size):
        super().__init__()
        self.input_size = input_size
        self.heads = heads
        self.emb_size = embed_size

        self.tokeys = nn.Linear(self.input_size, self.emb_size * heads, bias = False)
        self.toqueries = nn.Linear(self.input_size, self.emb_size * heads, bias = False)
        self.tovalues = nn.Linear(self.input_size, self.emb_size * heads, bias = False)
    def forward(self, q,k,v):
        b, t, hin = q.size()
        assert hin == self.input_size, f'Input size {{hin}} should match {{self.input_size}}'       
        h = self.heads 
        e = self.emb_size      
        keys = self.tokeys(q).view(b, t, h, e)
        queries = self.toqueries(k).view(b, t, h, e)
        values = self.tovalues(v).view(b, t, h, e)
        # dot-product attention
        # folding heads to batch dimensions
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))
        dot = th.bmm(queries, keys.transpose(1, 2))
        assert dot.size() == (b*h, t, t)
        # row wise self attention probabilities
        dot = F.softmax(dot, dim=2)
        out = th.bmm(dot, values).view(b, h, t, e)
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)
        return out