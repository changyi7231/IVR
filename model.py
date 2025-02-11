import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_entities, num_relations, model_name, dimension, parts, regularization, alpha):
        super(Model, self).__init__()
        self.model_name = model_name
        self.dimension = dimension
        self.parts = parts
        self.regularization = regularization
        self.p = 2
        self.q = alpha / 2
        bound = 0.01

        if model_name == 'CP':
            assert self.parts == 1
            self.register_buffer('W', torch.Tensor([[[1]]]))
            self.entity_h = nn.Embedding(num_entities, dimension, sparse=True)
            self.entity_t = nn.Embedding(num_entities, dimension, sparse=True)
            self.relation = nn.Embedding(num_relations, dimension, sparse=True)
            nn.init.uniform_(self.entity_h.weight, -bound, bound)
            nn.init.uniform_(self.entity_t.weight, -bound, bound)
            nn.init.uniform_(self.relation.weight, -bound, bound)
        elif model_name == 'ComplEx':
            assert self.parts == 2
            self.register_buffer('W', torch.Tensor([[[1, 0], [0, 1]], [[0, 1], [-1, 0]]]))
            self.entity = nn.Embedding(num_entities, dimension)
            self.relation = nn.Embedding(num_relations, dimension)
            nn.init.uniform_(self.entity.weight, -bound, bound)
            nn.init.uniform_(self.relation.weight, -bound, bound)
        elif model_name == 'SimplE':
            assert self.parts == 2
            self.register_buffer('W', torch.Tensor([[[0, 1], [0, 0]], [[0, 0], [1, 0]]]))
            self.entity = nn.Embedding(num_entities, dimension, sparse=True)
            self.relation = nn.Embedding(num_relations, dimension, sparse=True)
            nn.init.uniform_(self.entity.weight, -bound, bound)
            nn.init.uniform_(self.relation.weight, -bound, bound)
        elif model_name == 'ANALOGY':
            assert self.parts == 4
            self.register_buffer('W', torch.Tensor([[[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                                    [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]]))
            self.entity = nn.Embedding(num_entities, dimension, sparse=True)
            self.relation = nn.Embedding(num_relations, dimension, sparse=True)
            nn.init.uniform_(self.entity.weight, -bound, bound)
            nn.init.uniform_(self.relation.weight, -bound, bound)
        elif model_name == 'QuatE':
            assert self.parts == 4
            self.register_buffer('W', torch.Tensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                    [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]],
                                                    [[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]],
                                                    [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]]]))
            self.entity = nn.Embedding(num_entities, dimension, sparse=True)
            self.relation = nn.Embedding(num_relations, dimension, sparse=True)
            nn.init.uniform_(self.entity.weight, -bound, bound)
            nn.init.uniform_(self.relation.weight, -bound, bound)
        elif model_name == 'TuckER':
            self.W = nn.Parameter(torch.randn(parts, parts, parts))
            self.entity = nn.Embedding(num_entities, dimension, sparse=True)
            self.relation = nn.Embedding(num_relations, dimension, sparse=True)
            nn.init.uniform_(self.W, -bound, bound)
            nn.init.uniform_(self.entity.weight, -bound, bound)
            nn.init.uniform_(self.relation.weight, -bound, bound)
        else:
            raise RuntimeError('wrong model')

    def forward(self, heads, relations, tails):
        if self.model_name == 'CP':
            h = self.entity_h(heads).view(-1, self.dimension//self.parts, self.parts)
            r = self.relation(relations).view(-1, self.dimension//self.parts, self.parts)
            t = self.entity_t(tails).view(-1, self.dimension//self.parts, self.parts)
        else:
            h = self.entity(heads).view(-1, self.dimension//self.parts, self.parts)
            r = self.relation(relations).view(-1, self.dimension//self.parts, self.parts)
            t = self.entity(tails).view(-1, self.dimension//self.parts, self.parts)

        if self.regularization == 'N3':
            h_norm = ((torch.abs(h) ** 2).sum(2) ** 1.5).sum(1)
            r_norm = ((torch.abs(r) ** 2).sum(2) ** 1.5).sum(1)
            t_norm = ((torch.abs(t) ** 2).sum(2) ** 1.5).sum(1)
        else:
            h_norm = ((torch.abs(h) ** self.p).sum(2) ** self.q).sum(1)
            r_norm = ((torch.abs(r) ** self.p).sum(2) ** self.q).sum(1)
            t_norm = ((torch.abs(t) ** self.p).sum(2) ** self.q).sum(1)

        hr_norm = (((torch.abs(h) ** self.p).sum(2) * (torch.abs(r) ** self.p).sum(2)) ** self.q).sum(1)
        rt_norm = (((torch.abs(r) ** self.p).sum(2) * (torch.abs(t) ** self.p).sum(2)) ** self.q).sum(1)
        th_norm = (((torch.abs(t) ** self.p).sum(2) * (torch.abs(h) ** self.p).sum(2)) ** self.q).sum(1)

        x1 = torch.matmul(h, self.W.view(self.parts, -1))
        x2 = torch.matmul(r, self.W.permute(1, 2, 0).contiguous().view(self.parts, -1))
        x3 = torch.matmul(t, self.W.permute(2, 0, 1).contiguous().view(self.parts, -1))

        # Since the size of h is (b x dp) and the size of x1 is (b x d x p^2),
        # we divide the sum in wh_norm by self.parts to make the numerical scale similar
        wh_norm = (((torch.abs(x1) ** self.p).sum(2) / self.parts) ** self.q).sum(1)
        wr_norm = (((torch.abs(x2) ** self.p).sum(2) / self.parts) ** self.q).sum(1)
        wt_norm = (((torch.abs(x3) ** self.p).sum(2) / self.parts) ** self.q).sum(1)

        x1 = torch.matmul(r.unsqueeze(-2), x1.view(-1, self.dimension//self.parts, self.parts, self.parts)).squeeze(-2)
        x2 = torch.matmul(t.unsqueeze(-2), x2.view(-1, self.dimension//self.parts, self.parts, self.parts)).squeeze(-2)
        x3 = torch.matmul(h.unsqueeze(-2), x3.view(-1, self.dimension//self.parts, self.parts, self.parts)).squeeze(-2)

        whr_norm = ((torch.abs(x1) ** self.p).sum(2) ** self.q).sum(1)
        wrt_norm = ((torch.abs(x2) ** self.p).sum(2) ** self.q).sum(1)
        wth_norm = ((torch.abs(x3) ** self.p).sum(2) ** self.q).sum(1)

        if self.model_name == 'CP':
            scores = torch.matmul(x1.view(-1, self.dimension), self.entity_t.weight.t())
        else:
            scores = torch.matmul(x1.view(-1, self.dimension), self.entity.weight.t())
        if self.regularization == 'w/o':
            factor1, factor2, factor3, factor4 = 0.0, 0.0, 0.0, 0.0
        elif self.regularization == 'F2':
            factor1 = torch.mean(h_norm) + torch.mean(r_norm) + torch.mean(t_norm)
            factor2, factor3, factor4 = 0.0, 0.0, 0.0
        elif self.regularization == 'N3':
            factor1 = torch.mean(h_norm) + torch.mean(r_norm) + torch.mean(t_norm)
            factor2, factor3, factor4 = 0.0, 0.0, 0.0
        elif self.regularization == 'DURA':
            factor1 = torch.mean(h_norm) + torch.mean(t_norm)
            factor2, factor3 = 0.0, 0.0
            factor4 = torch.mean(whr_norm) + torch.mean(wrt_norm)
        elif self.regularization == 'IVR':
            factor1 = torch.mean(h_norm) + torch.mean(r_norm) + torch.mean(t_norm)
            factor2 = torch.mean(hr_norm) + torch.mean(rt_norm) + torch.mean(th_norm)
            factor3 = torch.mean(wh_norm) + torch.mean(wr_norm) + torch.mean(wt_norm)
            factor4 = torch.mean(whr_norm) + torch.mean(wrt_norm) + torch.mean(wth_norm)
        else:
            raise RuntimeError('wrong regularization')
        return scores, factor1, factor2, factor3, factor4
