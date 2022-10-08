import torch
from enum import Enum
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, PaperClassifier
from torch.nn import functional as F
from fc import FCNet, GTH
from attention import Att_0, Att_1, Att_2, Att_3, Att_P, Att_PD, Att_3S
import torch
import random


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        num_hid = opt.num_hid
        activation = opt.activation
        dropG = opt.dropG
        dropW = opt.dropW
        dropout = opt.dropout
        dropL = opt.dropL
        norm = opt.norm
        dropC = opt.dropC
        self.opt = opt

        self.w_emb = WordEmbedding(opt.ntokens, emb_dim=300, dropout=dropW)
        self.w_emb.init_embedding(opt.dataroot + 'glove6b_init_300d.npy')
        self.q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1,
                                       bidirect=False, dropout=dropG, rnn_type='GRU')

        self.q_net = FCNet([self.q_emb.num_hid, num_hid], dropout=dropL, norm=norm, act=activation)
        self.gv_net = FCNet([opt.v_dim, num_hid], dropout=dropL, norm=norm, act=activation)

        self.gv_att_1 = Att_3(v_dim=opt.v_dim, q_dim=self.q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                              act=activation)
        self.gv_att_2 = Att_3(v_dim=opt.v_dim, q_dim=self.q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                              act=activation)
        self.classifier = SimpleClassifier(in_dim=num_hid, hid_dim=2 * num_hid, out_dim=opt.ans_dim,
                                           dropout=dropC, norm=norm, act=activation)

        self.normal = nn.BatchNorm1d(num_hid,affine=False)

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def criterion(self,out_1,out_2,tau_plus=0.1,batch_size=64,beta=1.0,estimator='easy', temperature=0.5):
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        old_neg = neg.clone()
        mask = self.get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)


        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # negative samples similarity scoring
        if estimator=='hard':
            N = batch_size * 2 - 2
            imp = (beta* neg.log()).exp()
            reweight_neg = (imp*neg).sum(dim = -1) / imp.mean(dim = -1)
            Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        elif estimator=='easy':
            Ng = neg.sum(dim=-1)
        else:
            raise Exception('Invalid estimator selected. Please use any of [hard, easy]')
        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng) )).mean()
        return loss

    def forward(self, q, Shuffling_q, Removal_q, positive_q, gv_ori, temperature, estimator, tau_plus, beta):

        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # run GRU on word embeddings [batch, q_dim]
        q_repr = self.q_net(q_emb)
        batch_size = q.size(0)
        
        Shuffling_w_emb = self.w_emb(Shuffling_q)
        Shuffling_q_emb = self.q_emb(Shuffling_w_emb)  # run GRU on word embeddings [batch, q_dim]
        Shuffling_q_repr = self.q_net(Shuffling_q_emb)
        Removal_w_emb = self.w_emb(Removal_q)
        Removal_q_emb = self.q_emb(Removal_w_emb)  # run GRU on word embeddings [batch, q_dim]
        Removal_q_repr = self.q_net(Removal_q_emb)
        positive_w_emb = self.w_emb(positive_q)
        positive_q_emb = self.q_emb(positive_w_emb)  # run GRU on word embeddings [batch, q_dim]
        positive_q_repr = self.q_net(positive_q_emb)

        logits_ori, joint_repr_ori = self.compute_predict(q_repr, q_emb, gv_ori)
        logits_Shuffling, joint_repr_Shuffling = self.compute_predict(Shuffling_q_repr, Shuffling_q_emb, gv_ori)
        logits_Removal, joint_repr_Removal = self.compute_predict(Removal_q_repr, Removal_q_emb, gv_ori)
        logits_positive, joint_repr_positive = self.compute_predict(positive_q_repr, positive_q_emb, gv_ori)
        norm_repr_ori = F.normalize(joint_repr_ori, dim=-1)
        norm_repr_positive = F.normalize(joint_repr_positive, dim=-1)

        cl_loss = self.criterion(norm_repr_ori, norm_repr_positive, tau_plus, batch_size, beta, estimator, temperature) 

        return logits_ori, logits_Shuffling, logits_Removal, logits_positive, cl_loss 

    def compute_predict(self, q_repr, q_emb, v):

        att_1 = self.gv_att_1(v, q_emb)  # [batch, 1, v_dim]
        att_2 = self.gv_att_2(v, q_emb)  # [batch, 1, v_dim]
        att_gv = att_1 + att_2

        gv_embs = (att_gv * v)  # [batch, v_dim]
        gv_emb = gv_embs.sum(1)
        gv_repr = self.gv_net(gv_emb)

        joint_repr = q_repr * gv_repr

        joint_repr_normal = self.normal(joint_repr)
        logits = self.classifier(joint_repr_normal)

        return logits,  joint_repr_normal

