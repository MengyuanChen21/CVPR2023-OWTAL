import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from math import sqrt

import model
from PL import PL
from edl_loss import EvidenceLoss

torch.set_default_dtype(torch.float32)


def mutual_kl_loss(a_logits, b_logits):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    ab_kl_loss = kl_loss(
        torch.log_softmax(a_logits, dim=-1),
        torch.softmax(b_logits, dim=-1),
    )
    ba_kl_loss = kl_loss(
        torch.log_softmax(b_logits, dim=-1),
        torch.softmax(a_logits, dim=-1),
    )
    return (ab_kl_loss + ba_kl_loss) / 2.0


class LearnableGaussian(nn.Module):
    def __init__(self):
        super(LearnableGaussian, self).__init__()
        self.mu = torch.nn.Parameter(torch.tensor(0.2), requires_grad=False)
        # self.mu = torch.nn.Parameter(torch.tensor(1), requires_grad=False)
        # self.mu.data.fill_(0.2)
        # self.mu = 0.2

        self.sigma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.sigma.data.fill_(0.2)

    def forward(self, x):
        # y = 1 / ((2 * math.pi) ** 0.5 * self.sigma) * torch.exp(- (x - self.mu) ** 2 / (2 * self.sigma ** 2))
        y = torch.exp(- (x - self.mu) ** 2 / (self.sigma ** 2))
        return y

class SelfAttention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x, concat_orig=False):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        att = torch.bmm(dist, v)
        if concat_orig:
            # att = torch.cat([att, x], dim=-1)
            att = (att + x) / 2.0
        return att


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias) != type(None):
            m.bias.data.fill_(0)


class BWA_fusion_dropout_feat_v2(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()
        embed_dim = 1024
        self.bit_wise_attn = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, (3,), padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.channel_conv = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, (3,), padding=1), nn.LeakyReLU(0.2), nn.Dropout(0.5))
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, (3,), padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, (3,), padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Conv1d(512, 1, (1,)),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())
        self.channel_avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, vfeat, ffeat):
        channelfeat = self.channel_avg(vfeat)
        channel_attn = self.channel_conv(channelfeat)
        bit_wise_attn = self.bit_wise_attn(ffeat)
        filter_feat = torch.sigmoid(bit_wise_attn * channel_attn) * vfeat
        x_atn = self.attention(filter_feat)
        return x_atn, filter_feat


class CO2(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()

        self.args = args['opt']

        self.mu_path = './temp/' + args['opt'].group_name + '/' + args['opt'].model_name

        embed_dim = 2048
        dropout_ratio = args['opt'].dropout_ratio
        self.vAttn = getattr(model, args['opt'].AWM)(1024, args)
        self.fAttn = getattr(model, args['opt'].AWM)(1024, args)

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, (3,), padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_ratio)
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, (1,), padding=0),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_ratio)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, (3,), padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7),
            # nn.Conv1d(embed_dim, n_class + 1, (1,))
            nn.Conv1d(embed_dim, args['opt'].n_known_class + 1, (1,))
        )

        self.channel_avg = nn.AdaptiveAvgPool1d(1)
        self.batch_avg = nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()

        self.pl_module = PL(args['opt'])
        self.selfatt = SelfAttention(embed_dim, embed_dim, embed_dim)
        self.gaussian = LearnableGaussian()
        self.apply(weights_init)

    def intra_weight(self, temp_feat):
        # temp_feat: [n, d, t]
        temp_feat = temp_feat.transpose(-1, -2)                             # [n, t, d]
        cos = lambda m: F.normalize(m) @ F.normalize(m).t()
        cos_similarity = torch.stack([cos(m) for m in temp_feat])           # [n, t, t]
        cos_similarity = F.softmax(cos_similarity, dim=1)                   # [n, t, t]  在1维上进行softmax
        return cos_similarity

    def weighted_sum(self, temp_feat, cos_sim):
        # temp_feat: (n, d, t)      cos_sim: (n, t, t)
        weighted_feat = torch.bmm(temp_feat, cos_sim)                       # (n, d, t)
        weighted_feat = weighted_feat + temp_feat                           # (n, d, t)
        return weighted_feat

    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        v_atn, vfeat = self.vAttn(feat[:, :1024, :], feat[:, 1024:, :])
        f_atn, ffeat = self.fAttn(feat[:, 1024:, :], feat[:, :1024, :])
        x_atn = (f_atn + v_atn) / 2
        nfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(nfeat)

        x_cls = self.classifier(nfeat)

        nfeat = nfeat.transpose(-1, -2)
        x_cls = x_cls.transpose(-1, -2)
        x_atn = x_atn.transpose(-1, -2)
        v_atn = v_atn.transpose(-1, -2)
        f_atn = f_atn.transpose(-1, -2)

        k = max(1, int(x_atn.shape[-2] // args['opt'].k_edl))
        atn_values, atn_idx = torch.topk(x_atn, k=k, dim=1)
        atn_idx_expand = atn_idx.expand([-1, -1, args['opt'].feature_size])
        topk_feat = torch.gather(nfeat, 1, atn_idx_expand)

        topk_feat = self.selfatt(topk_feat, concat_orig=True).mean(dim=-2)

        # Module 1
        cls_logits = self.classifier(topk_feat.unsqueeze(-1)).squeeze(dim=-1)[:, :-1]
        EDLLoss = EvidenceLoss(num_classes=args['opt'].n_known_class, evidence='exp')

        ori_edl_results = EDLLoss(
            output=cls_logits,
            target=args['labels'],
            output_is_evidence=False
        )

        ori_edl_loss = ori_edl_results['loss_cls'].mean()
        ori_uct = ori_edl_results['uncertainty']
        ori_evidence = ori_edl_results['evidence']

        # Module 2
        if args['itr'] > args['opt'].interval:
            mu_from_train_set = np.load(os.path.join(self.mu_path, 'mu.npy'), allow_pickle=True)
            self.gaussian.mu = nn.Parameter(torch.tensor(mu_from_train_set))
        # amplifier_scale = self.gaussian(ori_uct)
        amplifier_scale = 1
        amplifier_coef, pl_loss = self.pl_module(x=topk_feat, labels=args['labels'])
        # scale to [-1, 1]
        amplifier_coef = torch.tanh(amplifier_coef)

        cali_coef = 1 + amplifier_scale * amplifier_coef
        cali_evidence = cali_coef * ori_evidence

        cali_edl_results = EDLLoss(
            output=cali_evidence,
            target=args['labels'],
            output_is_evidence=True,
        )

        cali_edl_loss = cali_edl_results['loss_cls'].mean()
        cali_uct = cali_edl_results['uncertainty']

        return {'feat': nfeat,
                'cas': x_cls,
                'attn': x_atn,
                'v_atn': v_atn,
                'f_atn': f_atn,
                'ori_edl_loss': ori_edl_loss,
                'cali_edl_loss': cali_edl_loss,
                'pl_loss': pl_loss,
                'ori_uct': ori_uct,
                'uct': cali_uct
                }

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):

        if args['itr'] == 190:
            print('Stop point')

        feat, element_logits, element_atn, v_atn, f_atn, ori_edl_loss, cali_edl_loss, pl_loss, ori_uct, uct = \
            outputs['feat'], outputs['cas'], outputs['attn'], outputs['v_atn'], outputs['f_atn'],\
            outputs['ori_edl_loss'], outputs['cali_edl_loss'], outputs['pl_loss'], outputs['ori_uct'], outputs['uct']

        mutual_loss = 0.5 * F.mse_loss(v_atn, f_atn.detach()) + 0.5 * F.mse_loss(f_atn, v_atn.detach())

        element_logits_supp = self._multiply(element_logits, element_atn, include_min=True)

        loss_mil_orig, _ = self.topkloss(element_logits,
                                         labels,
                                         is_back=True,
                                         rat=args['opt'].k,
                                         reduce=None)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                         labels,
                                         is_back=False,
                                         rat=args['opt'].k,
                                         reduce=None)

        loss_3_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)

        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn - element_logits.softmax(-1)[..., [-1]]).abs().mean()

        # total loss
        total_loss = (
                args['opt'].alpha_ori_edl * ori_edl_loss +
                args['opt'].alpha_cali_edl * cali_edl_loss +
                args['opt'].alpha_pl * pl_loss +
                args['opt'].alpha_cls * (loss_mil_orig.mean() + loss_mil_supp.mean()) +
                args['opt'].alpha3 * loss_3_supp_Contrastive +
                args['opt'].alpha4 * mutual_loss +
                args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3 +
                args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3)

        loss_dict = {
            'ori_edl_loss': args['opt'].alpha_ori_edl * ori_edl_loss,
            'cali_edl_loss': args['opt'].alpha_cali_edl * cali_edl_loss,
            'pl_loss': args['opt'].alpha_pl * pl_loss,
            'loss_mil_orig': args['opt'].alpha_cls * loss_mil_orig.mean(),
            'loss_mil_supp': args['opt'].alpha_cls * loss_mil_supp.mean(),
            'loss_supp_contrastive': args['opt'].alpha3 * loss_3_supp_Contrastive,
            'mutual_loss': args['opt'].alpha4 * mutual_loss,
            'norm_loss': args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3,
            'guide_loss': args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3,
            'total_loss': total_loss,
        }

        return total_loss, loss_dict

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)
        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )
        labels_with_back = labels_with_back / (torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back * F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def Contrastive(self, x, element_logits, labels, is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3 * 2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i + 1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)  # (n_feature, n_class)
            Hf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1) / n1)
            Lf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), (1 - atn2) / n2)

            d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (
                    torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))  # 1-similarity
            d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn = outputs

        return element_logits, element_atn


class ANT_CO2(torch.nn.Module):
    def __init__(self, n_feature, n_class, **args):
        super().__init__()

        self.args = args['opt']

        embed_dim = 2048
        dropout_ratio = args['opt'].dropout_ratio

        self.vAttn = getattr(model, args['opt'].AWM)(1024, args)
        self.fAttn = getattr(model, args['opt'].AWM)(1024, args)

        self.feat_encoder = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, (3,), padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_ratio)
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, (1,), padding=0),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_ratio)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv1d(embed_dim, embed_dim, (3,), padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7),
            # nn.Conv1d(embed_dim, n_class + 1, (1,))
            nn.Conv1d(embed_dim, args['opt'].n_known_class + 1, (1,))
        )

        self.channel_avg = nn.AdaptiveAvgPool1d(1)
        self.batch_avg = nn.AdaptiveAvgPool1d(1)
        self.ce_criterion = nn.BCELoss()
        _kernel = ((args['opt'].max_seqlen // args['opt'].t) // 2 * 2 + 1)
        self.pool = nn.AvgPool1d(_kernel, 1, padding=_kernel // 2, count_include_pad=True) \
            if _kernel is not None else nn.Identity()

        self.selfatt = SelfAttention(embed_dim, embed_dim, embed_dim)

        self.apply(weights_init)

    def forward(self, inputs, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        v_atn, vfeat = self.vAttn(feat[:, :1024, :], feat[:, 1024:, :])
        f_atn, ffeat = self.fAttn(feat[:, 1024:, :], feat[:, :1024, :])
        x_atn = (f_atn + v_atn) / 2
        nfeat = torch.cat((vfeat, ffeat), 1)
        nfeat = self.fusion(nfeat)
        x_cls = self.classifier(nfeat)
        x_cls = self.pool(x_cls)
        x_atn = self.pool(x_atn)
        f_atn = self.pool(f_atn)
        v_atn = self.pool(v_atn)

        nfeat = nfeat.transpose(-1, -2)
        x_cls = x_cls.transpose(-1, -2)
        x_atn = x_atn.transpose(-1, -2)
        v_atn = v_atn.transpose(-1, -2)
        f_atn = f_atn.transpose(-1, -2)

        vid_feat, labels = self.arpl_prepare(
            feat=nfeat,
            element_atn=x_atn,
            labels=args['labels']
        )

        arpl_logits, arpl_loss, uct = args['arpl_module'](vid_feat, args['labels'])

        return {'feat': nfeat,
                'cas': x_cls,
                'attn': x_atn,
                'v_atn': v_atn,
                'f_atn': f_atn,
                'arpl_logits': arpl_logits,
                'arpl_loss': arpl_loss,
                'uct': uct,
                }

        # return {'feat': nfeat.transpose(-1, -2),
        #         'cas': x_cls.transpose(-1, -2),
        #         'attn': x_atn.transpose(-1, -2),
        #         'v_atn': v_atn.transpose(-1, -2),
        #         'f_atn': f_atn.transpose(-1, -2)}

    def feat_selfatt_fusion(self, atn, nfeat):
        k = max(1, int(atn.shape[-2] // self.args.k))
        atn_values, atn_idx = torch.topk(
            atn,
            k=k,
            dim=1
        )
        atn_idx_expand = atn_idx.expand([-1, -1, self.args.feature_size])
        topk_feat = torch.gather(nfeat, 1, atn_idx_expand)
        selfatt_topk_feat = self.selfatt(topk_feat)
        return selfatt_topk_feat

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def criterion(self, outputs, labels, **args):
        feat, element_logits, element_atn = outputs['feat'], outputs['cas'], outputs['attn']
        v_atn = outputs['v_atn']
        f_atn = outputs['f_atn']

        arpl_loss = outputs['arpl_loss']
        arpl_logits = outputs['arpl_logits']

        # 按attention取出Top-k个snippet，用自注意力机制交互，分类，平均
        selfatt_topk_feat = self.feat_selfatt_fusion(element_atn, feat).transpose(-1, -2)
        selfatt_vid_logits = self.classifier(selfatt_topk_feat).mean(dim=-1)[..., :-1]

        evd_mutual_loss = 0.5 * F.mse_loss(arpl_logits, selfatt_vid_logits.detach()) + \
                            0.5 * F.mse_loss(selfatt_vid_logits, arpl_logits.detach())

        kl_loss = 1.0 * evd_mutual_loss

        mutual_loss = 0.5 * F.mse_loss(v_atn, f_atn.detach()) + 0.5 * F.mse_loss(f_atn, v_atn.detach())
        element_logits_supp = self._multiply(element_logits, element_atn, include_min=True)
        loss_mil_orig, _ = self.topkloss(element_logits,
                                         labels,
                                         is_back=True,
                                         rat=args['opt'].k,
                                         reduce=None)
        # SAL
        loss_mil_supp, _ = self.topkloss(element_logits_supp,
                                         labels,
                                         is_back=False,
                                         rat=args['opt'].k,
                                         reduce=None)

        loss_3_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)

        loss_norm = element_atn.mean()
        # guide loss
        loss_guide = (1 - element_atn -
                      element_logits.softmax(-1)[..., [-1]]).abs().mean()

        v_loss_norm = v_atn.mean()
        # guide loss
        v_loss_guide = (1 - v_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        f_loss_norm = f_atn.mean()
        # guide loss
        f_loss_guide = (1 - f_atn -
                        element_logits.softmax(-1)[..., [-1]]).abs().mean()

        # total loss
        total_loss = (
                args['opt'].alpha_arpl * arpl_loss +
                args['opt'].alpha_kl * kl_loss +
                loss_mil_orig.mean() + loss_mil_supp.mean() +
                args['opt'].alpha3 * loss_3_supp_Contrastive +
                args['opt'].alpha4 * mutual_loss +
                args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3 +
                args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3)

        loss_dict = {
            'arpl_loss': args['opt'].alpha_arpl * arpl_loss,
            'loss_kl': args['opt'].alpha_kl * kl_loss,
            'loss_mil_orig': loss_mil_orig.mean(),
            'loss_mil_supp': loss_mil_supp.mean(),
            'loss_supp_contrastive': args['opt'].alpha3 * loss_3_supp_Contrastive,
            'mutual_loss': args['opt'].alpha4 * mutual_loss,
            'norm_loss': args['opt'].alpha1 * (loss_norm + v_loss_norm + f_loss_norm) / 3,
            'guide_loss': args['opt'].alpha2 * (loss_guide + v_loss_guide + f_loss_guide) / 3,
            'total_loss': total_loss,
        }

        return total_loss, loss_dict

    def arpl_prepare(self, feat, element_atn, labels):
        args = self.args
        k = max(1, int(feat.shape[-2] // args.arpl_rat))
        atn_values, atn_idx = torch.topk(
            element_atn,
            k=k,
            dim=1
        )
        atn_idx_expand = atn_idx.expand([-1, -1, args.feature_size])
        topk_element_logits = torch.gather(feat, 1, atn_idx_expand)
        video_feature = topk_element_logits.mean(dim=-2)
        if labels is None: return video_feature, None
        labels = labels / torch.sum(labels, dim=1, keepdim=True)
        return video_feature, labels

    def topkloss(self,
                 element_logits,
                 labels,
                 is_back=True,
                 lab_rand=None,
                 rat=8,
                 reduce=None):

        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)

        topk_val, topk_ind = torch.topk(
            element_logits,
            k=max(1, int(element_logits.shape[-2] // rat)),
            dim=-2)
        instance_logits = torch.mean(
            topk_val,
            dim=-2,
        )
        labels_with_back = labels_with_back / (
                torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
        milloss = (-(labels_with_back *
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def Contrastive(self, x, element_logits, labels, is_back=False):
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3 * 2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i + 1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n - 1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)  # (n_feature, n_class)
            Hf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1) / n1)
            Lf2 = torch.mm(torch.transpose(x[i + 1], 1, 0), (1 - atn2) / n2)

            d1 = 1 - torch.sum(Hf1 * Hf2, dim=0) / (
                    torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))  # 1-similarity
            d2 = 1 - torch.sum(Hf1 * Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2 * Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d2 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            sim_loss = sim_loss + 0.5 * torch.sum(
                torch.max(d1 - d3 + 0.5, torch.FloatTensor([0.]).cuda()) * labels[i, :] * labels[i + 1, :])
            n_tmp = n_tmp + torch.sum(labels[i, :] * labels[i + 1, :])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def decompose(self, outputs, **args):
        feat, element_logits, atn_supp, atn_drop, element_atn = outputs

        return element_logits, element_atn
