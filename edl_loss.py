import torch
import torch.nn.functional as F

from base import BaseWeightedLoss


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


class EvidenceLoss(BaseWeightedLoss):
    """Evidential MSE Loss."""

    def __init__(self, num_classes,
                 evidence='relu',
                 loss_type='log',
                 with_kldiv=False,
                 with_annealing=False,
                 disentangle=False,
                 annealing_method='step',
                 annealing_start=0.01,
                 annealing_step=10,
                 redl_lamb=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.evidence = evidence
        self.loss_type = loss_type
        self.with_kldiv = with_kldiv
        self.with_annealing = with_annealing
        self.disentangle = disentangle
        self.annealing_method = annealing_method
        self.annealing_start = annealing_start
        self.annealing_step = annealing_step
        self.eps = 1e-10
        self.redl_lamb = redl_lamb

    def kl_divergence(self, alpha):
        beta = torch.ones([1, self.num_classes], dtype=torch.float32).to(alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - \
              torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                            keepdim=True) - torch.lgamma(S_beta)

        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)

        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                       keepdim=True) + lnB + lnB_uni
        return kl

    def loglikelihood_loss(self, y, alpha):
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum(
            (y - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        return loglikelihood_err, loglikelihood_var

    def mse_loss(self, y, alpha, annealing_coef):
        """Used only for loss_type == 'mse'
        y: the one-hot labels (batchsize, num_classes)
        alpha: the predictions (batchsize, num_classes)
        epoch_num: the current training epoch
        """
        losses = {}
        loglikelihood_err, loglikelihood_var = self.loglikelihood_loss(y, alpha)
        losses.update({'loss_cls': loglikelihood_err, 'loss_var': loglikelihood_var})

        losses.update({'lambda': annealing_coef})
        if self.with_kldiv:
            kl_alpha = (alpha - 1) * (1 - y) + 1
            kl_div = annealing_coef * \
                     self.kl_divergence(kl_alpha)
            losses.update({'loss_kl': kl_div})

        if self.with_avuloss:
            S = torch.sum(alpha, dim=1, keepdim=True)  # Dirichlet strength
            pred_score = alpha / S
            uncertainty = self.num_classes / S
            # avu_loss = annealing_coef * 
        return losses

    def ce_loss(self, target, y, alpha, annealing_coef):
        """Used only for loss_type == 'ce'
        target: the scalar labels (batchsize,)
        alpha: the predictions (batchsize, num_classes), alpha >= 1
        epoch_num: the current training epoch
        """
        losses = {}
        # (1) the classification loss term
        S = torch.sum(alpha, dim=1, keepdim=True)
        pred_score = alpha / S
        loss_cls = F.nll_loss(torch.log(pred_score), target, reduction='none')
        losses.update({'loss_cls': loss_cls})

        # (2) the likelihood variance term
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
        losses.update({'loss_var': loglikelihood_var})

        # (3) the KL divergence term
        kl_alpha = (alpha - 1) * (1 - y) + 1
        kl_div = annealing_coef * \
                 self.kl_divergence(kl_alpha)
        losses.update({'loss_kl': kl_div, 'lambda': annealing_coef})
        return losses

    def edl_loss(self, func, y, alpha, annealing_coef, target):
        """Used for both loss_type == 'log' and loss_type == 'digamma'
        func: function handler (torch.log, or torch.digamma)
        y: the one-hot labels (batch_size, num_classes)
        alpha: the predictions (batch_size, num_classes)
        epoch_num: the current training epoch
        """
        # BALD Uncertainty
        # --------------------------------------------------------------------------------
        # losses = {}
        # S = torch.sum(alpha, dim=1, keepdim=True)
        # pred = alpha / S
        # uncertainty = compute_BALD_uncertainty(pred)
        # label_num = torch.sum(y, dim=1, keepdim=True)
        # temp = 1 / alpha * y
        # g = (1 - uncertainty) * label_num * torch.div(temp, torch.sum(temp, dim=1, keepdim=True))
        # A = torch.sum(g * (func(S) - func(alpha)), dim=1, keepdim=True)

        # Final DELU
        # --------------------------------------------------------------------------------
        # losses = {}
        # S = torch.sum(alpha, dim=1, keepdim=True)
        # uncertainty = self.num_classes / S
        # label_num = torch.sum(y, dim=1, keepdim=True)
        # temp = 1 / alpha * y
        # g = (1 - uncertainty.detach()) * label_num * torch.div(temp, torch.sum(temp, dim=1, keepdim=True))
        # A = torch.sum(g * (func(S) - func(alpha)), dim=1, keepdim=True)

        # Traditional EDL
        # --------------------------------------------------------------------------------

        losses = {}
        if y is None:
            A = torch.tensor([0.])
        else:
            S = torch.sum(alpha, dim=1, keepdim=True)
            A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

        # --------------------------------------------------------------------------------

        losses.update({'loss_cls': A})

        if self.with_annealing:
            losses.update({'lambda': annealing_coef})

        if self.with_kldiv:
            kl_alpha = (alpha - 1) * (1 - y) + 1
            kl_div = annealing_coef * \
                     self.kl_divergence(kl_alpha)
            losses.update({'loss_kl': kl_div})

        return losses

    def compute_annealing_coef(self, **kwargs):
        assert 'epoch' in kwargs, "epoch number is missing!"
        assert 'total_epoch' in kwargs, "total epoch number is missing!"
        epoch_num, total_epoch = kwargs['epoch'], kwargs['total_epoch']
        # annealing coefficient
        if self.annealing_method == 'step':
            annealing_coef = torch.min(torch.tensor(
                1.0, dtype=torch.float32), torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32))
        elif self.annealing_method == 'exp':
            annealing_start = torch.tensor(self.annealing_start, dtype=torch.float32)
            annealing_coef = annealing_start * torch.exp(-torch.log(annealing_start) / total_epoch * epoch_num)
        else:
            raise NotImplementedError
        return annealing_coef

    def _forward(self, output, target, output_is_evidence=False, **kwargs):
        """Forward function.
        Args:
            output (torch.Tensor): The class score (before softmax).
            target (torch.Tensor): The ground truth label.
            epoch_num: The number of epochs during training.
        Returns:
            torch.Tensor: The returned EvidenceLoss loss.
        """
        if output_is_evidence:
            evidence = output
        else:
            # get evidence
            if self.evidence == 'relu':
                evidence = relu_evidence(output)
            elif self.evidence == 'exp':
                evidence = exp_evidence(output)
            elif self.evidence == 'softplus':
                evidence = softplus_evidence(output)
            else:
                raise NotImplementedError

        alpha = evidence + self.redl_lamb

        # Our target is a vector, as result, no need for one-hot embedding
        y = target

        # # one-hot embedding for the target
        # y = torch.eye(self.num_classes).to(output.device)
        # y = y[target]

        # compute annealing coefficient
        if self.with_annealing:
            annealing_coef = self.compute_annealing_coef(**kwargs)
        else:
            annealing_coef = None

        # compute the EDL loss
        if self.loss_type == 'mse':
            results = self.mse_loss(y, alpha, annealing_coef)
        elif self.loss_type == 'log':
            results = self.edl_loss(torch.log, y, alpha, annealing_coef, target)
        elif self.loss_type == 'digamma':
            results = self.edl_loss(torch.digamma, y, alpha, annealing_coef, target)
        elif self.loss_type == 'cross_entropy':
            results = self.ce_loss(target, y, alpha, annealing_coef)
        else:
            raise NotImplementedError

        uncertainty = self.redl_lamb * self.num_classes / torch.sum(alpha, dim=1, keepdim=True)
        results.update({'uncertainty': uncertainty})
        results.update({'evidence': evidence})

        return results
