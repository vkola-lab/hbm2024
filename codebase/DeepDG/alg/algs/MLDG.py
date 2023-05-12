# coding=utf-8
import torch
import copy
import torch.nn.functional as F

from alg.opt import get_optimizer
import torch.autograd as autograd
from datautil.util import random_pairs_of_minibatches_by_domainperm
from alg.algs.ERM import ERM


class MLDG(ERM):
    def __init__(self, args):
        super(MLDG, self).__init__(args)
        self.args = args

    def update(self, minibatches, opt, sch, scaler):
        """
        For computational efficiency, we do not compute second derivatives.
        """
        num_mb = len(minibatches)
        objective = 0

        opt.zero_grad()
        for p in self.network.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        device = list(self.parameters())[0].device
        self.args.lr = sch.get_last_lr()[0]
        ic(self.args.lr)
        for (xi, yi) , (xj, yj) in random_pairs_of_minibatches_by_domainperm(minibatches):
            # fine-tune clone-network on task "i"
            inner_net = copy.deepcopy(self.network)
            
            xi, yi, xj, yj = xi.to(device).float(), yi.to(device).long(), \
                            xj.to(device).float(), yj.to(device).long()

            inner_opt = get_optimizer(inner_net, self.args, True)
            # inner_sch = get_scheduler(inner_opt, self.args)

            inner_obj = self.criterion(inner_net(xi), yi)
            scaler.scale(inner_obj / self.accum_iter).backward()
            if self.acc_steps == self.accum_iter:
                scaler.step(inner_opt)
                scaler.update()
                inner_opt.zero_grad()
                

            for p_tgt, p_src in zip(self.network.parameters(),
                                    inner_net.parameters()):
                if p_src.grad is not None:
                    p_tgt.grad.data.add_((p_src.grad.data / num_mb) / self.accum_iter)

            objective += inner_obj.item() / self.accum_iter
            ic(inner_obj)
            loss_inner_j = self.criterion(inner_net(xj), yj)
            grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(),
                                         allow_unused=True)
            ic(loss_inner_j)
            objective += (self.args.mldg_beta * loss_inner_j).item() / self.accum_iter

            for p, g_j in zip(self.network.parameters(), grad_inner_j):
                if g_j is not None:
                    p.grad.data.add_((
                        self.args.mldg_beta * g_j.data / num_mb) / self.accum_iter)

        objective /= len(minibatches)
        ic(objective)
        self.acc_steps += 1
        if self.acc_steps == self.accum_iter:
            scaler.step(opt)
            scaler.update()
            self.zero_grad()
            if sch:
                sch.step()
            self.acc_steps = 0
            torch.cuda.empty_cache()

        return {'total': objective, 'inner': loss_inner_j.item(), 'class': inner_obj.item()}, sch
