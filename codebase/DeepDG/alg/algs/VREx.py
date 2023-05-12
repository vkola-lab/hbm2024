# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM


class VREx(ERM):
    """V-REx algorithm from http://arxiv.org/abs/2003.00688"""

    def __init__(self, args):
        super(VREx, self).__init__(args)
        self.register_buffer('update_count', torch.tensor([0]))
        self.args = args

    def update(self, minibatches, opt, sch, scaler):
        if self.update_count >= self.args.anneal_iters:
            penalty_weight = self.args.lam
        else:
            penalty_weight = 1.0

        with torch.cuda.amp.autocast():
            nll = 0.
            device = list(self.parameters())[0].device
            all_x = torch.cat([data[1].float().to(device) for data in minibatches])
            all_logits = self.network(all_x)
            all_logits_idx = 0
            losses = torch.zeros(len(minibatches)).to(device)

            for i, data in enumerate(minibatches):
                logits = all_logits[all_logits_idx:all_logits_idx +
                                    data[1].shape[0]]
                all_logits_idx += data[1].shape[0]
                nll = self.criterion(logits, data[2].long().to(device))
                losses[i] = nll

            mean = losses.mean()
            penalty = ((losses - mean) ** 2).mean()
            loss = mean + penalty_weight * penalty
            scaler.scale(loss / self.accum_iter).backward()
            
            self.acc_steps += 1
            if self.acc_steps == self.accum_iter:
                scaler.step(opt)
                scaler.update()
                self.zero_grad()
                if sch:
                    sch.step()
                self.acc_steps = 0
                torch.cuda.empty_cache()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
                'penalty': penalty.item()}, sch
