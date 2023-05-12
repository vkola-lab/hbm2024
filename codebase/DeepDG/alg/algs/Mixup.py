# coding=utf-8
import numpy as np
import torch
import torch.nn.functional as F

from datautil.util import random_pairs_of_minibatches
from alg.algs.ERM import ERM


class Mixup(ERM):
    def __init__(self, args):
        super(Mixup, self).__init__(args)
        self.args = args

    def update(self, minibatches, opt, sch):
        objective = 0

        for (xi, yi, di), (xj, yj, dj) in random_pairs_of_minibatches(self.args, minibatches):
            lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)

            x = (lam * xi + (1 - lam) * xj).cuda().float()

            predictions = self.predict(x)

            objective += lam * F.cross_entropy(predictions, yi.cuda().long())
            objective += (1 - lam) * \
                F.cross_entropy(predictions, yj.cuda().long())

        objective /= len(minibatches)

        opt.zero_grad()
        objective.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': objective.item()}


class OrgMixup(ERM):
    """
    Original Mixup independent with domains
    """
    def __init__(self, args):
        super(OrgMixup, self).__init__(args)
        self.args = args

    def update(self, minibatches, opt, sch, scaler):
        with torch.cuda.amp.autocast():
            device = list(self.parameters())[0].device
            all_x = torch.cat([data[1].float().to(device) for data in minibatches])
            all_y = torch.cat([data[2].long().to(device) for data in minibatches])

            indices = torch.randperm(all_x.size(0))
            all_x2 = all_x[indices]
            all_y2 = all_y[indices]

            lam = np.random.beta(self.args.mixupalpha, self.args.mixupalpha)

            all_x = lam * all_x + (1 - lam) * all_x2
            predictions = self.predict(all_x)

            objective = lam * self.criterion(predictions, all_y)
            objective += (1 - lam) * self.criterion(predictions, all_y2)

            scaler.scale(objective / self.accum_iter).backward()

            self.acc_steps += 1
            if self.acc_steps == self.accum_iter:
                scaler.step(opt)
                scaler.update()
                self.zero_grad()
                if sch:
                    sch.step()
                self.acc_steps = 0
                torch.cuda.empty_cache()

        return {"class": objective.item()}, sch
