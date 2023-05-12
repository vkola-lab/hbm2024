# coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM


class CORAL(ERM):
    def __init__(self, args):
        super(CORAL, self).__init__(args)
        self.args = args
        self.kernel_type = "mean_cov"

    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def update(self, minibatches, opt, sch, scaler):
        objective = 0
        penalty = 0
        nmb = len(minibatches)
        device = list(self.parameters())[0].device
        with torch.cuda.amp.autocast():
            features = [self.featurizer(
                data[1].to(device).float()) for data in minibatches]
            classifs = [self.classifier(fi) for fi in features]
            targets = [data[2].to(device).long() for data in minibatches]

            for i in range(nmb):
                objective += self.criterion(classifs[i], targets[i])
                for j in range(i + 1, nmb):
                    penalty += self.coral(features[i].view(features[i].size(0), -1), features[j].view(features[j].size(0), -1))

            objective /= nmb
            if nmb > 1:
                penalty /= (nmb * (nmb - 1) / 2)

            scaler.scale(objective + (self.args.mmd_gamma*penalty)).backward()
            
            self.acc_steps += 1
            if self.acc_steps == self.accum_iter:
                scaler.step(opt)
                scaler.update()
                self.zero_grad()
                if sch:
                    sch.step()
                self.acc_steps = 0
                torch.cuda.empty_cache()
            
        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'class': objective.item(), 'coral': penalty, 'total': (objective.item() + (self.args.mmd_gamma*penalty))}, sch
