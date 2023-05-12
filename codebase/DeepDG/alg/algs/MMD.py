# coding=utf-8
import torch
import torch.nn.functional as F

from alg.algs.ERM import ERM


class MMD(ERM):
    def __init__(self, args):
        super(MMD, self).__init__(args)
        self.args = args
        self.kernel_type = "gaussian"

    def my_cdist(self, x1, x2):
        print('---------MMD my_cdist()---------')
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        print('x1_norm: ', x1_norm.size())
        print('x2_norm: ', x2_norm.size())
        print(x2_norm.transpose(-2,-1).size())
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        ic(D)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        device = list(self.parameters())[0].device
        # weights = torch.tensor([self.weight[l] for l in y]).to(device)
        Kxx = (self.gaussian_kernel(x, x)).mean()
        Kyy = (self.gaussian_kernel(y, y)).mean()
        Kxy = (self.gaussian_kernel(x, y)).mean()
        return Kxx + Kyy - 2 * Kxy

    def update(self, minibatches, opt, sch, scaler):
        print('-----------def update-------------')
        device = list(self.parameters())[0].device
        self.weight = self.weight.to(device)
        with torch.cuda.amp.autocast():
            objective = 0
            penalty = 0
            nmb = len(minibatches)
            print('nmb: ', nmb)
            ic(len(minibatches), len(minibatches[0]))
            features = [self.featurizer(
                data[1].to(device).float()) for data in minibatches]
            # print('features: ', len(features), features[0].size())
            classifs = [self.classifier(fi) for fi in features]
            # print('classifs: ', len(classifs), classifs[0].size())
            targets = [data[2].to(device).long() for data in minibatches]

            for i in range(nmb):
                print('i: ', i)
                objective += self.criterion(classifs[i], targets[i])
                for j in range(i + 1, nmb):
                    print('j: ', j)
                    print(f'feat i: {features[i].size()}, feat j: {features[j].size()}')
                    ic(features[i].view(features[i].size(0), -1).size(), features[j].view(features[j].size(0), -1).size())
                    penalty += self.mmd(features[i].view(features[i].size(0), -1), features[j].view(features[j].size(0), -1))

            # add class conditional mmd for single DG
            ic(objective, penalty)

            objective /= nmb
            if nmb > 1:
                penalty /= (nmb * (nmb - 1) / 2)

            loss = objective + (self.args.mmd_gamma*penalty)
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
            
            if torch.is_tensor(penalty):
                penalty = penalty.item()

        return {'class': objective.item(), 'mmd': penalty, 'total': loss.item()}, sch
