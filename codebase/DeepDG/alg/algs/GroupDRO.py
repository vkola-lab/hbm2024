#coding=utf-8
import torch
import torch.nn.functional as F
from alg.algs.ERM import ERM

class GroupDRO(ERM):
    """
    Robust ERM minimizes the error at the worst minibatch
    Algorithm 1 from [https://arxiv.org/pdf/1911.08731.pdf]
    """
    def __init__(self,args):
        super(GroupDRO, self).__init__(args)
        self.register_buffer("q", torch.Tensor())
        self.args=args

    def update(self, minibatches, opt,sch, scaler):

        with torch.cuda.amp.autocast():
            if not len(self.q):
                self.q = torch.ones(len(minibatches)).cuda()

            losses = torch.zeros(len(minibatches)).cuda()

            for m in range(len(minibatches)):
                x, y = minibatches[m][1].cuda().float(),minibatches[m][2].cuda().long()
                losses[m] = self.criterion(self.predict(x), y)
                self.q[m] *= (self.args.groupdro_eta * losses[m].data).exp()

            self.q /= self.q.sum()

            loss = torch.dot(losses, self.q)
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

        return {'group': loss.item()}, sch