# coding=utf-8
import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd

from alg.algs.ERM import ERM


class RSC(ERM):
    def __init__(self, args):
        super(RSC, self).__init__(args)
        self.drop_f = (1 - args.rsc_f_drop_factor) * 100
        self.drop_b = (1 - args.rsc_b_drop_factor) * 100
        self.num_classes = args.num_classes
        self.accum_iter = args.accum_iter
        self.acc_steps = 0
        
    def update(self, minibatches, opt, sch, scaler):
        print('---------------RSC update---------------')
        device = list(self.parameters())[0].device
        with torch.cuda.amp.autocast():
            all_x = torch.cat([data[1].to(device).float() for data in minibatches])
            all_y = torch.cat([data[2].to(device).long() for data in minibatches])
            all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
            all_f = self.featurizer(all_x)
            f_shape = all_f.shape
            print('f_shape: ', f_shape)
            
            all_p = self.classifier(all_f)
            # if len(f_shape) > 2:
            #     all_f= all_f.view(all_f.size(0), -1)
            #     all_f._requires_grad = True
            print('all_f: ', all_f.size())
            print('all_p: ', all_p.size())
            print('all_o: ', all_o.size())

            # Equation (1): compute gradients with respect to representation
            all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]
            # print('all_g: ', all_g.size())

            # Equation (2): compute top-gradient-percentile mask
            if len(all_g.shape) > 2:
                all_g= all_g.view(all_g.size(0), -1)
            percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
            percentiles = torch.Tensor(percentiles)
            # print('percentiles: ', percentiles.size())
            percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
            # print('percentiles squeezed & repeated: ', percentiles.size())
            mask_f = all_g.lt(percentiles.cuda()).float()
            print('mask_f: ', mask_f.size())
            mask_f = mask_f.view(all_f.shape)
            # print('mask_f: ', mask_f.size())
            # Equation (3): mute top-gradient-percentile activations
            all_f_muted = all_f * mask_f
            # if len(f_shape) > 2:
            #     all_f = all_f.view(f_shape)
            # print('final all_f: ', all_f.size())    
            # Equation (4): compute muted predictions
            all_p_muted = self.classifier(all_f_muted)

            # Section 3.3: Batch Percentage
            
            all_s = F.softmax(all_p, dim=1)
            all_s_muted = F.softmax(all_p_muted, dim=1)
            changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
            percentile = np.percentile(changes.detach().cpu(), self.drop_b)
            mask_b = changes.lt(percentile).float()
            print('mask_b: ', mask_b.size())
            mask_b = mask_b.view(-1, 1)
            print('mask_b: ', mask_b.size())
            if len(all_f.shape) > 2: 
                mask = torch.logical_or(mask_f.view(mask_f.size(0),-1), mask_b).float()
                mask = mask.view(all_f.shape)
            else:
                mask = torch.logical_or(mask_f, mask_b).float()
            # print('mask: ', mask.size())
            # Equations (3) and (4) again, this time mutting over examples
            all_p_muted_again = self.classifier(all_f * mask)

            # Equation (5): update
            loss = self.criterion(all_p_muted_again, all_y)
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

        return {'class': loss.item()}, sch
