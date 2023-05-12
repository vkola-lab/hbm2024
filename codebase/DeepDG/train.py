# coding=utf-8
from datetime import datetime
import os
import sys
import time
import numpy as np
import argparse
import json
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import alg
from alg.opt import *
from alg import alg, modelopera
from network.img_network import UNet3DBase
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, Tee, img_param_init, print_environ, to_row
from datautil.getdataloader import get_img_dataloader
import wandb
import icecream
from icecream import ic, install
install()
ic.configureOutput(includeContext=True)
ic.enable()

def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--config', type=str, default=None, help='path to config json file')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--swad', type=str, default=None, choices=['LossValley', 'IIDMax'])
    parser.add_argument('--n_converge', type=int, default=3)
    parser.add_argument('--n_tolerance', type=int, default=6)
    parser.add_argument('--tolerance_ratio', type=float, default=0.3)
    parser.add_argument('--freeze_bn', action='store_true')
    
    parser.add_argument('--alpha', type=float,
                        default=1, help='DANN dis alpha')
    parser.add_argument('--anneal_iters', type=int,
                        default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size')
    parser.add_argument('--accum_iter', type=int, default=1,
                        help="Accumulate gradients")
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam hyper-param')
    parser.add_argument('--checkpoint_freq', type=int,
                        default=3, help='Checkpoint every N epoch')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn", "gap", "conv"])
    parser.add_argument('--attention', action='store_true', help='Option to add attention module after the feature extractor')
    parser.add_argument('--replace_att', action='store_true', help='Replace attention maps with SHAP priors')
    parser.add_argument('--data_file', type=str, default='',
                        help='root_dir')
    parser.add_argument('--dataset', type=str, default='office')
    parser.add_argument('--fold', type=int, help='Training data fold number')
    parser.add_argument('--cohort', type=str, default='NACC_NC_MCI_AD')
    parser.add_argument('--data_dir', type=str, default='', help='data dir')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--augmentation', action='store_true', help='Data augmentation')
    parser.add_argument('--ssl', action='store_true', help='Set true for Self-supervised learning')
    parser.add_argument('--dis_hidden', type=int,
                        default=256, help='dis hidden dimension')
    parser.add_argument('--gpu_id', type=int,
                        default=None, help="device id to run")
    parser.add_argument('--gpu_ids', type=int, nargs='+',
                        default=None, help="List of GPU IDs for DDP")
    parser.add_argument('--groupdro_eta', type=float,
                        default=1, help="groupdro eta")
    parser.add_argument('--inner_lr', type=float,
                        default=1e-2, help="learning rate used in MLDG")
    parser.add_argument('--lam', type=float,
                        default=1, help="tradeoff hyperparameter used in VREx")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=120, help="max iterations")
    parser.add_argument('--start_epoch', type=int,
                        default=None, help="start epoch")
    parser.add_argument('--steps_per_epoch', type=int,
                        default=10, help="steps per epoch")                    
    parser.add_argument('--mixupalpha', type=float,
                        default=0.2, help='mixup hyper-param')
    parser.add_argument('--mldg_beta', type=float,
                        default=1, help="mldg hyper-param")
    parser.add_argument('--mmd_gamma', type=float,
                        default=1, help='MMD, CORAL hyper-param')
    parser.add_argument('--cad_temp', type=float, default=0.07, help="CAD temperature")
    parser.add_argument('--is_conditional', action="store_true", help="Set to True if conditional CAD")
    parser.add_argument('is_project', action="store_true", help="Set to True to use projection head in CAD")
    parser.add_argument('is_normalized', action="store_true", help="Set to True for normalization to representation when computing loss in CAD")
    parser.add_argument('is_flipped', action="store_true", help="whether flip maximize log(p) (False) to minimize -log(1-p) (True) for the bottleneck loss in CAD")

    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')
    parser.add_argument('--net', type=str, default='resnet50',
                        help="featurizer: vgg16, resnet50, resnet101,DTNBase, resnet3d, attcnn")
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--rsc_f_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg", "mri_dg"], help='now only support image tasks')
    parser.add_argument('--tau', type=float, default=1, help="andmask tau")
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[0], help='target domains')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--resume', type=str, help='checkpoint path')
    parser.add_argument('--output', type=str,
                        default="train_output", help='result output path')
    parser.add_argument('--write_raw_score', action='store_true', help='Set True to write raw scores into .txt file')                        
    parser.add_argument('--save_emb', action='store_true', help='Set True to save embedding into.npy file')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # SHAP align arguments
    parser.add_argument('--layer', default='input', help='layer to use the SHAP maps at')
    parser.add_argument('--align_loss', default='cos', help='Similarity loss for XAIalign')
    parser.add_argument('--align_cohort', action='store_true', help='Align with cohort shap map as well')
    parser.add_argument('--align_start', type=int, help='Epoch at which to start align strategy')
    parser.add_argument('--align_lambda', type=float, help='Align loss regularizer 1')
    parser.add_argument('--align_gamma', type=float, help='Align loss regularizer 2')
    args = parser.parse_args()
    # args.steps_per_epoch = 10
    args.data_dir = args.data_file+args.data_dir
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.swad:
        args.save_path = os.path.join(args.output, f'{args.algorithm}_{args.swad}')
    else:
        args.save_path = os.path.join(args.output, args.algorithm)
    if args.pretrained:
        args.save_path = os.path.join(args.save_path, f'{args.net}_{args.classifier}_pretrained')
    else:
        args.save_path = os.path.join(args.save_path, f'{args.net}_{args.classifier}')
    if args.attention:
        args.save_path += f'_attention_{args.cohort}'
    else:    
        args.save_path += f'_{args.cohort}'
    args.save_path += '_replace_att' if args.replace_att else ''
    args.save_path += f'aug_minmax_normalize_lr{args.lr}' if args.augmentation else f'_minmax_normalize_lr{args.lr}'
    if args.algorithm == 'XAIalign':
        if args.align_cohort:
            args.save_path += '_align_cohort_'
        args.save_path += f'layer_feats_{args.layer}_{args.align_loss}_start{args.align_start}_lambda{args.align_lambda}_gamma{args.align_gamma}'
    elif args.algorithm == 'MLDG':
        args.save_path += f'_mldg_beta{args.mldg_beta}'
    elif args.algorithm == 'MMD':
        args.save_path += f'_mmd_gamma{args.mmd_gamma}'
    args.save_path += f'_{args.fold}'

    print('save path: ', args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    # sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    # sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args)
    
    if args.config:
        assert os.path.exists(args.config), 'Config file not found!'
        with open(args.config, 'r') as f:
            json_dict = json.load(f)
            if args.eval:
                del json_dict['img_dataset']
                del json_dict['save_path']
                del json_dict['wandb']
            args.__dict__.update(json_dict)
            args = parser.parse_args(namespace=args)
    
    print_environ()
    return args

class ModifiedFeaturizer(torch.nn.Module):
    def __init__(self, featurizer, shap_priors) -> None:
        super().__init__()
        self.featurizer = featurizer
        self.register_buffer('shap_priors', torch.nn.functional.softmax(shap_priors))

    def forward(self, x, attention=False):
        if hasattr(self.featurizer, 'instance_norm') and self.featurizer.instance_norm:
            x = self.featurizer.norm(x)

        if isinstance(self.featurizer, UNet3DBase):
            x, _ = self.featurizer.down_tr64(x)
            # ic(self.out64.shape, self.skip_out64.shape)
            x, _ = self.featurizer.down_tr128(x)
            # ic(self.out128.shape, self.skip_out128.shape)
            x, _ = self.featurizer.down_tr256(x)
            # ic(self.out256.shape, self.skip_out256.shape)
            x, _ = self.featurizer.down_tr512(x)
        else:
            x = self.featurizer.block_modules(x)
        ic(x.size())

        if hasattr(self.featurizer, 'attention_module'):
            feats = self.featurizer.attention_module.conv(x)
            ic(feats.size(), self.shap_priors.size())
            out = feats * self.shap_priors
            return out if not attention else (self.shap_priors, out)
        else:
            self.att = None
            out = x
            return out


if __name__ == '__main__':
    sss = datetime.now()
    print('threads: ', torch.get_num_threads())
    # torch.set_num_threads(1)
    # print('threads: ', torch.get_num_threads())
    args = get_args()
    set_random_seed(args.seed)
    ic(args)
    s = print_args(args, [])
    args.num_classes = 2 if len(args.cohort.split('_')[1:]) == 2 else 3
    ic(args.num_classes)
    # exit()
    if args.wandb:
        wandb.init(project='DeepDG', name=args.save_path, settings=wandb.Settings(start_method='fork'))
        wandb.config.update({
             k: v for k, v in vars(args).items() if (isinstance(v, str) or isinstance(v, int) or isinstance(v, float))
            }, allow_val_change=True)
        wandb.run.log_code('..')
        print('Started wandb logging....')
    

    loss_list = alg_loss_dict(args)
    print('loss list: ', loss_list)
    train_loaders, eval_loaders, test_loaders = get_img_dataloader(args)
    ic('loaded data.')
    eval_name_dict = train_valid_target_eval_names(args)
    ic('eval name dict: ', eval_name_dict)

    if not args.eval:
        with open(f'{args.save_path}/config.json', 'w') as f:
            json.dump(vars(args), f, indent=4)

    args.weights, args.counts = [], []
    for ldr in train_loaders:
        weights, count = ldr.dataset.get_sample_weights()
        args.weights.append(weights)
        args.counts.append(count)
    print(args.counts)

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id is not None else 'cpu')
    algorithm = algorithm_class(args).to(device)
    algorithm.train()
    ic()
    ic(args.num_classes)

    # if args.net == 'unet3d':
    #     algorithm.network = algorithm.network.half()

    if args.replace_att:
        priors_path = os.path.join(f'/data_1/dlteif/shap_maps/9x11x9/ERM/{args.net}_{args.classifier}_attention_{args.cohort}aug_minmax_normalize_lr0.0008_{args.fold}', args.cohort, args.layer)
        ic(priors_path)
        classes = ['NC','AD'] if args.num_classes == 2 else ['NC', 'MCI', 'AD']

        shap_priors = []
        for i, cls in enumerate(classes):
            assert os.path.exists(f'{priors_path}/{cls}_avg.npy')
            shap_priors.append(torch.from_numpy(np.load(f'{priors_path}/{cls}_avg.npy')[i]))


        shap_priors = torch.stack(shap_priors).float()
        ic(shap_priors.shape)

        algorithm.network[0] = ModifiedFeaturizer(algorithm.network[0], shap_priors.to(device))

    if args.swad:
        swad_algorithm = alg.swa_utils.AveragedModel(algorithm)
        swad_cls = alg.get_algorithm_class(args.swad)
        swad = swad_cls(args.n_converge, args.n_tolerance, args.tolerance_ratio)

    opt = get_optimizer(algorithm, args)
    sch = get_scheduler(opt, args)
        
        # algorithm.load_checkpoint(args.resume)

    print('=======hyper-parameter used========')
    print(s)
    acc_record = {}
    loss_record = {}
        
    best_epoch  = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        try:
            ic(1)
            state_dict = ckpt['model_dict']
        except:
            try:
                ic(2)
                state_dict = ckpt['state_dict']
            except:
                ic(3)
                state_dict = ckpt
        ic(state_dict.keys())
        # exit()
        # try:
        #     del state_dict['criterion.weight']
        # except:
        #     pass
        try:
            algorithm.load_state_dict(state_dict)
        except:
            if args.swad:
                swad_algorithm.load_state_dict(state_dict)
                algorithm = swad_algorithm.module
                state_dict['model_dict'] = algorithm.state_dict
                save_checkpoint(f'{args.save_path}/model_swad_final', state_dict, is_best=False)
                exit()
        
        print('loaded checkpoint at ', args.resume)
            
        try:
            best_epoch = ckpt['epoch']
            best_valid_acc = ckpt['accuracy']
            print(f'epoch: {best_epoch}, best acc: {best_valid_acc}')
        except:
            pass
    
    if args.eval:
        acc_type_list = ['valid','target']
        s = ''
        print('acc type list: ', acc_type_list)
        print('eval loaders: ', eval_name_dict['valid'])
        print('target loaders: ', eval_name_dict['target'])
        for item in acc_type_list:
            if item == 'valid':
                loaders = eval_loaders
            else:
                loaders = test_loaders
            acc_loss = [modelopera.accuracy(
                algorithm, ldr, mode=item, save_path=args.save_path if not args.resume else os.path.dirname(args.resume), num_classes=args.num_classes, write_raw_score=args.write_raw_score, save_embedding=args.save_emb) for ldr in loaders]
            acc_record[item] = np.mean(np.array([tup[0] for tup in acc_loss]))
            loss_record[item] = np.mean(np.array([tup[1] for tup in acc_loss]))
            s += (item+'_acc:%.4f,' % acc_record[item])
        print(s[:-1])
        print('total cost time:%s\n' % (str(datetime.now()-sss)))
        exit()
    
    
    
    acc_type_list = ['valid']
    train_minibatches_iterator = zip(*train_loaders)
    print('train loaders: ', len(train_loaders))
    # print('train_minibatches_iterator: ', len(list(train_minibatches_iterator)))
    best_valid_acc, best_valid_loss, target_acc = 0, 10000000, 0
    print('===========start training===========')
    # sss = time.time()

    if not args.start_epoch is None:
        best_epoch = args.start_epoch

    args.steps_per_epoch *= args.accum_iter
    ic(args.steps_per_epoch)
    scaler = GradScaler()
    for epoch in tqdm(range(best_epoch, args.max_epoch)):
        print(f'===========epoch {epoch}=============')
        for iter_num in tqdm(range(args.steps_per_epoch)):
            minibatches_device = [(data)
                                for data in next(train_minibatches_iterator)]
            print(len(minibatches_device))
            if args.algorithm == 'VREx' and algorithm.update_count == args.anneal_iters:
                opt = get_optimizer(algorithm, args)
                sch = get_scheduler(opt, args)
            step_vals, sch = algorithm.update(minibatches_device, opt, sch, scaler)

            del minibatches_device
        
        if args.swad:
            swad_algorithm.update_parameters(algorithm, step=epoch)

        if args.wandb:
            print('started wandb logging...')
            log_dict = {
                    'lr': opt.param_groups[0]['lr'],
                }
            wandb.log(log_dict, step=epoch)
            print('logging done...')



        if (epoch in [int(args.max_epoch*0.7), int(args.max_epoch*0.9)]) and (not args.schuse):
            print('manually descrease lr')
            for params in opt.param_groups:
                params['lr'] = params['lr']*0.1

        if (epoch == (args.max_epoch-1)) or (epoch % args.checkpoint_freq == 0):
            print('===========epoch %d===========' % (epoch))
            s = ''
            for item in loss_list:
                s += (item+'_loss:%.4f,' % step_vals[item])
            print(s[:-1])
            print('-----------------')
            torch.cuda.empty_cache()
            s = ''
            print('acc type list: ', acc_type_list)
            for item in acc_type_list:
                acc_loss = [modelopera.accuracy(
                    algorithm, ldr, mode=item, epoch=epoch, num_classes=args.num_classes) for ldr in eval_loaders]
                acc_record[item] = np.mean(np.array([tup[0] for tup in acc_loss]))
                loss_record[item] = np.mean(np.array([tup[1] for tup in acc_loss]))
                s += (item+'_acc:%.4f,' % acc_record[item])
            print(s[:-1])
            
            if args.save_model_every_checkpoint:
                save_dict = {
                    "epoch": epoch,
                    "args": vars(args),
                    "accuracy": acc_record['valid'],
                    "loss": loss_record['valid'],
                    "model_dict": algorithm.state_dict()
                }
                save_checkpoint(f'{args.save_path}/model_ep{epoch}', save_dict, is_best=False)
                
            
                if args.swad:
                    def prt_results_fn(results, avgmodel):
                        step_str = f'[{avgmodel.start_step}-{avgmodel.end_step}]'
                        row = to_row([val for key,val in acc_record.item()] + [val for key,val in loss_record.items()])
                        print(row, step_str)

                    swad.update_and_evaluate(swad_algorithm, acc_record['valid'], loss_record['valid'], prt_results_fn)
                    swad_algorithm = swad.get_final_model()
                    if not args.freeze_bn:
                        # update SWAD BN statistics for {n_steps} steps
                        # call evaluate on swad_algorithm
                        n_steps = 500
                        alg.swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, n_steps)
                        
                    print('Evaluating SWAD algorithm final model....')
                    s = ''
                    print('acc type list: ', acc_type_list)
                    for item in acc_type_list:
                        acc_loss = [modelopera.accuracy(
                            swad_algorithm, ldr, mode=item, epoch=epoch, num_classes=args.num_classes) for ldr in eval_loaders]
                        acc_record[item] = np.mean(np.array([tup[0] for tup in acc_loss]))
                        loss_record[item] = np.mean(np.array([tup[1] for tup in acc_loss]))
                        s += (item+'_acc:%.4f,' % acc_record[item])
                    print(s[:-1])   

                    save_dict = {
                    "epoch": epoch,
                    "args": vars(args),
                    "accuracy": acc_record['valid'],
                    "loss": loss_record['valid'],
                    "model_dict": swad_algorithm.module.state_dict()
                    }
                    save_checkpoint(f'{args.save_path}/model_swad_final', save_dict, is_best=False)
                else:
                    save_dict = {
                    "epoch": epoch,
                    "args": vars(args),
                    "accuracy": best_valid_acc,
                    "loss":  loss_record['valid'],
                    "model_dict": algorithm.state_dict()
                    }
                    save_checkpoint(f'{args.save_path}/model', save_dict, is_best=loss_record['valid'] < best_valid_loss)
                    if acc_record['valid'] > best_valid_acc:
                        best_valid_acc = acc_record['valid']
                    if loss_record['valid'] < best_valid_loss:
                        best_valid_loss = loss_record['valid']
                
            if args.wandb:
                print('started wandb logging...')
                log_dict = {
                    'best_acc': best_valid_acc,
                    'best_loss': best_valid_loss,
                    'valid_acc': acc_record['valid'],
                    'valid_loss': loss_record['valid'],
                } 
                for item in loss_list:
                    log_dict[f'{item}_loss'] = step_vals[item]
                wandb.log(log_dict, step=epoch)
                print('logging done.')

            # print('total cost time:%s\n' % (str(datetime.now()-sss)))

    
        # algorithm = algorithm.cuda()
        print('valid acc: %.4f' % best_valid_acc)
        # print('DG result: %.4f' % target_acc)

        with open(f'{args.save_path}/done.txt', 'w') as f:
            f.write(f'Trained for {epoch+1} epochs\n')
            f.write('total cost time:%s\n' % (str(datetime.now()-sss)))
            f.write('valid acc:%.4f\n' % (best_valid_acc))
            # f.write('target acc:%.4f' % (target_acc))
