# coding=utf-8

import os
import sys
sys.path.append('/projectnb/ivc-ml/dlteif/transferlearning/codebase')
sys.path.append('/projectnb/ivc-ml/dlteif/transferlearning/codebase/DeepDG/alg')
import time
import numpy as np
import argparse
from tqdm import tqdm
import torch.distributed as dist

import alg
from alg.opt import *
from alg import alg, modelopera
from utils.util import save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, Tee, img_param_init, print_environ, to_row, \
                        get_world_rank, initialize_dist, get_cuda_device, get_local_rank, get_world_size
from train import get_args, ModifiedFeaturizer
import wandb
import icecream
from icecream import install
install()


from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint

# torch.set_float32_matmul_precision('high')

def cleanup():
    dist.destroy_process_group()

def main(world_size, args, use_cuda=False):
    # dist.init_process_group(backend='nccl')
    args = get_args()
    ic(args.save_path)
    
    wandb_logger = WandbLogger(project='DeepDG', log_model='all', name=args.save_path) if args.wandb else None
    loss_list = alg_loss_dict(args)
    print('loss list: ', loss_list)
    
    algorithm_class = alg.get_algorithm_class('pl_'+args.algorithm)
    algorithm = algorithm_class(args)

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

        algorithm.network[0] = ModifiedFeaturizer(algorithm.network[0], shap_priors)
   
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = ckpt
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

    # train_loaders, eval_loaders, test_loaders = 
    algorithm.get_dataloaders()
    # print('loaded data.')
    if args.eval:
        trainer = pl.Trainer(
        max_epochs=args.max_epoch,
        devices=[args.gpu_id] if args.gpu_ids is None else args.gpu_ids,
        num_nodes=1,
        accelerator='gpu',
        strategy=DDPStrategy(find_unused_parameters=False),
        logger=wandb_logger,
        default_root_dir=args.save_path,
        # fast_dev_run=True,
        )

        trainer.test(algorithm)
        exit()

    checkpoint_callback = ModelCheckpoint(monitor='mean_val_acc', mode='max')
    trainer = pl.Trainer(
        max_epochs=args.max_epoch,
        devices=args.gpus if args.gpu_ids is None else args.gpu_ids,
        accelerator='gpu',
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=wandb_logger,
        default_root_dir=args.save_path,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=args.accum_iter,
        precision=16,
        # replace_sampler_ddp=False,
        # max_steps=args.steps_per_epoch,
        # fast_dev_run=True,
    )

    algorithm.train()
    
    eval_name_dict = train_valid_target_eval_names(algorithm.args)
    print('eval name dict: ', eval_name_dict)

    trainer.fit(algorithm)
    exit()


if __name__ == '__main__':
    print('threads: ', torch.get_num_threads())
    # torch.set_num_threads(1)
    # print('threads: ', torch.get_num_threads())
    args = get_args()
    # set_random_seed(args.seed)
    seed_everything(args.seed)
    s = print_args(args, [])

    if args.wandb:
        wandb.init(project='DeepDG', name=args.save_path, settings=wandb.Settings(start_method='fork'))
        wandb.config.update({
             k: v for k, v in vars(args).items() if (isinstance(v, str) or isinstance(v, int) or isinstance(v, float))
            }, allow_val_change=True)
        wandb.run.log_code('..')

    world_size = len(args.gpu_ids)
    use_cuda = args.gpu_ids and torch.cuda.is_available()
    
    out_str = str(args)
    if get_world_rank() == 0:
        print(out_str)
        print("We have available ", torch.cuda.device_count(), "GPUs! but using ", world_size," GPUs")
    
    main(world_size, args, use_cuda=use_cuda)