# coding=utf-8
import os
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader

import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from pl_datautil.mydataloader import DistributedWeightedSampler, DynamicBalanceClassSampler
from pl_datautil.mydataloader import InfiniteDataLoader as pl_InfiniteDataLoader
from datautil.mydataloader import InfiniteDataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler



def get_img_dataloader(args):
    print('-------------------def get_img_dataloader-------------------')
    rate = 0.2
    trdatalist, valdatalist, tedatalist = [], [], []
    if args.eval:
        tr_mmap, val_mmap = False, False
        te_mmap = False
    else:
        tr_mmap, val_mmap = False, False
        te_mmap = False
    print(args.img_dataset)
    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    prefix = '/projectnb/ivc-ml/dlteif/' if 'scc' in os.uname().nodename else '/home/diala/csv_files/' if os.uname().nodename == 'ayan' else '/home/dlteif/csv_files/'
    for i in range(len(names)):
        print(f'({i}) name: ', names[i], 'test envs: ', args.test_envs)
        filename = None
        if args.task == 'mri_dg':
            if names[i] == 'NACC':
                filename= prefix + 'NACC_NC_MCI_AD.csv'
            elif names[i] == 'ADNI':
                filename= prefix + 'ADNI_NC_MCI_AD.csv'
            elif names[i] == 'AIBL':
                filename= prefix + 'AIBL_NC_MCI_AD.csv'
            elif names[i] == 'FHS':
                filename= prefix + 'FHS_NC_MCI_AD.csv'
        if i in args.test_envs:
            print('in test_envs')
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_test(args.dataset), test_envs=args.test_envs, filename=filename, num_classes=args.num_classes, mmap=te_mmap))
        else:
            print('not in test_envs')
            root_dir = f'{prefix}{names[i]}_NC_MCI_AD_folds/fold_{args.fold}'
            filename = os.path.join(root_dir, 'train.csv')
            # tmpdatay = ImageDataset(args.dataset, args.task, args.data_dir,
            #                         names[i], i, transform=imgutil.image_train(args.dataset), test_envs=args.test_envs, filename=filename).labels
            # l = len(tmpdatay)
            # if args.split_style == 'strat':
            #     lslist = np.arange(l)
            #     stsplit = ms.StratifiedShuffleSplit(
            #         2, test_size=rate, train_size=1-rate, random_state=args.seed)
            #     stsplit.get_n_splits(lslist, tmpdatay)
            #     indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            # else:
            #     indexall = np.arange(l)
            #     np.random.seed(args.seed)
            #     np.random.shuffle(indexall)
            #     ted = int(l*rate)
            #     indextr, indexte = indexall[:-ted], indexall[-ted:]

            trdatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_train(args.dataset), indices=None, test_envs=args.test_envs, filename=filename, num_classes=args.num_classes, mmap=tr_mmap))
            f = os.path.join(root_dir, 'val.csv')
            valdatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                    names[i], i, transform=imgutil.image_test(args.dataset), indices=None, test_envs=args.test_envs, filename=f, num_classes=args.num_classes, mmap=val_mmap))
            f = os.path.join(root_dir, 'test.csv')
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                    names[i], i, transform=imgutil.image_test(args.dataset), indices=None, test_envs=args.test_envs, filename=f, num_classes=args.num_classes, mmap=te_mmap))

    if args.gpu_ids is not None and len(args.gpu_ids) > 1:
        train_loaders = {env.domain_name: pl_InfiniteDataLoader(
            dataset=env,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.N_WORKERS)
            for env in trdatalist}
        print('train loaders: ', len(train_loaders))
        for name, loader in train_loaders.items():
            print(f'train loader: ', name, loader.dataset.filename)
    else:
        train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env.get_sample_weights()[0],
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]
        print('train loaders: ', len(train_loaders))
        for idx, loader in enumerate(train_loaders):
            print(f'train loader: ', idx, loader.dataset.filename)

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in valdatalist]

    test_loaders = [DataLoader(
        dataset=env,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in tedatalist]

    

    print('eval loaders: ', len(eval_loaders))
    for i, loader in enumerate(eval_loaders):
        print(f'({i}) eval loader: ', loader.dataset.domain_name, loader.dataset.filename)
    
    print('test loaders: ', len(test_loaders))
    for i, loader in enumerate(test_loaders):
        print(f'({i}) test loader: ', loader.dataset.domain_name, loader.dataset.filename)
    
    
    return train_loaders, eval_loaders, test_loaders
