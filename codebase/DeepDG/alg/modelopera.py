# coding=utf-8
import os
import sys
sys.path.append('..')
from feature_extractor.for_image_data.backbone import ResNet3D
import torch
from network import img_network
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import numpy as np
import wandb
from datautil.imgdata.util import AverageMeter
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt 

def get_fea(args):
    if args.dataset == 'dg5':
        net = img_network.DTNBase()
    elif args.net.startswith('res'):
        net = img_network.ResBase(args.net, num_classes=args.num_classes, attention=args.attention)
    elif 'cnn' in args.net:
        net = img_network.CNNBase(args.net, num_classes=args.num_classes, attention=args.attention)
    elif args.net == 'unet3d':
        net = img_network.UNet3DBase(n_class=args.num_classes, attention=args.attention, pretrained=args.pretrained, blocks=args.blocks)
    else:
        net = img_network.VGGBase(args.net)
    return net

def write_scores(f, preds, labels):
    preds = preds.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    for index, pred in enumerate(preds):
        label = str(labels[index])
        pred = "__".join(map(str, list(pred)))
        f.write(pred + '__' + label + '\n')


def accuracy(network, loader, wandb_log=False, mode='valid', epoch=0, save_path=None, num_classes=2, write_raw_score=False, save_embedding=False):
    print('---------------def accuracy----------------')
    correct = 0
    total = 0
    print(loader.dataset.domain_name, loader.dataset.filename)
    confusion_matrices = []
    losses = AverageMeter()
    network.eval()
    true = []
    pred = []
    with torch.no_grad():
        for data in tqdm(loader):
            fnames = data[0]
            ic(len(fnames), fnames[0])
            x = data[1].float()
            y = data[2].long()
            print(x.size(), y.size())
            device = list(network.parameters())[0].device
            x = x.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast():
                p = network.predict(x)
                p = F.softmax(p)
                if write_raw_score:
                    with open(f'{save_path}/{loader.dataset.domain_name}_raw_scores.txt', 'a') as myfile:
                        write_scores(myfile, p, y)

                pred_labels = p.argmax(1)
                # print('pred labels: ', pred_labels.size())
                if p.size(1) == 1:
                    correct += (p.gt(0).eq(y).float()).sum().item()
                else:
                    correct += (pred_labels.eq(y).float()).sum().item()
                total += len(x)
                
                if hasattr(network,'criterion'):
                    print('output: ', p)
                    # probs = F.softmax(p, dim=1)
                    # print('probs: ', probs)
                    loss = network.criterion(p,y)
                    losses.update(loss.item(), x.size(0))

            cm = confusion_matrix(y.cpu().numpy(), pred_labels.cpu(), labels=range(num_classes))
            assert cm.shape == (num_classes,num_classes)
            confusion_matrices.append(cm[None,...])
            true.extend(y.cpu().numpy())
            pred.extend(pred_labels.cpu())
            # break

        print('confusion matrices: ', len(confusion_matrices))

        # for i in range(len(confusion_matrices)):
        #     print(i, confusion_matrices[i].shape)

        cm = np.squeeze(np.stack(confusion_matrices, axis=0).sum(axis=0))

        print('final confusion matrix: ', cm.shape)
        print(cm)

        if mode == 'target':
            ax = sns.heatmap(cm, annot=True, fmt='g')
            ax.set_title('Confusion matrix')
            cls_labels = ['NC', 'MCI', 'AD'] if num_classes == 3 else ['NC', 'AD']
            ax.set_xticklabels(cls_labels)
            ax.set_yticklabels(cls_labels)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            # plt.show()
            ic(save_path)
            plt.savefig(f'{save_path}/confusion_matrix_{loader.dataset.domain_name}.png', dpi=150)
            plt.close()

        if num_classes == 2:
            TN, FP, FN, TP = np.ravel(cm)
        else:
            FP = cm.sum(axis=0) - np.diag(cm) 
            FN = cm.sum(axis=1) - np.diag(cm)
            TP = np.diag(cm)
            TN = cm.sum() - (FP + FN + TP)
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN + sys.float_info.epsilon)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PREC = TP/(TP+FP + sys.float_info.epsilon)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        # Overall accuracy for each class
        ACC = (TP+TN)/(TP+FP+FN+TN)
        TOTAL_ACC = np.diag(cm).sum() / cm.sum()
        if num_classes > 2:
            CLS_BALANCED_ACC = TPR.sum() / num_classes
        else:
            CLS_BALANCED_ACC = (TPR + TNR) / 2

        F1 = 2 * PREC * TPR / (PREC + TPR + sys.float_info.epsilon)
        MACRO_F1 = F1.sum() / num_classes
        WEIGHTED_F1 = (F1 * cm.sum(axis=0)).sum() / cm.sum() 

        MCC = (TP*TN - FP*FN) / (np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)) + 0.000000001)
        MACRO_MCC = MCC.sum() / num_classes
        sk_MCC = matthews_corrcoef(np.asarray(true), np.asarray(pred))
        WEIGHTED_MCC = (MCC * cm.sum(axis=0)).sum() / cm.sum()
        print('TPR: ', TPR)
        print('FPR: ', FPR)
        print('FNR: ', FNR)
        print('TNR: ', TNR)
        print('PREC: ', PREC)
        print('ACC: ', ACC)
        print('F1 : ', F1)
        print('MCC : ', MCC)
        print('MACRO F1 : ', MACRO_F1)
        print('MACRO MCC : ', MACRO_MCC)
        print('SKLEARN MCC : ', sk_MCC)
        print('WEIGHTED F1 : ', WEIGHTED_F1)
        print('WEIGHTED MCC : ', WEIGHTED_MCC)

        print('TOTAL ACC: ', TOTAL_ACC)
        print('CLASS BALANCED ACC: ', CLS_BALANCED_ACC)

    network.train()
    if num_classes > 2:
        print(f"{','.join(map(str,ACC))},{','.join(map(str,TPR))},{','.join(map(str,TNR))},{','.join(map(str,F1))},{','.join(map(str,MCC))},{MACRO_F1},{MACRO_MCC},{WEIGHTED_F1},{WEIGHTED_MCC},{TOTAL_ACC},{CLS_BALANCED_ACC}")
    else:
        print(f"{ACC},{TPR},{TNR},{F1},{MCC},{MACRO_F1},{sk_MCC},{WEIGHTED_F1},{WEIGHTED_MCC},{TOTAL_ACC},{CLS_BALANCED_ACC}")

    return CLS_BALANCED_ACC, losses.avg
    # return correct / total, losses.avg
