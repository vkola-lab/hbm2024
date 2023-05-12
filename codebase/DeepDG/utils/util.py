# coding=utf-8
import random
import numpy as np
import torch
import sys
import os
import torchvision
import PIL


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(filename, save_dict, is_best=False):
    torch.save(save_dict, filename + '.pkl')
    if is_best:
        torch.save(save_dict, filename + '_best.pkl')



def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    t = 0
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append(t)
            t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['valid'].append(t)
        else:
            eval_name_dict['target'].append(t)
        t += 1
    return eval_name_dict


def alg_loss_dict(args):
    loss_dict = {'ANDMask': ['total'],
                 'CORAL': ['class', 'coral', 'total'],
                 'DANN': ['class', 'dis', 'total'],
                 'ERM': ['class'],
                 'Mixup': ['class'],
                 'OrgMixup': ['class'],
                 'MLDG': ['inner', 'class', 'total'],
                 'MMD': ['class', 'mmd', 'total'],
                 'GroupDRO': ['group'],
                 'RSC': ['class'],
                 'VREx': ['loss', 'nll', 'penalty'],
                 'XAIalign': ['class', 'align', 'total'],
                 'LossValley': ['class', 'total'],
                 'Vanilla': ['class'],
                 }
    return loss_dict[args.algorithm]


def print_args(args, print_list):
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s

def to_row(row, colwidth=10, latex=False):
    """Convert value list to row string"""
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.6f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    return sep.join([format_val(x) for x in row]) + " " + end_


def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def img_param_init(args):
    dataset = args.dataset
    if dataset == 'office':
        domains = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-caltech':
        domains = ['amazon', 'dslr', 'webcam', 'caltech']
    elif dataset == 'office-home':
        domains = ['Art', 'Clipart', 'Product', 'Real_World']
    elif dataset == 'dg5':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
    elif dataset == 'PACS':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset == 'VLCS':
        domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    elif dataset == 'MRI':
        domains = ['NACC', 'ADNI', 'AIBL', 'FHS']
    else:
        print('No such dataset exists!')
    args.domains = domains
    args.img_dataset = {
        'office': ['amazon', 'dslr', 'webcam'],
        'office-caltech': ['amazon', 'dslr', 'webcam', 'caltech'],
        'office-home': ['Art', 'Clipart', 'Product', 'Real_World'],
        'PACS': ['art_painting', 'cartoon', 'photo', 'sketch'],
        'dg5': ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'],
        'VLCS': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],
        'MRI': ['NACC', 'ADNI', 'AIBL', 'FHS']
    }
    if dataset == 'dg5':
        args.input_shape = (3, 32, 32)
        args.num_classes = 10
    else:
        args.input_shape = (3, 224, 224)
        if args.dataset == 'office-home':
            args.num_classes = 65
        elif args.dataset == 'office':
            args.num_classes = 31
        elif args.dataset == 'PACS':
            args.num_classes = 7
        elif args.dataset == 'VLCS':
            args.num_classes = 5
        elif args.dataset == 'MRI':
            args.num_classes = 3
            args.input_shape = (1,182,218,182)
    return args

#######################################################
############ Distributed Training Utils ###############

def get_num_gpus():
    """Number of GPUs on this node."""
    return torch.cuda.device_count()


def get_local_rank():
    if 'MV2_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_RANK'])
    elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    elif 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    else:
        return 0

def get_local_size():
    if 'MV2_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_LOCAL_SIZE'])
    elif 'OMPI_COMM_WORLD_LOCAL_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
    elif 'SLURM_NTASKS_PER_NODE' in os.environ:
        return int(os.environ['SLURM_NTASKS_PER_NODE'])
    else:
        return 1

def get_world_rank():
    if 'MV2_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_RANK'])
    elif 'OMPI_COMM_WORLD_RANK' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        return int(os.environ['SLURM_PROCID'])
    else:
        return 0

def get_world_size():
    if 'MV2_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['MV2_COMM_WORLD_SIZE'])
    elif 'OMPI_COMM_WORLD_SIZE' in os.environ:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])
    elif 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS'])
    else:
        return 1


def initialize_dist(init_file):
    """Initialize PyTorch distributed backend."""
    torch.cuda.init()
    torch.cuda.set_device(get_local_rank())
    init_file = os.path.abspath(init_file)
    # init_file = os.path.basename(init_file)
    print(init_file)
    torch.distributed.init_process_group(
        backend='nccl', init_method=f'file://{init_file}',
        rank=get_world_rank(), world_size=get_world_size())
    # torch.distributed.init_process_group(
    #     backend='nccl',
    #     rank=get_world_rank(), world_size=get_world_size())
    torch.distributed.barrier()
    # Ensure the init file is removed.
    if get_world_rank() == 0 and os.path.exists(init_file):
        os.unlink(init_file)

def get_cuda_device():
    return torch.device(f'cuda:{get_local_rank()}')

def allreduce_tensor(t):
    rt = t.clone().detach()
    torch.distributed.all_reduce(rt)
    rt /= get_world_size()
    return rt

