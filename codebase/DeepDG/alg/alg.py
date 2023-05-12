# coding=utf-8
from alg.algs.ERM import ERM
from alg.algs.MMD import MMD
from alg.algs.CORAL import CORAL
from alg.algs.DANN import DANN
from alg.algs.RSC import RSC
from alg.algs.Mixup import Mixup, OrgMixup
from alg.algs.MLDG import MLDG
from alg.algs.GroupDRO import GroupDRO
from alg.algs.ANDMask import ANDMask
from alg.algs.VREx import VREx
from alg.algs.XAI_align import XAIalign
from alg.algs.vanilla import Vanilla
from alg.algs.SWAD import LossValley, IIDMax
from alg.algs import swa_utils
from alg.pl_algs.ERM import ERM as pl_ERM
from alg.pl_algs.XAI_align import XAIalign as pl_XAIalign

ALGORITHMS = [
    'ERM',
    'Mixup',
    'CORAL',
    'MMD',
    'DANN',
    'MLDG',
    'GroupDRO',
    'RSC',
    'ANDMask',
    'VREx',
    'XAIalign',
    'LossValley',
    'IIDMax',
    'Vanilla',
    'pl_ERM',
    'pl_XAIalign'
]


def get_algorithm_class(algorithm_name):
    ic(globals())
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
