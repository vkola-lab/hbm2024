# coding=utf-8
import pytorch_lightning as pl

class Algorithm(pl.LightningModule):

    def __init__(self, args):
        super(Algorithm, self).__init__()
        self.args = args
        self.save_hyperparameters()
        self.batch_size = args.batch_size

    def update(self, minibatches, opt, sch):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError
