# pylint: disable=W0221,E1101
import torch.nn as nn
from models.layers.verbose_gradients import VerboseGradients
from models.layers.balance_labels import BalanceLabels
from models.layers.utils import gtmat
from models.criteria.utils import unroll_time, winsmooth
from models.criteria.criterion import Criterion


class crossentropycriterion(Criterion):
    def __init__(self, args):
        super(crossentropycriterion, self).__init__(args)
        self.loss  = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        #import ipdb;ipdb.set_trace()
        loss = self.loss(pred, target)
        return  loss
