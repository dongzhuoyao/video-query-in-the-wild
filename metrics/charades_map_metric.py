from metrics.utils import AverageMeter, charades_map
from metrics.metric import Metric
import numpy as np
from .multilabelMetrics.labelbasedclassification import accuracyMacro, accuracyMicro, precisionMacro, precisionMicro, recallMacro, recallMicro, fbetaMacro, fbetaMicro

class CharadesMAPMetric(Metric):
    def __init__(self):
        self.am = AverageMeter()
        self.predictions = []
        self.targets = []
        self.shit_threshold = 0.1

    def update(self, prediction, target):
        #assert target.dim() == 2 and prediction.dim()==2
        #import ipdb;ipdb.set_trace()
        self.targets.append(target)
        self.predictions.append(prediction)

    def __repr__(self):
        return 'charades map: {}'.format(self.compute()[1])


    def compute(self):
        #import ipdb;ipdb.set_trace()
        total_pred = np.vstack(self.predictions)
        total_gt =  np.vstack(self.targets)

        threshholded_pred = (total_pred > self.shit_threshold)

        mAP, _, ap = charades_map(total_pred,total_gt)
        #print(ap)
        _accuracyMacro = accuracyMacro(total_gt, threshholded_pred)
        _accuracyMicro = accuracyMicro(total_gt, threshholded_pred)
        _precisionMacro = precisionMacro(total_gt, threshholded_pred)
        _precisionMicro = precisionMicro(total_gt, threshholded_pred)
        _recallMacro = recallMacro(total_gt, threshholded_pred)
        _recallMicro = recallMicro(total_gt, threshholded_pred)
        _fbetaMacro = fbetaMacro(total_gt, threshholded_pred, beta=1)
        _fbetaMicro = fbetaMicro(total_gt, threshholded_pred, beta=1)

        return dict(map=mAP,
                    accuracyMacro = _accuracyMacro,
                    accuracyMicro = _accuracyMicro,
                    precisionMacro = _precisionMacro,
                    precisionMicro = _precisionMicro,
                    recallMacro = _recallMacro,
                    recallMicro = _recallMicro,
                    fbetaMacro = _fbetaMacro,
                    fbetaMicro = _fbetaMicro
                    )

