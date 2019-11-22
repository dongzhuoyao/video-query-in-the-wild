"""
   Normalize gradient of multiple inputs to have the same norm as the gradient of the first input
"""
from torch.autograd import Function, Variable
from pytorchgo.utils import logger

VERBOSE = True


def dprint(message, *args):
    if VERBOSE:
        print(message.format(*args))


class VerboseGradients(Function):
    @staticmethod
    def forward(ctx, *inputs):
        output = [x.clone() for x in inputs]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grad_outputs):
        norms = [x.data.norm() for x in grad_outputs]
        #logger.info('gradnorms: {}'.format(' \t'.join([str(x) for x in norms])))
        return tuple(grad_outputs)
