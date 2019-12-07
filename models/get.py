"""
Initialize the model module
New models can be defined by adding scripts under models/
"""
from importlib import import_module
from models.utils import set_distributed_backend, replace_last_layer, generic_load, case_getattr
import torch.nn
from pytorchgo.utils import logger

def get_model(args):
    model = generic_load(args.arch, args.pretrained, args.pretrained_weights, args)
    if args.replace_last_layer:
        logger.warn('replacing last layer')
        model = replace_last_layer(model, args)
    for module in model.modules():
        if args.dropout != 0 and isinstance(module, torch.nn.modules.Dropout):
            logger.warn('setting Dropout p to {}'.format(args.dropout))
            module.p = args.dropout

    wrapper = case_getattr(import_module('models.wrappers.' + args.wrapper), args.wrapper)
    model = wrapper(model, args)
    model = set_distributed_backend(model, args)

    return model