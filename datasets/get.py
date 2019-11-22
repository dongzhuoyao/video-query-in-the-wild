""" Initilize the datasets module
    New datasets can be added with python scripts under datasets/
"""
import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate
import torch.utils.data.distributed
import importlib
import collections


def case_getattr(obj, attr):
    casemap = {}
    for x in obj.__dict__:
        casemap[x.lower()] = x
    return getattr(obj, casemap[attr.replace('_', '')])


def my_collate(batch):
    if isinstance(batch[0], collections.Mapping) and 'do_not_collate' in batch[0]:
        return batch
    if isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [my_collate(samples) for samples in transposed]
    else:
        return default_collate(batch)



def get_dataset(args):

    dataset = args.dataset
    if isinstance(dataset, str):
        obj = importlib.import_module('.' + dataset, package='datasets')
        dataset = case_getattr(obj, dataset)
    train_dataset = dataset.get(args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=my_collate, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=False)

    return train_loader
