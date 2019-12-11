import torch.backends.cudnn as cudnn
from models.get import get_model
from datasets.get import get_dataset
from misc_utils.utils import seed
import pdb
import numpy as np
from bdb import BdbQuit
import traceback
from sklearn import metrics as sklearn_metrics
import sys, os
from pytorchgo_logger as logger
from pytorchgo.utils.pytorch_utils import model_summary, optimizer_summary

from pytorchgo.utils.pytorch_utils import set_gpu
# pytorch bugfixes
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


logger.auto_set_dir()

import torch
import itertools
from misc_utils.utils import AverageMeter, Timer
from tqdm import tqdm
import argparse
import os

short_train_num = 1
novel_img_num = 5
input_size = 112
nclass = 200

arch = 'resnet18_3d_f2f'
cur_dataset = "arv120_20_60_triplet_clsrank_fs"
from retrievel_evaluation1202060 import ARV_Retrieval_Clip,ARV_Retrieval_Moment,ARV_Retrieval
cur_criterion = 'crossentropy_criterion'

init_lr = 1e-4
eval_per = 15
lr_decay_rate = '90'
epochs = 150
batch_size = 4*4
test_batch_size = 12*3
triplet_margin = 1
pretrained = True
eval_split = 'testing'
train_frame=32
no_novel = False
dropout = 0.5
temporal_stride = 1
clip_sec = 6

metric_feat_dim = 512



def parse():
    print('parsing arguments')
    parser = argparse.ArgumentParser(description='PyVideoResearch')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate on validation sets')
    # Data parameters
    parser.add_argument('--dataset', default=cur_dataset, help='name of dataset under datasets/')
    # Model parameters
    parser.add_argument('--arch', '-a', default=arch, help='model architecture: ')
    parser.add_argument('--input_size', default=input_size, type=int)
    parser.add_argument('--dropout', default=dropout, type=float, help='[0-1], 0 = leave defaults')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--pretrained_weights', default='')
    parser.add_argument('--nclass', default=nclass, type=int)
    parser.add_argument('--criterion', default=cur_criterion, help=' ''default_criterion'' for sigmoid loss')
    parser.add_argument('--features', default='fc', help='conv1;layer1;layer2;layer3;layer4;fc')

    # System parameters
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--print_freq', '-p', default=50, type=int, help='print frequency (default: 10)')
    parser.add_argument('--manual_seed', default=0, type=int)

    parser.add_argument('--wrapper', default='default_wrapper',
                        help='child of nn.Module that wraps the base arch. ''default_wrapper'' for no wrapper')

    # Training parameters
    parser.add_argument('--optimizer', default='adam', type=str, help='sgd | adam')
    parser.add_argument('--epochs', default=epochs, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=batch_size, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--test_batch_size', default=test_batch_size, type=int, help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning_rate', default=init_lr, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', default=lr_decay_rate, type=str)
    parser.add_argument('--accum_grad', default=1, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--freeze_batchnorm', dest='freeze_batchnorm', action='store_true')
    parser.add_argument('--synchronous', dest='synchronous', action='store_true')
    parser.add_argument('--replace_last_layer', dest='freeze_batchnorm', default=True, action='store_true')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--test_load', type=str)#"'train_log/v2.arv.resnet18_3d_f2f.evaltest.nonovel.clsrank.bs12/best.pth.tar'
    parser.add_argument('--novel_img_num', default=novel_img_num, type=int)
    parser.add_argument('--triplet_margin', default=triplet_margin, type=float)
    parser.add_argument('--eval_split', default=eval_split, type=str)
    parser.add_argument('--train_frame', default=train_frame, type=int)
    parser.add_argument('--test_frame_num', default=train_frame, type=int)
    parser.add_argument('--no_novel', default=no_novel, action='store_true')
    parser.add_argument('--eval_untrimmed', action='store_true')
    parser.add_argument('--eval_clip', action='store_true')
    parser.add_argument('--temporal_stride', default=temporal_stride, type=int)
    parser.add_argument('--clip_sec', default=clip_sec, type=int)
    parser.add_argument('--metric_feat_dim', default=metric_feat_dim, type=int)
    parser.add_argument('--use_faiss', default=True, action='store_true')
    parser.add_argument('--read_cache_feat', default=False, action='store_true')
    parser.add_argument('--memory_leak_debug', default=False, action='store_true')
    parser.add_argument('--topk_per_video', default=None,  type=int)
    parser.add_argument('--percent_per_gallery', default=None, type=float)
    parser.add_argument('--triple_eval', action='store_true')








    args = parser.parse_args()

    args.replace_last_layer = True
    args.pretrained = pretrained

    return args


def adjust_learning_rate(startlr, decay_rate, optimizer, epoch):
    decay_rates = int(decay_rate)
    if epoch == decay_rates:
            for g_id, param_group in enumerate(optimizer.param_groups):
                origin_lr = param_group['lr']
                param_group['lr'] = origin_lr*0.1
                logger.warning('update optimizer group {} from lr = {} to {}'.format(g_id, origin_lr, param_group['lr']))

    show_lr=optimizer.param_groups[0]['lr']
    logger.warning("current lr={}".format(show_lr))
    return show_lr



def part(x, iter_size):
    n = int(len(x) * iter_size)
    if iter_size > 1.0:
        x = itertools.chain.from_iterable(itertools.repeat(x))
    return itertools.islice(x, n)


def train(loader, model, optimizer, epoch, args):
    timer = Timer()
    data_time = AverageMeter()
    losses = AverageMeter()
    triple_losses = AverageMeter()
    ce_losses = AverageMeter()

    # switch to train mode
    cur_lr = adjust_learning_rate(args.lr, args.lr_decay_rate, optimizer, epoch)
    model.train()
    optimizer.zero_grad()
    iter_size = 1.0
    import torch.nn as nn
    ce_loss_criterion = nn.CrossEntropyLoss()
    triple_loss_criterion = nn.TripletMarginLoss(margin = args.triplet_margin, p = 2)
    for i, (input, meta) in tqdm(enumerate(part(loader, iter_size)), desc='Train Epoch'):
        if short_train_num <= i and args.debug: break
        data_time.update(timer.thetime() - timer.end)

        _batch_size = len(meta)
        target = []
        for _ in range(_batch_size):
            target.extend(meta[_]['labels'])
        target = torch.from_numpy(np.array(target))
        target = target.long().cuda()
        input = input.view(_batch_size * 3, input.shape[2], input.shape[3], input.shape[4], input.shape[5])
        output, metric_feat = model(input, meta)
        ce_loss = ce_loss_criterion(output, target)
        metric_feat = metric_feat.mean(dim=2) #reduce temporal dimension
        metric_feat = metric_feat.view(_batch_size, 3, metric_feat.shape[1])
        metric_feat = torch.nn.functional.normalize(metric_feat, dim=-1)
        #triple_loss = triple_loss_criterion(metric_feat[:, 0, :], metric_feat[:, 1, :], metric_feat[:, 2, :])
        loss = ce_loss #+ triple_loss

        loss.backward()
        losses.update(loss.item())
        ce_losses.update(ce_loss.item())
        #triple_losses.update(triple_loss.item())
        if i % args.accum_grad == args.accum_grad - 1:
            optimizer.step()
            optimizer.zero_grad()

        timer.tic()
        if i % args.print_freq == 0 and i>0:
            logger.info('[{0}][{1}/{2}({3})]\t'
                        'Time={timer.avg:.3f}\t'
                        'Dataload_Time={data_time.avg:.3f}\t'
                        'Loss={loss.avg:.4f}\t'
                        'CELoss={ce_loss.avg:.4f}\t'
                        'LR={cur_lr:.7f}\t'
                        'bestAP={ap:.3f}'.format(
                epoch, i, int(len(loader) * iter_size), len(loader), timer=timer,
                data_time=data_time, loss=losses, ce_loss=ce_losses,  ap=args.best_score,
                cur_lr=cur_lr))
            losses.reset()
            ce_losses.reset()
        del loss, output  # make sure we don't hold on to the graph


def val(args, model, triple_eval=False):
    model.eval()
    def metric_func(query, candidate):
        # numpy
        return sklearn_metrics.pairwise.cosine_similarity(query.reshape(1, -1), candidate.reshape(1, -1))
        # return np.linalg.norm(query-candidate)

    def feat_func(input):
        logits, metric_feat= model(input, {})  # [B,C]
        # metric_feat = F.normalize(metric_feat, p=2, dim=1)
        return metric_feat.data.cpu().numpy()


    if args.eval_clip:
        arv_retrieval = ARV_Retrieval_Clip(args=args, feat_extract_func=feat_func)
        score_dict = arv_retrieval.evaluation()
    elif args.eval_untrimmed:
        arv_retrieval = ARV_Retrieval_Moment(args=args, feat_extract_func=feat_func)
        score_dict = arv_retrieval.evaluation()
    else:
        if args.triple_eval:
            arv_retrieval = ARV_Retrieval(args=args, feat_extract_func=feat_func)
            score_dict = arv_retrieval.evaluation()
            arv_retrieval = ARV_Retrieval_Clip(args=args, feat_extract_func=feat_func)
            arv_retrieval.evaluation()
            arv_retrieval = ARV_Retrieval_Moment(args=args, feat_extract_func=feat_func)
            arv_retrieval.evaluation()
        else:
            arv_retrieval = ARV_Retrieval(args=args, feat_extract_func=feat_func)
            score_dict = arv_retrieval.evaluation()

    model.train()
    return score_dict


def main():
    args = parse()

    set_gpu(args.gpu)

    args.best_score = 0
    args.best_result_dict = {}

    seed(args.manual_seed)

    model = get_model(args)

    # define loss function
    from importlib import import_module
    from models.utils import case_getattr


    if args.evaluate:
        logger.info(vars(args))
        if args.test_load is not None:
            weight_path = args.test_load
        else:
            weight_path = os.path.join("train_log", os.path.basename(__file__).replace(".py", ""), "best.pth.tar")

        saved_dict = torch.load(weight_path)
        logger.warning("loading weight {}".format(weight_path))
        model.load_state_dict(saved_dict['state_dict'], strict=True)
        args.read_cache_feat = True
        score_dict = val(args=args, model=model)
        return

    logger.warning("using {}".format(args.optimizer))
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
    else:
        assert False, "invalid optimizer"

    model_summary(model)
    optimizer_summary(optimizer)

    train_loader = get_dataset(args)

    logger.info(vars(args))

    for epoch in range(args.epochs):

        train(train_loader, model, optimizer, epoch, args)

        if (epoch % eval_per == 0 and epoch > 0) or epoch == args.epochs - 1:
            score_dict = val(args=args, model=model)

            score = score_dict['ap']
            is_best = score > args.best_score
            if is_best:
                # args.best_result_dict = score_dict
                args.best_score = max(score, args.best_score)
                logger.warning("saving best snapshot..")
                torch.save({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'score': args.best_score,
                    # 'score_dict': args.best_result_dict,
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(logger.get_logger_dir(), "best.pth.tar"))

    weigth_path = os.path.join("train_log", os.path.basename(__file__).replace(".py", ""), "best.pth.tar")
    saved_dict = torch.load(weigth_path)
    logger.warning("loading weight {}".format(weigth_path))
    model.load_state_dict(saved_dict['state_dict'], strict=True)
    args.eval_split = 'testing'
    logger.info(vars(args))
    score_dict = val(args=args, model=model,triple_eval=True)


def pdbmain():
    try:
        main()
    except BdbQuit:
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print('shit always happens')
        pdb.post_mortem()
        sys.exit(1)


if __name__ == '__main__':
    #raise,"metric feat normalize twice?"
    main()