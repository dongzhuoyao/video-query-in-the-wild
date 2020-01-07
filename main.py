from misc_utils.utils import seed
import pdb
import numpy as np
from bdb import BdbQuit
import traceback
import sys
from misc_utils import pytorchgo_logger as logger
from misc_utils.pytorch_util import model_summary, optimizer_summary, set_gpu

import cv2
import torch.nn as nn
import torch.nn.functional as F

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import torch
from misc_utils.utils import AverageMeter, Timer
from tqdm import tqdm
import argparse
import os

debug_short_train_num = 1
novel_num = 5
input_size = 112
nclass = 200

from dataloader_baseline import (
    ARV_Retrieval,
    ARV_Retrieval_Clip,
    ARV_Retrieval_Moment,
)

init_lr = 1e-4
eval_per = 15
lr_decay_rate = "90"
epochs = 150
batch_size = 10
test_batch_size = 10 * 3
triplet_margin = 1
pretrained = True
eval_split = "testing"
train_frame = 32
dropout = 0.5
temporal_stride = 1
clip_sec = 6
metric_feat_dim = 512
moving_average = 0.9


def parse():
    print("parsing arguments")
    parser = argparse.ArgumentParser(description="Video Retrieval In the Wild")
    parser.add_argument(
        "--method",
        default="baseline",
        choices=["baseline", "va", "vasa"],
        type=str,
    )
    parser.add_argument(
        "--meta_split",
        default="100_20_80",
        choices=["100_20_80", "120_20_60", "80_20_100"],
        type=str,
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="evaluate on validation sets"
    )
    # Model parameters
    parser.add_argument("--input_size", default=input_size, type=int)
    parser.add_argument(
        "--dropout",
        default=dropout,
        type=float,
        help="[0-1], 0 = leave defaults",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="use pre-trained model"
    )
    parser.add_argument("--pretrained_weights", default="")
    parser.add_argument("--nclass", default=nclass, type=int)
    parser.add_argument(
        "--features", default="fc", help="conv1;layer1;layer2;layer3;layer4;fc"
    )
    parser.add_argument(
        "--semantic_json",
        default="word_embed/wordembed_elmo_d1024.json",
        type=str,
    )

    # System parameters
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--print_freq",
        default=50,
        type=int,
        help="print frequency (default: 10)",
    )
    parser.add_argument("--manual_seed", default=0, type=int)
    parser.add_argument("--query_num", default=1, type=int)

    # Training parameters
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--optimizer", default="adam", type=str, help="sgd | adam"
    )
    parser.add_argument(
        "--epochs",
        default=epochs,
        type=int,
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch_size",
        default=batch_size,
        type=int,
        help="mini-batch size (default: 256)",
    )
    parser.add_argument(
        "--test_batch_size",
        default=test_batch_size,
        type=int,
        help="mini-batch size (default: 256)",
    )
    parser.add_argument(
        "--lr", default=init_lr, type=float, help="initial learning rate"
    )
    parser.add_argument("--lr_decay_rate", default=lr_decay_rate, type=str)
    parser.add_argument("--accum_grad", default=1, type=int)
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--wd", default=1e-5, type=float, help="weight decay")
    parser.add_argument("--test_load", type=str)
    parser.add_argument("--novel_num", default=novel_num, type=int)
    parser.add_argument("--triplet_margin", default=triplet_margin, type=float)
    parser.add_argument("--eval_split", default=eval_split, type=str)
    parser.add_argument("--train_frame", default=train_frame, type=int)
    parser.add_argument("--test_frame_num", default=train_frame, type=int)
    parser.add_argument("--temporal_stride", default=temporal_stride, type=int)
    parser.add_argument("--clip_sec", default=clip_sec, type=int)
    parser.add_argument("--metric_feat_dim", default=metric_feat_dim, type=int)
    parser.add_argument("--read_cache_feat", action="store_true")
    parser.add_argument("--memory_leak_debug", action="store_true")

    parser.add_argument("--eval_moment", action="store_true")
    parser.add_argument("--eval_clip", action="store_true")
    parser.add_argument("--eval_all", action="store_true")
    parser.add_argument("--log_action", default="n", type=str)
    parser.add_argument("--moving_average", default=moving_average, type=int)

    args = parser.parse_args()

    if args.debug:
        args.epochs = 2

    args.pretrained = pretrained
    args.logger_dir = "train_log/{}_{}_novel{}_mv{}".format(
        os.path.basename(__file__).replace(".py", ""),
        args.method,
        args.novel_num,
        args.moving_average,
    )
    logger.set_logger_dir(args.logger_dir, args.log_action)
    return args


def adjust_learning_rate(decay_rate, optimizer, epoch):
    decay_rates = int(decay_rate)
    if epoch == decay_rates:
        for g_id, param_group in enumerate(optimizer.param_groups):
            origin_lr = param_group["lr"]
            param_group["lr"] = origin_lr * 0.1
            logger.warning(
                "update optimizer group {} from lr = {} to {}".format(
                    g_id, origin_lr, param_group["lr"]
                )
            )

    show_lr = optimizer.param_groups[0]["lr"]
    logger.warning(
        "current lr={}, logger_dir={}".format(show_lr, logger.get_logger_dir())
    )
    return show_lr


def get_model(args):
    if args.method == "baseline":
        from models.resnet18_3d_f2f import ResNet3D, BasicBlock
    elif args.method == "va":
        from models.resnet18_va import ResNet3D, BasicBlock
    elif args.method == "vasa":
        from models.resnet18_vasa import ResNet3D, BasicBlock
    else:
        raise
    model = ResNet3D(
        args, BasicBlock, [2, 2, 2, 2], num_classes=args.nclass
    )  # 50
    if args.pretrained:
        print("loading pretrained weight.!")
        from torchvision.models.resnet import resnet18

        model2d = resnet18(pretrained=True)
        model.load_2d(model2d)
    from misc_utils.model_utils import set_distributed_backend

    model = set_distributed_backend(
        model, args
    )  # TODO, dongzhuoyao, why this sentence is so vital for speed up?
    return model


def do_eval(args, model):
    model.eval()

    def feat_func(input):
        if args.method == "baseline":
            metric_feat = model(input)  # [B,C,T]
        elif args.method == "va":
            metric_feat = model(input, None, None)  # [B,C,T]
        elif args.method == "vasa":
            metric_feat = model(input, None, None, None)  # [B,C,T]
        else:
            raise
        metric_feat = F.normalize(metric_feat, p=2, dim=1)  # normalize on C
        return metric_feat.data.cpu().numpy()

    if args.eval_clip:
        arv_retrieval = ARV_Retrieval_Clip(
            args=args, feat_extract_func=feat_func
        )
        score_dict = arv_retrieval.evaluation()
    elif args.eval_moment:
        arv_retrieval = ARV_Retrieval_Moment(
            args=args, feat_extract_func=feat_func
        )
        score_dict = arv_retrieval.evaluation()
    elif args.eval_all:
        arv_retrieval = ARV_Retrieval(args=args, feat_extract_func=feat_func)
        score_dict = arv_retrieval.evaluation()
        arv_retrieval = ARV_Retrieval_Clip(
            args=args, feat_extract_func=feat_func
        )
        arv_retrieval.evaluation()
        arv_retrieval = ARV_Retrieval_Moment(
            args=args, feat_extract_func=feat_func
        )
        arv_retrieval.evaluation()
    else:  # only evaluate trimmed retrieval by default
        arv_retrieval = ARV_Retrieval(args=args, feat_extract_func=feat_func)
        score_dict = arv_retrieval.evaluation()
    model.train()
    return score_dict


def train_ranking(loader, model, optimizer, epoch, args):
    timer = Timer()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()
    cur_lr = adjust_learning_rate(args.lr_decay_rate, optimizer, epoch)
    model.train()
    optimizer.zero_grad()
    ce_loss_criterion = nn.CrossEntropyLoss()
    for i, (input, meta) in tqdm(enumerate(loader), desc="Train Epoch"):
        if args.debug and i >= debug_short_train_num:
            break
        data_time.update(timer.thetime() - timer.end)

        _batch_size = len(meta)
        target = []
        for _ in range(_batch_size):
            target.extend(meta[_]["labels"])
        target = torch.from_numpy(np.array(target))
        input = input.view(
            _batch_size * 3,
            input.shape[2],
            input.shape[3],
            input.shape[4],
            input.shape[5],
        )
        metric_feat, cls_logits, reg_logits = model(
            input, target, temperature=0.1
        )
        ce_loss = ce_loss_criterion(cls_logits.cuda(), target.long().cuda())
        register_loss = ce_loss_criterion(
            reg_logits.cuda(), target.long().cuda()
        )
        loss = ce_loss + register_loss

        loss.backward()
        loss_meter.update(loss.item())
        ce_loss_meter.update(ce_loss.item())
        reg_loss_meter.update(register_loss.item())
        if i % args.accum_grad == args.accum_grad - 1:
            optimizer.step()
            optimizer.zero_grad()

        if i % args.print_freq == 0 and i > 0:
            logger.info(
                "[{0}][{1}/{2}({3})]\t"
                "Dataload_Time={data_time.avg:.3f}\t"
                "Loss={loss.avg:.4f}\t"
                "CELoss={ce_loss.avg:.4f}\t"
                "RegLoss={reg_loss.avg:.4f}\t"
                "LR={cur_lr:.7f}\t"
                "bestAP={ap:.3f}".format(
                    epoch,
                    i,
                    len(loader),
                    len(loader),
                    data_time=data_time,
                    loss=loss_meter,
                    ce_loss=ce_loss_meter,
                    reg_loss=reg_loss_meter,
                    ap=args.best_score,
                    cur_lr=cur_lr,
                )
            )
            loss_meter.reset()
            ce_loss_meter.reset()


def train_vasa(loader, model, optimizer, epoch, args):
    timer = Timer()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()
    word_loss_meter = AverageMeter()
    cur_lr = adjust_learning_rate(args.lr_decay_rate, optimizer, epoch)
    model.train()
    optimizer.zero_grad()
    ce_loss_criterion = nn.CrossEntropyLoss()
    for i, (input, meta) in tqdm(enumerate(loader), desc="Train Epoch"):
        if args.debug and i >= debug_short_train_num:
            break
        data_time.update(timer.thetime() - timer.end)

        _batch_size = len(meta)
        target = []
        for _ in range(_batch_size):
            target.extend(meta[_]["labels"])
        target = torch.from_numpy(np.array(target))
        input = input.view(
            _batch_size * 3,
            input.shape[2],
            input.shape[3],
            input.shape[4],
            input.shape[5],
        )
        metric_feat, cls_logit, reg_logit, word_logit = model(
            input, target, temperature=0.1
        )
        ce_loss = ce_loss_criterion(cls_logit.cuda(), target.long().cuda())
        reg_loss = ce_loss_criterion(reg_logit.cuda(), target.long().cuda())
        word_loss = ce_loss_criterion(word_logit.cuda(), target.long().cuda())
        loss = ce_loss + reg_loss + word_loss

        loss.backward()
        loss_meter.update(loss.item())
        ce_loss_meter.update(ce_loss.item())
        reg_loss_meter.update(reg_loss.item())
        word_loss_meter.update(word_loss.item())
        if i % args.accum_grad == args.accum_grad - 1:
            optimizer.step()
            optimizer.zero_grad()

        if i % args.print_freq == 0 and i > 0:
            logger.info(
                "[{0}][{1}/{2}({3})]\t"
                "Dataload_Time={data_time.avg:.3f}\t"
                "Loss={loss.avg:.4f}\t"
                "CELoss={ce_loss.avg:.4f}\t"
                "RegLoss={reg_loss.avg:.4f}\t"
                "WordLoss={word_loss.avg:.4f}\t"
                "LR={cur_lr:.7f}\t"
                "bestAP={ap:.3f}".format(
                    epoch,
                    i,
                    len(loader),
                    len(loader),
                    data_time=data_time,
                    loss=loss_meter,
                    ce_loss=ce_loss_meter,
                    reg_loss=reg_loss_meter,
                    word_loss=word_loss_meter,
                    ap=args.best_score,
                    cur_lr=cur_lr,
                )
            )
            loss_meter.reset()
            ce_loss_meter.reset()


def train_va(loader, model, optimizer, epoch, args):
    timer = Timer()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    reg_loss_meter = AverageMeter()
    cur_lr = adjust_learning_rate(args.lr_decay_rate, optimizer, epoch)
    model.train()
    optimizer.zero_grad()
    ce_loss_criterion = nn.CrossEntropyLoss()
    for i, (input, meta) in tqdm(enumerate(loader), desc="Train Epoch"):
        if args.debug and i >= debug_short_train_num:
            break
        data_time.update(timer.thetime() - timer.end)

        _batch_size = len(meta)
        target = []
        for _ in range(_batch_size):
            target.extend(meta[_]["labels"])
        target = torch.from_numpy(np.array(target))
        input = input.view(
            _batch_size * 3,
            input.shape[2],
            input.shape[3],
            input.shape[4],
            input.shape[5],
        )
        metric_feat, cls_logits, reg_logits = model(
            input, target, temperature=0.1, mv=args.moving_average
        )
        ce_loss = ce_loss_criterion(cls_logits.cuda(), target.long().cuda())
        register_loss = ce_loss_criterion(
            reg_logits.cuda(), target.long().cuda()
        )
        loss = ce_loss + register_loss

        loss.backward()
        loss_meter.update(loss.item())
        ce_loss_meter.update(ce_loss.item())
        reg_loss_meter.update(register_loss.item())
        if i % args.accum_grad == args.accum_grad - 1:
            optimizer.step()
            optimizer.zero_grad()

        if i % args.print_freq == 0 and i > 0:
            logger.info(
                "[{0}][{1}/{2}({3})]\t"
                "Dataload_Time={data_time.avg:.3f}\t"
                "Loss={loss.avg:.4f}\t"
                "CELoss={ce_loss.avg:.4f}\t"
                "RegLoss={reg_loss.avg:.4f}\t"
                "LR={cur_lr:.7f}\t"
                "bestAP={ap:.3f}".format(
                    epoch,
                    i,
                    len(loader),
                    len(loader),
                    data_time=data_time,
                    loss=loss_meter,
                    ce_loss=ce_loss_meter,
                    reg_loss=reg_loss_meter,
                    ap=args.best_score,
                    cur_lr=cur_lr,
                )
            )
            loss_meter.reset()
            ce_loss_meter.reset()


def train(loader, model, optimizer, epoch, args):
    timer = Timer()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    cur_lr = adjust_learning_rate(args.lr_decay_rate, optimizer, epoch)
    model.train()
    optimizer.zero_grad()
    ce_loss_criterion = nn.CrossEntropyLoss()
    for i, (input, meta) in tqdm(enumerate(loader), desc="Train Epoch"):
        if args.debug and i >= debug_short_train_num:
            break
        data_time.update(timer.thetime() - timer.end)

        _batch_size = len(meta)
        target = []
        for _ in range(_batch_size):
            target.extend(meta[_]["labels"])
        target = torch.from_numpy(np.array(target))
        input = input.view(
            _batch_size * 3,
            input.shape[2],
            input.shape[3],
            input.shape[4],
            input.shape[5],
        )
        metric_feat, output = model(input)
        ce_loss = ce_loss_criterion(output.cuda(), target.long().cuda())
        loss = ce_loss

        loss.backward()
        loss_meter.update(loss.item())
        ce_loss_meter.update(ce_loss.item())
        if i % args.accum_grad == args.accum_grad - 1:
            optimizer.step()
            optimizer.zero_grad()

        if i % args.print_freq == 0 and i > 0:
            logger.info(
                "[{0}][{1}/{2}({3})]\t"
                "Dataload_Time={data_time.avg:.3f}\t"
                "Loss={loss.avg:.4f}\t"
                "CELoss={ce_loss.avg:.4f}\t"
                "LR={cur_lr:.7f}\t"
                "bestAP={ap:.3f}".format(
                    epoch,
                    i,
                    len(loader),
                    len(loader),
                    data_time=data_time,
                    loss=loss_meter,
                    ce_loss=ce_loss_meter,
                    ap=args.best_score,
                    cur_lr=cur_lr,
                )
            )
            loss_meter.reset()
            ce_loss_meter.reset()


def main():
    args = parse()
    set_gpu(args.gpu)
    args.best_score = 0
    args.best_result_dict = {}

    from dataloader_baseline import get_my_dataset

    train_loader = get_my_dataset(args)
    args.semantic_mem = train_loader.dataset.semantic_mem
    seed(args.manual_seed)
    model = get_model(args)

    if args.evaluate:
        logger.info(vars(args))
        assert args.test_load is not None
        saved_dict = torch.load(args.test_load)
        logger.warning("loading weight {}".format(args.test_load))
        model.load_state_dict(saved_dict["state_dict"], strict=True)
        args.read_cache_feat = True
        score_dict = do_eval(args=args, model=model)
        return

    logger.warning("using {}".format(args.optimizer))
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), args.lr, weight_decay=args.wd
        )
    else:
        assert False, "invalid optimizer"

    model_summary(model)
    optimizer_summary(optimizer)

    logger.info(vars(args))

    for epoch in range(args.epochs):
        if args.method == "baseline":
            train(train_loader, model, optimizer, epoch, args)
        elif args.method == "va":
            train_va(train_loader, model, optimizer, epoch, args)
        elif args.method == "vasa":
            train_vasa(train_loader, model, optimizer, epoch, args)
        elif args.method == "ranking":
            train_ranking(train_loader, model, optimizer, epoch, args)
        else:
            raise
        if (epoch % eval_per == 0 and epoch > 0) or epoch == args.epochs - 1:
            score_dict = do_eval(args=args, model=model)

            score = score_dict["ap"]
            is_best = score > args.best_score
            if is_best:
                # args.best_result_dict = score_dict
                args.best_score = max(score, args.best_score)
                logger.warning("saving best snapshot..")
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "score": args.best_score,
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(logger.get_logger_dir(), "best.pth.tar"),
                )

    weigth_path = os.path.join(logger.get_logger_dir(), "best.pth.tar")
    saved_dict = torch.load(weigth_path)
    logger.warning(
        "loading weight {}, best validation result={}".format(
            weigth_path, saved_dict["score"]
        )
    )
    model.load_state_dict(saved_dict["state_dict"], strict=True)
    args.eval_split = "testing"
    logger.info(vars(args))
    score_dict = do_eval(args=args, model=model)
    logger.info(
        "training finish. snapshot weight in {}".format(logger.get_logger_dir())
    )


def pdbmain():
    try:
        main()
    except BdbQuit:
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print("shit always happens")
        pdb.post_mortem()
        sys.exit(1)


if __name__ == "__main__":
    main()
