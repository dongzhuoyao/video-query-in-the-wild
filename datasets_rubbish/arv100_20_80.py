""" Video loader for the Charades dataset """
import torch, os
import torchvision.transforms as transforms
import numpy as np
from glob import glob
from datasets.utils import default_loader
import datasets.video_transforms as videotransforms
from tqdm import tqdm
import random, json
import torch.utils.data as data
from data_generate.activitynet_label_100_20_80 import (
    arv_train_label,
    arv_test_label,
    arv_val_label,
    activitynet_label_list,
    json_path,
)
from .dongzhuoyao_utils import (
    fps,
    noisy_label,
    activtynet_fps3_path,
    read_video,
    read_activitynet,
)


class ARV(data.Dataset):
    def __init__(self, args, transform, input_size, target_transform=None):
        self.num_classes = len(activitynet_label_list)
        self.transform = transform
        self.target_transform = target_transform
        self.input_size = input_size
        self.fps = fps
        self.train_frame = args.train_frame
        self.args = args
        self.split = "training"
        self.novel_img_num = args.novel_img_num
        self.no_novel = (
            False if self.args.no_novel == None else self.args.no_novel
        )

        self._data = self.load_data()
        self.sanity_check()
        random.shuffle(self.data_list[self.split])

    def sanity_check(self):
        _new_list = []
        _removed_dict = {}
        for d in self.data_list[self.split]:
            activitynet_subset = d["activitynet_subset"]
            video_id = d["video_id"]
            # if video_id != "2YSsqivrvR4":continue
            if os.path.isdir(
                os.path.join(activtynet_fps3_path, activitynet_subset, video_id)
            ):
                _new_list.append(d)
                continue
            _removed_dict[video_id] = "shit"
        print(
            "sanity check, removing {} items, keep {} items".format(
                len(list(_removed_dict.keys())), len(_new_list)
            )
        )
        self.data_list["training"] = _new_list

    def load_data(self):
        self.data_dict = json.load(open(json_path))
        self.data_list = dict(
            training=list(), testing=list(), validation=list()
        )
        self.cls2int = {
            label: i for i, label in enumerate(activitynet_label_list)
        }

        for k, v in self.data_dict[self.split].items():
            if (
                k in arv_val_label or k in arv_test_label
            ):  # only keep minimal novel class
                self.data_list[self.split].extend(v[: self.novel_img_num])
            else:
                self.data_list[self.split].extend(v)
        _new_list = []

        cls_label_list = set()
        for d in self.data_list[self.split]:
            if d["label"] == noisy_label:
                continue
            if self.no_novel and d["retrieval_type"] == "novel":
                continue
            _new_list.append(d)
            cls_label_list.add(d["label"])
        self.data_list[self.split] = _new_list  # remove noise video.
        cls_label_list = list(cls_label_list)
        self.cls2int = {label: i for i, label in enumerate(cls_label_list)}
        assert (
            len(list(self.cls2int.keys())) == self.args.nclass
        ), "{} not equal {}".format(
            len(list(self.cls2int.keys())), self.args.nclass
        )

        print("load data done. #class_num={}".format(len(cls_label_list)))

    def __getitem__(self, index):
        meta = {}
        meta["do_not_collate"] = True
        cur_video = self.data_list[self.split][index]
        assert cur_video["label"] != noisy_label
        start_frame_idx, frame_num, frame_path, activitynet_frame_num = read_activitynet(
            cur_video
        )
        images = read_video(
            frame_path=frame_path,
            start_frame_idx=start_frame_idx,
            gt_frame_num=frame_num,
            train_frame_num=self.train_frame,
            video_transform=self.transform,
            activitynet_frame_num=activitynet_frame_num,
        )

        target = self.cls2int[cur_video["label"]]
        return images, target, meta

    def __len__(self):
        if "training" in self.split:
            return len(self.data_list[self.split])
        else:
            raise

    @classmethod
    def get(cls, args):
        input_size = args.input_size
        train_dataset = cls(
            args,
            transform=transforms.Compose(
                [
                    videotransforms.RandomCrop(input_size),
                    # videotransforms.RandomHorizontalFlip()
                ]
            ),
            input_size=input_size,
        )
        return train_dataset


if __name__ == "__main__":
    print("aa")
    train_dataset, val_dataset = ARV.get(None)

    for d in train_dataset:
        print("a")
