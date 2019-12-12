""" Video loader for the Charades dataset """
import torch,os
import torchvision.transforms as transforms
import numpy as np
from glob import glob
from datasets.utils import default_loader
import datasets.video_transforms as videotransforms
from tqdm import tqdm
import random,json
import torch.utils.data as data
from data_generate.activitynet_label_100_20_80 import arv_train_label,arv_test_label,arv_val_label,activitynet_label_list,json_path
from .dongzhuoyao_utils import fps,noisy_label,activtynet_fps3_path,read_video,read_activitynet


class arv1002080tripletclsfsnosample(data.Dataset):
    def __init__(self, args,transform, input_size,target_transform=None):
        self.num_classes = len(activitynet_label_list)
        self.transform = transform
        self.target_transform = target_transform
        self.input_size = input_size
        self.fps = fps
        self.train_frame = args.train_frame
        self.args = args
        self.split = "training"
        self.novel_img_num = args.novel_img_num

        self._data = self.load_data()
        self.sanity_check()
        l = 0
        for cls_name in self.data_dict[self.split]:
            if cls_name in set(activitynet_label_list)-set(list(noisy_label)):
                l += len(self.data_dict[self.split][cls_name])
        self.length = l


    def sanity_check(self):
        for cls_name in self.data_dict[self.split]:
            _new_list = []
            _removed_dict = {}
            for d in self.data_dict[self.split][cls_name]:
                activitynet_subset = d['activitynet_subset']
                video_id = d['video_id']
                #if video_id != "2YSsqivrvR4":continue
                if os.path.isdir(os.path.join(activtynet_fps3_path,activitynet_subset,video_id)):
                    _new_list.append(d)
                    continue
                _removed_dict[video_id]="shit"
            self.data_dict[self.split][cls_name] = _new_list
        print("sanity check, removing {} items".format(len(list(_removed_dict.keys()))))



    def load_data(self):
        self.data_dict = json.load(open(json_path))
        print("load data done.")
        new_dict = {}
        self.cur_label_list = []
        video_list = []        #make a list
        for cls_name, item_list in self.data_dict[self.split].items():
            if cls_name == noisy_label:continue
            if self.novel_img_num> 0 and (cls_name in arv_val_label or cls_name in arv_test_label):#only keep minimal novel class
                new_dict[cls_name] = item_list[:self.novel_img_num]
            elif self.novel_img_num == 0  and (cls_name in arv_val_label or cls_name in arv_test_label):
                continue
            else:
                new_dict[cls_name] = item_list
            self.cur_label_list.append(cls_name)
            video_list.extend(new_dict[cls_name])

        self.data_dict[self.split]=new_dict#remove novel and noisy label
        self.cls2int = {label: i for i, label in enumerate(self.cur_label_list)}
        assert len(list(self.cls2int.keys())) == self.args.nclass, "{} not equal {}".format(
            len(list(self.cls2int.keys())) , self.args.nclass
        )

        #generate whole video list
        self.whole_video_list = video_list




    def __getitem__(self, index):


        meta =  {}
        meta['do_not_collate'] = True

        def _read(video_dict):
            assert  video_dict['label'] != noisy_label
            start_frame_idx, frame_num, frame_path, activitynet_frame_num = read_activitynet(video_dict)
            images = read_video(frame_path=frame_path, start_frame_idx=start_frame_idx,
                                gt_frame_num=frame_num, train_frame_num=self.train_frame,
                                video_transform=self.transform,
                                activitynet_frame_num=activitynet_frame_num)
            return images

        _anchor, _positive, _negative = random.sample(self.whole_video_list,3)

        anchor = _read(_anchor)[np.newaxis, :]
        positive = _read(_positive)[np.newaxis, :]
        negative = _read(_negative)[np.newaxis, :]

        triple = np.concatenate([anchor,positive,negative],axis=0)
        meta['labels']=[self.cls2int[_anchor['label']], self.cls2int[_positive['label']], self.cls2int[_negative['label']]]
        return triple, meta

    def __len__(self):

        return self.length//3

    @classmethod
    def get(cls, args,novel_img_num=5):
        input_size = args.input_size
        train_dataset = cls(
            args,
            transform=transforms.Compose([
                videotransforms.RandomCrop(input_size),
                #videotransforms.RandomHorizontalFlip()
            ]),
            input_size=input_size)
        return train_dataset


if __name__ == '__main__':
    print("aa")
    train_dataset, val_dataset = ARV.get(None,)

    for d in train_dataset:
        print('a')

