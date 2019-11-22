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
from data_generate.activitynet_label import arv_train_label,arv_test_label,arv_val_label,activitynet_label_list
import getpass


from .dongzhuoyao_utils import fps,noisy_label,activtynet_fps3_path,json_path,read_video,read_activitynet



class arvtripletmixup(data.Dataset):
    def __init__(self, args,transform, input_size,target_transform=None):
        self.num_classes = len(activitynet_label_list)
        self.cls2int = {label:i for i,label in enumerate(activitynet_label_list)}
        self.cls_ids = list(self.cls2int.values())
        self.transform = transform
        self.target_transform = target_transform
        self.input_size = input_size
        self.fps = fps
        self.train_frame = args.train_frame
        self.args = args
        self.split = "training"
        self.novel_img_num = args.novel_img_num
        self.mixup_positive =args.mixup_positive

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




    def __getitem__(self, index):

        anchor_class_name = random.choice(list(set(activitynet_label_list)-set(list(noisy_label))))
        negative_class_name = random.choice(list(set(activitynet_label_list)-set(list(anchor_class_name))))
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

        anchor = random.choice(self.data_dict[self.split][anchor_class_name])
        negative = random.sample(self.data_dict[self.split][negative_class_name],1)[0]

        anchor = _read(anchor)[np.newaxis, :]
        negative = _read(negative)[np.newaxis, :]
        if self.mixup_positive != -1:
            positive = self.mixup_positive *anchor+(1-self.mixup_positive )*negative
        else:
            random_mix = random.uniform(0, 1)
            if random_mix == 0:
                random_mix = 0.5
            positive = random_mix * anchor + (1 - random_mix) * negative

        triple = np.concatenate([anchor,positive,negative],axis=0)
        return triple, meta

    def __len__(self):

        return self.length

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

