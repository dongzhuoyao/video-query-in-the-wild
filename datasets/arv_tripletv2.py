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
import getpass, copy

from .dongzhuoyao_utils import fps,noisy_label,activtynet_fps3_path,json_path,read_video,read_activitynet



class arvtripletv2(data.Dataset):
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
        l = 0
        for cls_name in self.data_dict[self.split]:
            if cls_name in set(activitynet_label_list)-set(list(noisy_label)):
                l += len(self.data_dict[self.split][cls_name])
        self.length = l

        #dongzhuoyao########################################
        self.avail_classes = sorted(list(self.image_dict.keys()))#names,

        # Convert image dictionary from classname:content to class_idx:content, because the initial indices are not necessarily from 0 - <n_classes>.
        self.image_dict = {i: self.image_dict[key] for i, key in enumerate(self.avail_classes)}
        self.avail_classes = sorted(list(self.image_dict.keys()))#ids,

        # Init. properties that are used when filling up batches.
        self.samples_per_class = args.samples_per_class
        # Select current class to sample images from up to <samples_per_class>
        self.current_class = np.random.randint(len(self.avail_classes))
        self.classes_visited = [self.current_class, self.current_class]
        self.n_samples_drawn = 0

        # Convert Image-Dict to list of (image_path, image_class). Allows for easier direct sampling.
        #self.image_list = [[(x, key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        #self.image_list = [x for y in self.image_list for x in y]

        # Flag that denotes if dataset is called for the first time.
        self.is_init = True


    def load_data(self):
        self.data_dict = json.load(open(json_path))
        print("load data done.")
        new_dict = {}
        self.cur_label_list = []
        for cls_name, item_list in self.data_dict[self.split].items():
            if item_list[0]['retrieval_type'] != 'base':
                continue
            new_dict[cls_name] = item_list
            self.cur_label_list.append(cls_name)

        self.image_dict = new_dict
        #self.data_dict[self.split]=new_dict#remove novel and noisy label
        #self.cls2int = {label: i for i, label in enumerate(self.cur_label_list)}
        assert len(list(self.image_dict.keys())) == self.args.nclass





    def __getitem__(self, index):

        ####ddd########
        meta = {}
        meta['do_not_collate'] = True

        def _read(video_dict):
            assert video_dict['label'] != noisy_label
            start_frame_idx, frame_num, frame_path, activitynet_frame_num = read_activitynet(video_dict)
            images = read_video(frame_path=frame_path, start_frame_idx=start_frame_idx,
                                gt_frame_num=frame_num, train_frame_num=self.train_frame,
                                video_transform=self.transform,
                                activitynet_frame_num=activitynet_frame_num)
            return images


        if self.is_init:
            self.current_class = self.avail_classes[index % len(self.avail_classes)]
            self.is_init = False

        if self.samples_per_class == 1:
            raise
            #return self.image_list[index][-1], self.transform(self.ensure_3dim(Image.open(self.image_list[idx][0])))

        if self.n_samples_drawn == self.samples_per_class:
            # Once enough samples per class have been drawn, we choose another class to draw samples from.
            # Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
            # previously or one before that.
            counter = copy.deepcopy(self.avail_classes)
            for prev_class in self.classes_visited:
                if prev_class in counter: counter.remove(prev_class)

            self.current_class = counter[index % len(counter)]
            self.classes_visited = self.classes_visited[1:] + [self.current_class]
            self.n_samples_drawn = 0

        class_sample_idx = index % len(self.image_dict[self.current_class])
        self.n_samples_drawn += 1

        out_img = _read(self.image_dict[self.current_class][class_sample_idx])
        return  out_img,self.current_class, meta

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

