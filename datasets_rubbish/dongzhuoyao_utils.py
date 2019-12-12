import torch,os
import torchvision.transforms as transforms
import numpy as np
from datasets.utils import default_loader
import getpass

fps = 3
activtynet_fps3_path  = '/home/tao/dataset/v1-3/mkv_train_val_frames_3'


username = getpass. getuser()
activtynet_fps3_path = activtynet_fps3_path.replace("tao",username)

json_path = "/home/tao/lab/video-query-in-the-wild/data_generate/arv_db.json"
json_path = json_path.replace("tao",username)


json_path_v2 = "/home/tao/lab/video-query-in-the-wild/data_generate/arv_db_100_20_80.json"
json_path_v2 = json_path_v2.replace("tao",username)

json_path_1002080 = "/home/tao/lab/video-query-in-the-wild/data_generate/arv_db_100_20_80.json"
json_path_1002080 = json_path_1002080.replace("tao",username)

noisy_label = "distractor_activity"

def read_activitynet(video_dict):
        frame_duration_num = int((video_dict['segment'][1] - video_dict['segment'][0])*fps)
        start_frame_idx = int(video_dict['segment'][0]*fps)
        activitynet_subset = video_dict['activitynet_subset']
        video_id = video_dict['video_id']
        frame_path = os.path.join(activtynet_fps3_path,activitynet_subset,video_id)
        activitynet_frame_num = len(os.listdir(frame_path))
        return start_frame_idx, frame_duration_num, frame_path, activitynet_frame_num


def read_video(start_frame_idx, gt_frame_num, train_frame_num, video_transform, frame_path, activitynet_frame_num):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        def read_img(loc):
            ii = int(np.floor(loc))
            ii = 1 if ii <= 0 else ii  # make sure start from 1
            ii = activitynet_frame_num if ii > activitynet_frame_num else ii  # make sure inside!
            path = os.path.join(frame_path, 'image_{:05d}.jpg'.format(ii))
            try:
                img = default_loader(path)
            except Exception as e:
                print('failed to load image {}'.format(path))
                print(e)
                raise
            # img = resize(img)
            img = transforms.ToTensor()(img)
            img = normalize(img)
            return img

        images = []
        if gt_frame_num < train_frame_num:
            for i in range(train_frame_num):#repeat the frame from start if too less frame
                idd = i % gt_frame_num
                idd = start_frame_idx+idd+1 # start from 1
                img = read_img(idd)
                images.append(img)
        else:#just take average
            for loc in np.linspace(start_frame_idx, start_frame_idx + gt_frame_num - 1, num=train_frame_num):
                img = read_img(loc)
                images.append(img)

        images = torch.stack(images).permute(0, 2, 3, 1).numpy()  # BCWH->BWHC
        if video_transform is not None:
            images = video_transform(images)
        return images