import torch, os
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import getpass
import data_generate

fps = 3
noisy_label = "distractor_activity"
activtynet_fps3_path = "data/activitynet1.3_train_val_frames_fps3"


dataset_config = {
    "100_20_80": dict(
        arv_train_label=data_generate.activitynet_label_100_20_80.arv_train_label,
        arv_test_label=data_generate.activitynet_label_100_20_80.arv_test_label,
        arv_val_label=data_generate.activitynet_label_100_20_80.arv_val_label,
        activitynet_label_list=data_generate.activitynet_label_100_20_80.activitynet_label_list,
        json_path=data_generate.activitynet_label_100_20_80.json_path,
        moment_eval_json_path=data_generate.activitynet_label_100_20_80.moment_eval_json_path,
    ),
    "120_20_60": dict(
        arv_train_label=data_generate.activitynet_label_120_20_60.arv_train_label,
        arv_test_label=data_generate.activitynet_label_120_20_60.arv_test_label,
        arv_val_label=data_generate.activitynet_label_120_20_60.arv_val_label,
        activitynet_label_list=data_generate.activitynet_label_120_20_60.activitynet_label_list,
        json_path=data_generate.activitynet_label_120_20_60.json_path,
        moment_eval_json_path=data_generate.activitynet_label_120_20_60.moment_eval_json_path,
    ),
    "80_20_100": dict(
        arv_train_label=data_generate.activitynet_label_80_20_100.arv_train_label,
        arv_test_label=data_generate.activitynet_label_80_20_100.arv_test_label,
        arv_val_label=data_generate.activitynet_label_80_20_100.arv_val_label,
        activitynet_label_list=data_generate.activitynet_label_80_20_100.activitynet_label_list,
        json_path=data_generate.activitynet_label_80_20_100.json_path,
        moment_eval_json_path=data_generate.activitynet_label_80_20_100.moment_eval_json_path,
    ),
}


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # for _ in range(5):
    #    try:
    #        with open(path, 'rb') as f:
    #            img = Image.open(f)
    #            return img.convert('RGB')
    #    except IOError as e:
    #        print(e)
    #        print('waiting 5 sec and trying again')
    #        time.sleep(5)
    # raise IOError
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


def read_activitynet(video_dict):
    frame_duration_num = int(
        (video_dict["segment"][1] - video_dict["segment"][0]) * fps
    )
    start_frame_idx = int(video_dict["segment"][0] * fps)
    activitynet_subset = video_dict["activitynet_subset"]
    video_id = video_dict["video_id"]
    frame_path = os.path.join(
        activtynet_fps3_path, activitynet_subset, video_id
    )
    activitynet_frame_num = len(os.listdir(frame_path))
    return (
        start_frame_idx,
        frame_duration_num,
        frame_path,
        activitynet_frame_num,
    )


def read_video(
    start_frame_idx,
    gt_frame_num,
    train_frame_num,
    video_transform,
    frame_path,
    activitynet_frame_num,
):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    def read_img(loc):
        ii = int(np.floor(loc))
        ii = 1 if ii <= 0 else ii  # make sure start from 1
        ii = (
            activitynet_frame_num if ii > activitynet_frame_num else ii
        )  # make sure inside!
        path = os.path.join(frame_path, "image_{:05d}.jpg".format(ii))
        try:
            img = default_loader(path)
        except Exception as e:
            print("failed to load image {}".format(path))
            print(e)
            raise
        # img = resize(img)
        img = transforms.ToTensor()(img)
        img = normalize(img)
        return img

    images = []
    if gt_frame_num < train_frame_num:
        for i in range(
            train_frame_num
        ):  # repeat the frame from start if too less frame
            idd = i % gt_frame_num
            idd = start_frame_idx + idd + 1  # start from 1
            img = read_img(idd)
            images.append(img)
    else:  # just take average
        for loc in np.linspace(
            start_frame_idx,
            start_frame_idx + gt_frame_num - 1,
            num=train_frame_num,
        ):
            img = read_img(loc)
            images.append(img)

    images = torch.stack(images).permute(0, 2, 3, 1).numpy()  # BCWH->BWHC
    if video_transform is not None:
        images = video_transform(images)
    return images
