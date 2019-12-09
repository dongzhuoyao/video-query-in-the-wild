import json
import numpy as np
import random
from tqdm import tqdm

saliency_ratio = 2
resolution = 100
noisy_label = "noisy_activity"
noise_video_scan_stride = 10 * resolution  # unit: sec
noise_video_length_random = [10 * resolution, 180 * resolution]  # unit: sec
activity_minimal_sec = 3 * resolution

data = json.load(open('./activity_net.v1-3.min.json'))

video_segment_list = []
class_label_set = set()
for video_id, video in tqdm(data['database'].items()):
    duration = int(video['duration'] * resolution)
    subset = video['subset']
    if subset == "testing":
        continue  # skip test data, we don't have gt!
    annotations = video['annotations']
    segment_list = []

    discretized_utilization_flag = np.zeros((duration, 1))  # zero means not used, one means used.
    label_indicator_dict = {}
    for annotation in annotations:
        label = annotation['label']
        class_label_set.add(label)
        seg_start, seg_end = annotation['segment']
        seg_start = max(int(seg_start * resolution), 0)  # for defect data annotation
        seg_end = min(int(seg_end * resolution), duration)  # for defect data annotation
        discretized_utilization_flag[seg_start:seg_end] = 1
        segment_list.append([seg_start, seg_end])
        label_indicator_dict[(seg_start + seg_end) // 2] = label  # save the mid time dict, use it later label obtain

    # add background
    sorted(segment_list)  # order by segment_s of every segment
    for seg_start, seg_end in segment_list:
        if seg_start == seg_end: continue  # bad data in ActivityNet!
        if seg_end - seg_start < activity_minimal_sec: continue  # too short,skip!

        # corner case, marginal padding.
        padding_left = min(seg_start - 0, random.randint(0, int(
            (seg_end - seg_start) * saliency_ratio)))  # dongzhuoyao, should remove noisy background here.
        padding_right = min(duration - seg_end - 1, random.randint(0, int((seg_end - seg_start) * saliency_ratio)))

        try:
            iter_time = 0
            while np.sum(discretized_utilization_flag[
                         seg_start - padding_left:seg_start]) > 0:  # gradually find unused background noise segment
                padding_left = padding_left // 2
                if iter_time > 4:  # if still cannot find it, just directly don't use any noise background segment.
                    padding_left = 0
                    break
                iter_time += 1

            iter_time = 0
            while np.sum(discretized_utilization_flag[seg_end: seg_end + padding_right]) > 0:
                padding_right = padding_right // 2
                if iter_time > 4:
                    padding_right = 0
                    break
                iter_time += 1
        except:
            raise
            print("sth wrong")

        if np.sum(discretized_utilization_flag[seg_end:seg_end + padding_right]) > 0 or np.sum(
                discretized_utilization_flag[seg_start - padding_left:seg_start]) > 0:
            continue  # current area is already occupied by others, skip it.

        # update flag
        discretized_utilization_flag[seg_start - padding_left:seg_end + padding_right] = 1
        video_segment_list.append(dict(
            border=[(seg_start - padding_left) / resolution, (seg_end + padding_right) / resolution],
            segment=[seg_start / resolution, seg_end / resolution],  # apply resolution, convert to seconds.
            label=label_indicator_dict[(seg_start + seg_end) // 2],
            video_id=video_id,
            activitynet_duration=duration / resolution,
            activitynet_subset=subset,
        ))

    # add pure-noise video from unused segments
    for i in range(0, discretized_utilization_flag.shape[0], noise_video_scan_stride):
        current_noise_length = random.randint(noise_video_length_random[0], noise_video_length_random[1])
        if i + current_noise_length >= discretized_utilization_flag.shape[0]:
            continue  # out of the border of this activitynet video, skip it.
        if np.sum(discretized_utilization_flag[i:i + current_noise_length]) == 0:
            # print("find a noise video")
            discretized_utilization_flag[i:i + current_noise_length] = 1
            video_segment_list.append(dict(
                border=[i / resolution, (i + current_noise_length) / resolution],
                segment=[i / resolution, (i + current_noise_length) / resolution],#main field
                label=noisy_label,
                video_id=video_id,
                activitynet_duration=duration / resolution,
                activitynet_subset=subset,
            ))

noise_video_num = 0
for x in video_segment_list:
    if x['label'] == noisy_label:
        noise_video_num += 1

print("total_video num={}, normal num={}, noise num={}, noise ratio={}%".format(len(video_segment_list),
                                                                                len(video_segment_list) - noise_video_num,
                                                                                noise_video_num,
                                                                                noise_video_num * 100.0 / len(
                                                                                    video_segment_list)))
_ = ["'{}'".format(tmp) for tmp in sorted(list(class_label_set))]
print(",".join(_))
with open('video_segment.json', 'w') as f:
    json.dump(video_segment_list, f)
