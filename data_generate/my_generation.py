import json
import numpy as np
import random
from tqdm import tqdm
#python 3


data = json.load(open('./activity_net.v1-3.min.json'))

saliency_ratio = 2
resolution = 100
noisy_label = "noisy_activity"
noise_video_scan_stride  = 10*resolution #unit: sec
noise_video_length_random = [10*resolution, 180*resolution] #unit: sec
activity_minimal_sec = 3*resolution

result_list = []
class_label_dict = {}
for video_id, video in tqdm(data['database'].items()):
    #if video_id != "sCzauf2u4dc":continue

    duration = int(video['duration']*resolution)
    subset = video['subset']
    if subset == "testing":
        continue#skip test data, we don't have gt!
    annotations = video['annotations']
    segment_list = []

    discretized_flag = np.zeros((duration,1))
    label_indicator_dict = {}
    for annotation in annotations:
        label = annotation['label']
        class_label_dict[label] = label
        segment_s, segment_t =annotation['segment']
        segment_s = max(int(segment_s*resolution),0)#for defect data annotation
        segment_t = min(int(segment_t*resolution),duration)#for defect data annotation
        discretized_flag[segment_s:segment_t] = 1
        segment_list.append([segment_s,segment_t])
        label_indicator_dict[(segment_s+segment_t)//2] = label

    # add background
    sorted(segment_list)#order by segment_s of every segment
    for segment_s, segment_t  in segment_list:
        if segment_s == segment_t: continue#bad data in ActivityNet!
        if segment_t - segment_s <activity_minimal_sec:continue#too short,skip!

        padding_left = random.randint(0,int((segment_t - segment_s)*saliency_ratio))
        padding_right = random.randint(0,int((segment_t - segment_s)*saliency_ratio))

        # corner case, marginal padding.
        padding_left = min(segment_s-0,padding_left)
        padding_right= min(duration-segment_t-1, padding_right)

        try:
            iter_time = 0
            while np.sum(discretized_flag[segment_s-padding_left:segment_s]) > 0:
                padding_left = padding_left//2
                if iter_time > 4:
                    padding_left = 0
                    break
                iter_time += 1

            iter_time = 0
            while np.sum(discretized_flag[segment_t: segment_t + padding_right]) > 0:
                padding_right = padding_right//2
                if iter_time > 4:
                    padding_right = 0
                    break
                iter_time += 1
        except:
            raise
            print("ss")

        if np.sum(discretized_flag[segment_t:segment_t+padding_right]) > 0 or np.sum(discretized_flag[segment_s-padding_left:segment_s]) > 0:
            continue#current area is already occupied by others, skip it.

        #update flag
        discretized_flag[segment_s-padding_left:segment_t + padding_right] = 1
        result_list.append(dict(
            border = [(segment_s-padding_left)/resolution, (segment_t + padding_right)/resolution],
            segment = [segment_s/resolution, segment_t/resolution],
            label = label_indicator_dict[(segment_s+segment_t)//2],
            video_id = video_id,
            activitynet_duration = duration/resolution,
            activitynet_subset = subset,
        ))


    # add pure-noise video
    for i in range(0,discretized_flag.shape[0],noise_video_scan_stride):
        current_noise_length = random.randint(noise_video_length_random[0],noise_video_length_random[1])
        if i+current_noise_length >= discretized_flag.shape[0]:
            continue#out of the border of this activitynet video, skip it.
        if np.sum(discretized_flag[i:i+current_noise_length]) == 0:
            #print("find a noise video")
            discretized_flag[i:i + current_noise_length] = 1
            result_list.append(dict(
                border=[i/resolution, (i + current_noise_length)/resolution],
                segment=[i/resolution, (i + current_noise_length)/resolution],
                label=noisy_label,
                video_id = video_id,
                activitynet_duration=duration/resolution,
            activitynet_subset = subset,
            ))


noise_video_num = 0
for x in result_list:
    if x['label'] == noisy_label:
        noise_video_num += 1


print("total_video num={}, normal num={}, noise num={}, noise ratio={}%".format(len(result_list),len(result_list)-noise_video_num, noise_video_num, noise_video_num*100.0/len(result_list)))
_ = ["'{}'".format(tmp) for tmp in list(class_label_dict.keys())]
print(",".join(_))
with open('shit.json', 'w') as f:
     json.dump(result_list, f)
