import json
import numpy as np
import matplotlib.pyplot as plt

resolution = 100
input_json = "video_segment.json"
#output_json = 'arv_db_untrimmed.json'
output_json = 'arv_db_120_20_60_untrimmed.json'
minimal_sample_per_novel_class = 5
noisy_label = "noisy_activity"
validation_sample_per_class = 10

from activitynet_label_120_20_60 import activitynet_label_list,arv_val_label,arv_test_label,arv_train_label

activitynet_data = json.load(open('./activity_net.v1-3.min.json'))

query_list = []
gallery_list = []

segment_duration_list = []
for video_id, annotations in activitynet_data['database'].items():
        activitynet_subset = annotations['subset']
        if activitynet_subset!="validation":
            continue
        duration = annotations['duration']
        gallery_list.append(dict(
                    video_id=video_id,
                    segment=[0,duration],
                    border=[0,duration],
                    activitynet_duration = duration,
                    activitynet_subset = activitynet_subset,
                    annotations = annotations['annotations'],
                ))
        for _seg in annotations['annotations']:
            label = _seg['label']
            segment = _seg['segment']
            segment_duration_list.append(segment[1]-segment[0])
            if segment[1]-segment[0]<=10:continue#bad data
            query_dict = dict(
                video_id=video_id,
                label=label,
                segment=segment,
                border=segment,
                activitynet_duration=duration,
                activitynet_subset=activitynet_subset,
            )
            if label in arv_test_label:
                query_dict['retrieval_type']="novel"
                query_list.append(query_dict)
            elif label in arv_train_label:
                query_dict['retrieval_type']="base"
                query_list.append(query_dict)
            else:
                assert label in arv_val_label

query_list = query_list[:]
gallery_list = gallery_list[:]
print("query number={}, gallery number={}".format(len(query_list),len(gallery_list)))
duration_list = [g['activitynet_duration'] for g in gallery_list]
print("average video duration in gallery= {} seconds".format(sum(duration_list)/len(duration_list)))
print("average activity duration in gallery= {} seconds".format(sum(segment_duration_list)/len(segment_duration_list)))

with open(output_json, 'w') as f:
    json.dump(dict(query=query_list,gallery=gallery_list), f)

def find_closest_moment(c,m,gallery):
    def cal_iou(min1, max1, min2, max2):
        #assert s1<=e2 and s2<=e1
        overlap = max(0, min(max1, max2) - max(min1, min2))
        return overlap*1.0/(max(max2,max1) - min(min1,min2))

    ious = []
    for g in gallery:
        duration = g['activitynet_duration']
        scattered = duration*1.0/c
        for seg in g['annotations']:
                start = seg['segment'][0]*1.0/c
                end = seg['segment'][1]*1.0/c
                if seg['label']  in arv_val_label:
                    continue
                if end-start <= m:
                    if start == end:
                        continue#bad data
                    iou = max(
                        cal_iou(start, end,  int(start),int(end)),
                        cal_iou(start, end, int(start)+1, int(end)),
                        cal_iou(start, end, int(start), int(end)+1),
                        cal_iou(start, end, int(start)+1, int(end)+1),
                    )
                else:
                    iou = m*1.0/(end-start)
                assert iou <= 1
                ious.append(iou)
    return ious

#clip, moment statistics.
clip_sec = [5, 7, 9]
max_clip_per_moment = [14,20,26]
print("clip moment statistics")
for c in clip_sec:
    for m in max_clip_per_moment:
        ious = find_closest_moment(c=c,m=m,gallery=gallery_list)
        npos = len(ious)
        ious = np.array(ious)
        np.sum((ious>0.5))*1.0/npos
        print("iou0.5={}, iou0.7={}, clip_sec={}, max_clip_per_moment={}".format(np.sum((ious>0.5))*1.0/npos,
                                                                      np.sum((ious > 0.7)) * 1.0 / npos,
                                                                      c,m))

#query statistics and gallery statistics
duration_list = list()
for q in query_list:
    duration = q['segment'][1] - q['segment'][0]
    duration_list.append(duration)

print(np.histogram(np.array(duration_list), bins=[0, 20, 40,  60,  80,  100,  120, 140, 160, 180, 200, 220, 240, 260]))


gallery_duration_list = []
for g in gallery_list:
    duration = g['segment'][1] - g['segment'][0]
    gallery_duration_list.append(duration)

print(np.histogram(np.array(gallery_duration_list), bins=[0, 20, 40,  60,  80,  100,  120, 140, 160, 180, 200, 220, 240, 260]))



print("done")






