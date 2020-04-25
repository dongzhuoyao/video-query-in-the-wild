import json
import numpy as np

input_json = "video_segment.json"
from activitynet_label_120_20_30_unseen30 import (
    activitynet_label_list,
    arv_val_label,
    arv_test_label,
    arv_train_label,arv_unseen_label,
json_name,
moment_json_name
)
output_json = moment_json_name


activitynet_data = json.load(open("./activity_net.v1-3.min.json"))

query_list = []
gallery_list = []
segment_duration_list_4statistics = []

for video_id, annotations in activitynet_data["database"].items():
    activitynet_subset = annotations["subset"]
    if activitynet_subset != "validation":
        continue  # only consider testing data in ActivityNet.
    duration = annotations["duration"]
    gallery_list.append(
        dict(
            video_id=video_id,
            segment=[0, duration],  # used for reading whole long video
            border=[0, duration],  # not used
            activitynet_duration=duration,
            activitynet_subset=activitynet_subset,
            annotations=annotations["annotations"],
        )
    )
    for _seg in annotations["annotations"]:
        label = _seg["label"]
        segment = _seg["segment"]
        if segment[1] - segment[0] <= 10:
            continue  # bad data
        segment_duration_list_4statistics.append(segment[1] - segment[0])
        query_dict = dict(
            video_id=video_id,
            label=label,
            segment=segment,
            border=segment,
            activitynet_duration=duration,
            activitynet_subset=activitynet_subset,
        )
        if label in arv_test_label or label in arv_val_label:
            query_dict["retrieval_type"] = "novel"
        elif label in arv_train_label:
            query_dict["retrieval_type"] = "base"
        elif label in arv_unseen_label:
            query_dict["retrieval_type"] = "unseen"
        else:raise
            #assert label in arv_val_label
        query_list.append(query_dict)

print(
    "query number={}, gallery number={}".format(
        len(query_list), len(gallery_list)
    )
)
duration_list = [g["activitynet_duration"] for g in gallery_list]
print(
    "average video duration in gallery= {} seconds".format(
        sum(duration_list) / len(duration_list)
    )
)
print(
    "average activity duration in gallery= {} seconds".format(
        sum(segment_duration_list_4statistics)
        / len(segment_duration_list_4statistics)
    )
)

with open(output_json, "w") as f:
    json.dump(dict(query=query_list, gallery=gallery_list), f)


##################Annalyis Part#######################
def find_closest_moment(gallery, defined_clip_sec, defined_clips_per_moment):
    """
    Notice: this statistics don't consider the case that
    """

    def cal_iou(min1, max1, min2, max2):
        overlap = max(0, min(max1, max2) - max(min1, min2))
        try:
            return overlap * 1.0 / (max(max2, max1) - min(min1, min2))
        except:
            print("a")

    ious = []
    for g in gallery:
        for segment in g["annotations"]:
            clip_start_idx = segment["segment"][0] * 1.0 / defined_clip_sec
            clip_end_idx = segment["segment"][1] * 1.0 / defined_clip_sec
            if clip_start_idx == clip_end_idx:
                continue  # bad data annotation in ActivityNet, eg:0.0==0.0,26.54==26.54
            if segment["label"] in arv_val_label:
                continue  # only consider testing data in ActivityNet
            if (
                clip_end_idx - clip_start_idx < defined_clips_per_moment
            ):  # grip search, most cases fall in this condition
                iou = max(
                    cal_iou(
                        clip_start_idx,
                        clip_end_idx,
                        int(clip_start_idx),
                        int(clip_end_idx),
                    ),
                    cal_iou(
                        clip_start_idx,
                        clip_end_idx,
                        int(clip_start_idx) + 1,
                        int(clip_end_idx),
                    ),
                    cal_iou(
                        clip_start_idx,
                        clip_end_idx,
                        int(clip_start_idx),
                        int(clip_end_idx) + 1,
                    ),
                    cal_iou(
                        clip_start_idx,
                        clip_end_idx,
                        int(clip_start_idx) + 1,
                        int(clip_end_idx) + 1,
                    ),
                )
            else:  # > or =
                iou = (
                    defined_clips_per_moment
                    * 1.0
                    / (clip_end_idx - clip_start_idx)
                )
            assert iou <= 1
            ious.append(iou)
    return ious


# clip, moment statistics.
clip_sec = [5, 7, 9]
max_clip_per_moment = [14, 20, 26]
print("clip moment statistics")
for c in clip_sec:
    for m in max_clip_per_moment:
        ious = find_closest_moment(
            defined_clip_sec=c, defined_clips_per_moment=m, gallery=gallery_list
        )
        npos = len(ious)
        ious = np.array(ious)
        np.sum((ious > 0.5)) * 1.0 / npos
        print(
            "iou0.5={}, iou0.7={}, clip_sec={}, max_clip_per_moment={}".format(
                np.sum((ious > 0.5)) * 1.0 / npos,
                np.sum((ious > 0.7)) * 1.0 / npos,
                c,
                m,
            )
        )

# query statistics and gallery statistics
duration_list = list()
for q in query_list:
    duration = q["segment"][1] - q["segment"][0]
    duration_list.append(duration)

print(
    np.histogram(
        np.array(duration_list),
        bins=[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260],
    )
)

gallery_duration_list = []
for g in gallery_list:
    duration = g["segment"][1] - g["segment"][0]
    gallery_duration_list.append(duration)

print(
    np.histogram(
        np.array(gallery_duration_list),
        bins=[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260],
    )
)

print("done")
