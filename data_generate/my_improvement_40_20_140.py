import json
import numpy as np
import matplotlib.pyplot as plt

resolution = 100
input_json = "video_segment.json"

minimal_sample_per_novel_class = 5
noisy_label = "noisy_activity"
validation_sample_per_class = 10

#from activitynet_label import activitynet_label_list,arv_val_label,arv_test_label,arv_train_label
#output_json = 'arv_db_0830.json'

from activitynet_label_40_20_140 import activitynet_label_list,arv_val_label,arv_test_label,arv_train_label
output_json = 'arv_db_40_20_140.json'

print("train label number={}".format(len(arv_train_label)))
print(", ".join(arv_train_label))
print("validation label number={}".format(len(arv_val_label)))
print(", ".join(arv_val_label))
print("testing label number={}".format(len(arv_test_label)))
print(", ".join(arv_test_label))



activitynet_data = json.load(open('./activity_net.v1-3.min.json'))
data = json.load(open(input_json))
my_dict = dict(training=dict(),validation=dict(),testing=dict())

def _add_item_to_dict(_dict,label,data):
    if label in _dict:
        _dict[label].append(data)
    else:
        _dict[label] = [data]
    return _dict

for d in data:
    if d['activitynet_subset'] == "training":
        my_dict['training'] = _add_item_to_dict(_dict=my_dict['training'],label=d['label'],data=d)
    elif d['activitynet_subset'] == "validation":
        my_dict['testing'] = _add_item_to_dict(_dict=my_dict['testing'],label=d['label'],data=d)
    else:
        raise

#generating validation data and clean up training data.
new_training_list = []



for label in arv_train_label:
    data_list = my_dict['training'][label]
    my_dict['validation'][label] = data_list[:validation_sample_per_class]
    my_dict['training'][label] = data_list[validation_sample_per_class:]
for label in arv_val_label:
    data_list = my_dict['training'][label]
    my_dict['training'][label] = data_list[:minimal_sample_per_novel_class]
    my_dict['validation'][label] = data_list[minimal_sample_per_novel_class:minimal_sample_per_novel_class+validation_sample_per_class]
for label in arv_test_label:
    data_list = my_dict['training'][label]
    my_dict['training'][label] = data_list[:minimal_sample_per_novel_class]
    my_dict['validation'][label] = data_list[minimal_sample_per_novel_class:minimal_sample_per_novel_class+validation_sample_per_class]
for label in [noisy_label]:
    data_list = my_dict['training'][label]
    my_dict['validation'][label] = data_list[:validation_sample_per_class*20]
    my_dict['training'][label] = data_list[validation_sample_per_class*20:]


cur_split = "training"
for cls_dict in my_dict[cur_split].keys():
    _new_listttt = []
    for d in my_dict[cur_split][cls_dict]:
        if d['label'] in arv_train_label:
            d['is_query'] = -1#-1 means useless info
            d['retrieval_type'] = "base"

        elif d['label'] in arv_val_label:
            d['is_query'] = -1
            d['retrieval_type'] = "novel"

        elif d['label'] in arv_test_label:
            d['is_query'] = -1
            d['retrieval_type'] = "novel"

        elif  d['label']==noisy_label:
            d['is_query'] = -1
            d['retrieval_type'] = "noise"

        else:raise
        _new_listttt.append(d)
    my_dict[cur_split][cls_dict] = _new_listttt


cur_split = "validation"
for cls_dict in my_dict[cur_split].keys():
    _new_listttt = []
    for d in my_dict[cur_split][cls_dict]:
        if d['label'] in arv_train_label:
            d['is_query'] = 1
            d['retrieval_type'] = "base"

        elif d['label'] in arv_val_label:
            d['is_query'] = 1
            d['retrieval_type'] = "novel"

        elif d['label'] in arv_test_label:
            d['is_query'] = 0
            d['retrieval_type'] = "novel"

        elif  d['label']==noisy_label:
            d['is_query'] = 0
            d['retrieval_type'] = "noise"
        else:
            raise
        _new_listttt.append(d)
    my_dict[cur_split][cls_dict] = _new_listttt


cur_split = "testing"
for cls_dict in my_dict[cur_split].keys():
    _new_listttt = []
    for d in my_dict[cur_split][cls_dict]:
        if d['label'] in arv_train_label:
            d['is_query'] = 1
            d['retrieval_type'] = "base"

        elif d['label'] in arv_val_label:
            d['is_query'] = 0
            d['retrieval_type'] = "novel"

        elif d['label'] in arv_test_label:
            d['is_query'] = 1
            d['retrieval_type'] = "novel"

        elif  d['label']==noisy_label:
            d['is_query'] = 0
            d['retrieval_type'] = "noise"

        else:
            raise
        _new_listttt.append(d)
    my_dict[cur_split][cls_dict] = _new_listttt

with open(output_json, 'w') as f:
    json.dump(my_dict, f)


####### statistics ##########
avg_video_num_per_class_hist= []
avg_fg_duration_hist = []
avg_fgbg_ratio_hist = []


for _subset in ['training','validation','testing']:
    _dataset_dict = my_dict[_subset]
    total_video_num = sum([len(_) for _ in list(_dataset_dict.values())])
    def _show_class_sample_num():
        for label in arv_train_label:
            print("{}, base_class(from train-label), #{}={}".format(_subset,label,len(_dataset_dict[label])))
        for label in arv_val_label:
            print("{}, novel_class(from val-label), #{}={}".format(_subset,label,len(_dataset_dict[label])))
        for label in arv_test_label:
                print("{}, novel_class(from test-label), #{}={}".format(_subset,label,len(_dataset_dict[label])))
        for label in [noisy_label]:
                print("{}, noisy_class, #{}={}".format(_subset,label,len(_dataset_dict[label])))
    #_show_class_sample_num()

    #avg_video_num_per_class
    l = [len(v) for k, v in _dataset_dict.items()]
    l = np.array(l)
    print("{} avg_video_num_per_class histogram:".format(_subset))
    print(np.histogram(l, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]))
    avg_video_num_per_class_hist.append(l)

    #
    print("{} total=non-noise + noise video, {} = {} + {}".format(_subset, total_video_num, total_video_num-len(_dataset_dict[noisy_label]), len(_dataset_dict[noisy_label])))


    #avg_video_foreground_length
    foreground_length_list = []
    foreground_background_ratio = []
    for label in activitynet_label_list:
        data_list = _dataset_dict[label]
        for data in data_list:
            foreground_length_list.append((data['segment'][1]-data['segment'][0]))
            foreground_background_ratio.append((data['segment'][1]-data['segment'][0])/(data['border'][1]-data['border'][0]))

    print("{} fg_duration(sec) histogram:".format(_subset))
    avg_fg_duration_hist.append(np.array(foreground_length_list))
    print(np.histogram(np.array(foreground_length_list), bins=[0, 20, 40,  60,  80,  100,  120, 140, 160, 180, 200, 220, 240, 260]))
    print("{} fg_bg_ratio histogram:".format(_subset))
    avg_fgbg_ratio_hist.append(np.array(foreground_background_ratio))
    print(np.histogram(np.array(foreground_background_ratio),
                       bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))



fig1 = plt.gcf()
plt.hist(avg_video_num_per_class_hist[0], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],alpha=0.5, label="train")  # arguments are passed to np.histogram
plt.hist(avg_video_num_per_class_hist[2], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],alpha=0.5, label="test")
plt.legend(loc='upper right')
plt.title("avg_video_num_per_class histogram")
plt.show()
fig1.savefig('avg_video_num_per_class.png')


fig1 = plt.gcf()
plt.hist(avg_fg_duration_hist[0], bins=[0, 20, 40,  60,  80,  100,  120, 140, 160, 180, 200, 220, 240, 260],alpha=0.5, label="train")  # arguments are passed to np.histogram
plt.hist(avg_fg_duration_hist[2], bins=[0, 20, 40,  60,  80,  100,  120, 140, 160, 180, 200, 220, 240, 260],alpha=0.5, label="test")
plt.legend(loc='upper right')
plt.title("avg_fg_duration histogram")
plt.show()
fig1.savefig('avg_fg_duration_hist.png')

fig1 = plt.gcf()
plt.hist(avg_fgbg_ratio_hist[0], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],alpha=0.5, label="train")  # arguments are passed to np.histogram
plt.hist(avg_fgbg_ratio_hist[2], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],alpha=0.5, label="test")
plt.legend(loc='upper left')
plt.title("avg_fgbg_ratio_hist histogram")
plt.show()
fig1.savefig('avg_fgbg_ratio_hist.png')



