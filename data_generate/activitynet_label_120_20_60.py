from data_generate.activitynet_label import activitynet_label_list
import random

random.seed(620)
random.shuffle(activitynet_label_list)
arv_train_label = activitynet_label_list[:120]
arv_val_label = activitynet_label_list[120:140]
arv_test_label = activitynet_label_list[140:]
short_name = "120_20_60"
json_path = "data_generate/arv_db_{}.json".format(short_name)
moment_eval_json_path = "data_generate/arv_db_{}_untrimmed.json".format(
    short_name
)
