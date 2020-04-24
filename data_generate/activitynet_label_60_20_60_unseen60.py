from data_generate.activitynet_label import activitynet_label_list
import random,os


random.seed(620)
random.shuffle(activitynet_label_list)
arv_train_label = activitynet_label_list[:60]
arv_val_label = activitynet_label_list[60:80]
arv_test_label = activitynet_label_list[80:140]
arv_unseen_label = activitynet_label_list[140:]

short_name = "60_20_60_unseen60"
json_name = "arv_db_{}.json".format(short_name)
moment_json_name  = "arv_db_{}_untrimmed.json".format(short_name)

json_path = os.path.join("data_generate",json_name)
moment_eval_json_path = os.path.join("data_generate",moment_json_name)


